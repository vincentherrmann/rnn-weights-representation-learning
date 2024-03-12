import torch
import torch.nn.functional as F
import argparse
import socket
import os
import wandb
from matplotlib import pyplot as plt

from training_configs import encoder_configs
from dataset_classes import ParametersFormalLanguageDataset, ParametersSequentialMNISTDataset, task_entropy_criterion
from utilities import (unflatten_params, ModelState, convert_parameter_names, num_params, seed_everything, fix_seed,
                       figure2PIL, MultiMeter, Logger)
from modules_rnn_encoders import (InteractiveEncoder, FlatParameterEncoder, NFLSTMParameterEncoder,
                                  ParameterTransformerEncoder, ParameterStatisticsEncoder)
from modules_rnn_models import RNNFast, RNNFunctionalizable
from dataset_creation_formal_language import RelativeOccurrenceLanguage
from modules_neural_functionals import prepare_lstm_params_for_np
from lstm_neuron_permutation_testing import permuteLSTMNeurons
from evaluation_visualization import (visualize_embedding_space, visualize_probing_inputs_formal_language,
                                      visualize_probing_inputs_sequential_mnist, visualize_probing_outputs,
                                      visualize_output_distributions, evaluate_clones)


def parse_args():
    parser = argparse.ArgumentParser()

    # global
    parser.add_argument("--wandb_logging", type=int, default=0,
                        help="log with wandb if 1, if 0, log on the terminal")
    parser.add_argument("--save_model", type=int, default=0,
                        help="save model if 1, if 0, don't save")
    parser.add_argument("--seed", type=int, default=-1,
                        help="seed of the experiment")
    parser.add_argument("--cuda", type=int, default=1,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--config", type=str, default="nf",
                        help="which pre-defined config to use (from training_configs.py). Empty string means no config is used")

    # dataset
    parser.add_argument("--dataset", type=str, default="formal_languages_tokens_4_length_42_2x32",
                        choices=[
                            "formal_languages_tokens_4_length_42_2x32",
                            "sequential_mnist_tile_4x4_model_2x32"
                        ])
    parser.add_argument("--permutation_augmentation", type=int, default=0,
                        help="if toggled, the the hidden neurons of the lstm with be randomly permuted during training")

    # rnn
    parser.add_argument("--rnn_num_layers", type=int, default=2,
                        help="number of layers of the RNN")
    parser.add_argument("--rnn_hidden_size", type=int, default=32,
                        help="hidden size of the RNN")

    # training
    parser.add_argument("--total_training_steps", type=int, default=100_000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="the weight decay of the optimizer")
    parser.add_argument("--criterion", type=str, default="total_variation",
                        choices=["kl", "kl_reverse", "js_div", "total_variation", "ce", "cosine", "l2"],
                        help="the training criterion to use for the behavior cloning")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout rate")
    parser.add_argument("--gradient_clipping", type=float, default=0.1,
                        help="if not 0, the gradients will be clipped to this value")
    parser.add_argument("--rnn_batch_size", type=int, default=64,
                        help="number of different actors to sample at each training step")
    parser.add_argument("--sequence_batch_size", type=int, default=1,
                        help="number of input-output sequences to sample from each rnn at each training step")
    parser.add_argument("--normalize_embedding", type=int, default=1,
                        help="if toggled, the rnn weight embedding is normalized to have unit norm")
    parser.add_argument("--early_stopping", type=int, default=0,
                        help="if > 0, the training will stop when the ood validation loss does not improve for this number of steps")
    parser.add_argument("--num_evaluation_policies", type=int, default=20,
                        help="number of policies to evaluate at each evaluation step")

    # rnn weight encoder
    parser.add_argument('--encoder_type', default="interactive", type=str, required=False,
                        choices=["interactive", "noninteractive", "flat", "nf", "parameter_transformer",
                                 "layerwise_statistics"],
                        help="""type of rnn weight encoder: 
                        interactive uses recurrent fingerprinting, 
                        flat uses the flattened parameters,
                        nf uses a neural functional architecture""")
    parser.add_argument('--rnn_embedding_size', default=16, type=int, required=False,
                        help="size of the rnn weight embedding")
    parser.add_argument('--encoder_core_num_layers', default=2, type=int, required=False,
                        help="number of layers in the probing rnn")
    parser.add_argument('--encoder_core_hidden_size', default=256, type=int, required=False,
                        help="size of the probing rnn hidden state")
    parser.add_argument('--encoder_residual', default=1, type=int, required=False,
                        help="if toggled, the fingerprinting encoder uses residual connections")
    parser.add_argument('--probing_sequence_length', default=22, type=int, required=False,
                        help="length of the probing state rollout in the interactive encoder")
    parser.add_argument('--num_parallel_probing_sequences', default=1, type=int, required=False,
                        help="number of parallel probing states in the interactive encoder")
    parser.add_argument('--probing_input_softmax', default=1, type=int, required=False,
                        help="if toggled, the probing input distribution is softmaxed")
    parser.add_argument('--encoder_mlp_num_layers', default=1, type=int, required=False,
                        help="number of layers in the encoder mlps")
    parser.add_argument('--encoder_mlp_hidden_size', default=128, type=int, required=False,
                        help="size of the hidden layers in the encoder mlps")
    parser.add_argument('--encoder_num_heads', default=4, type=int, required=False,
                        help="number of heads in the transformer encoder")
    parser.add_argument('--static_weight', default=0., type=float, required=False,
                        help="weight of the static probing inputs in the rnn weight encoder")
    parser.add_argument('--encoder_open_gate_init', default=1, type=int, required=False,
                        help="Whether to initialize the forget gates of the lstm encoder to 1")
    parser.add_argument('--encoder_rnn_type', default="lstm", type=str, required=False,
                        choices=["lstm", "gru"])

    # cloning actor
    parser.add_argument('--emulator_num_layers', default=2, type=int, required=False,
                        help="number of layers in the cloning actor")
    parser.add_argument('--emulator_hidden_size', default=256, type=int, required=False,
                        help="size of the hidden layers in the cloning actor")
    parser.add_argument('--emulator_only_condition_bos', default=1, type=int, required=False,
                        help="if toggled, the cloning actor conditions only the beginning of sequence token on the rnn weight embedding")
    parser.add_argument('--emulator_open_gate_init', default=1, type=int, required=False,
                        help="Whether to initialize the forget gates of the lstm actor to 1")

    args = parser.parse_args()
    return args


def conditional_args(args):
    dataset = args.dataset
    # parse the dataset string so that e.g. "formal_languages_tokens_4_length_42_2x32" leads to
    # dataset_task = "formal_languages", num_tokens = 4, sequence_length = 42, rnn_num_layers = 2, rnn_hidden_size = 32

    if "formal_languages" in dataset:
        args.dataset_task = "formal_languages"
    elif "sequential_mnist" in dataset:
        args.dataset_task = "sequential_mnist"
        args.one_hot_tokens = 0
        args.probing_input_softmax = 0
        args.tile_size = int(dataset.split("_tile_")[1].split("_")[0].split("x")[0])
        tile_length = args.tile_size ** 2
        args.probing_input_dim = tile_length
        args.probing_output_dim = 10
        args.sequence_length = 784 // tile_length
        args.probing_sequence_length = args.sequence_length + 1

    if "tokens" in dataset:
        args.num_tokens = int(dataset.split("_tokens_")[1].split("_")[0])
        args.probing_input_dim = args.num_tokens + 2
        args.probing_output_dim = args.num_tokens + 2
        args.one_hot_tokens = 1
        args.probing_input_softmax = 1

    if "length" in dataset:
        args.sequence_length = int(dataset.split("_length_")[1].split("_")[0])

    if args.encoder_type == "flat" or args.encoder_type == "parameter_transformer":
        args.permutation_augmentation = 1
    else:
        args.permutation_augmentation = 0

    return args


class EmulatorLSTM(torch.nn.Module):
    def __init__(self, rnn_embedding_size, obs_dim, probing_output_dim, hidden_size, num_layers,
                 output_activation=torch.nn.Identity, only_condition_bos=True):
        super().__init__()
        self.embedding_projection = torch.nn.Linear(rnn_embedding_size, hidden_size)
        self.input_projection = torch.nn.Linear(obs_dim, hidden_size)
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_projection = torch.nn.Linear(hidden_size, probing_output_dim)
        self.output_activation = output_activation()
        self.only_condition_bos = only_condition_bos

    def forward(self, input_seq, rnn_embedding, hidden_state=None):
        input_projection = self.input_projection(input_seq)
        if self.only_condition_bos:
            if hidden_state is None:
                rnn_projection = self.embedding_projection(rnn_embedding)
                input_projection[:, 0, :] += rnn_projection
        else:
            rnn_projection = self.embedding_projection(rnn_embedding).unsqueeze(1)
            input_projection = input_projection + rnn_projection

        lstm_output, hidden_state = self.lstm(input_projection, hidden_state)
        output = self.output_activation(self.output_projection(lstm_output))
        return output, hidden_state

    def open_forget_gates(self):
        for l in range(self.lstm.num_layers):
            bias_hh = getattr(self.lstm, f"bias_hh_l{l}")
            bias_hh.data[self.lstm.hidden_size:2 * self.lstm.hidden_size] = 1
            bias_ih = getattr(self.lstm, f"bias_ih_l{l}")
            bias_ih.data[self.lstm.hidden_size:2 * self.lstm.hidden_size] = 1


class CombinedEncoderEmulator(torch.nn.Module):
    def __init__(self, encoder, actor, args, state):
        super().__init__()
        self.encoder = encoder
        self.emulator = actor
        self.rnn_params_shapes = state.rnn_params_shapes
        self.device = state.device
        self.args = args

    def forward(self, batch, diagnostics=False):
        actor_params_batch = batch["parameters"].to(self.device)
        sequence_batch = batch["sequences"].to(self.device)
        logits_batch = batch["logits"].to(self.device)

        # start sequences with a zero
        if self.args.one_hot_tokens:
            input_sequence_batch = sequence_batch[..., :-1]
        else:
            input_sequence_batch = sequence_batch

        combined_batch_size = input_sequence_batch.shape[0] * input_sequence_batch.shape[1]

        actor_params_batch = unflatten_params(actor_params_batch, self.rnn_params_shapes)
        if self.args.permutation_augmentation:
            for l in range(args.rnn_num_layers):
                actor_params_batch = permuteLSTMNeurons(actor_params_batch, l)
        if self.args.encoder_type == "nf":
            actor_params_batch = prepare_lstm_params_for_np(actor_params_batch)
        diagnostic_dict = {}

        if diagnostics:
            rnn_embedding, diagnostic_dict = self.encoder(actor_params_batch, diagnostics=True)
        else:
            rnn_embedding = self.encoder(actor_params_batch)

        if self.args.normalize_embedding:
            rnn_embedding = rnn_embedding / torch.norm(rnn_embedding, dim=-1, keepdim=True)

        rnn_embedding = rnn_embedding.unsqueeze(1).repeat(1, self.args.sequence_batch_size, 1)
        rnn_embedding = rnn_embedding.view(combined_batch_size, -1)
        if self.args.one_hot_tokens:
            flat_sequence_batch = input_sequence_batch.reshape(combined_batch_size, -1)
            flat_sequence_batch = F.one_hot(flat_sequence_batch, num_classes=self.args.probing_input_dim).float()
        else:
            flat_sequence_batch = input_sequence_batch.reshape(combined_batch_size,
                                                               input_sequence_batch.shape[2],
                                                               input_sequence_batch.shape[3])

        bc_output, _ = self.emulator(flat_sequence_batch,
                                     rnn_embedding)

        return bc_output, rnn_embedding, diagnostic_dict


def define_language(args):
    if args.dataset_task == "formal_languages":
        get_language = lambda task: RelativeOccurrenceLanguage(occurrence_differences=task,
                                                               sequence_length=args.sequence_length,
                                                               max_n=10)
    else:
        get_language = None

    return get_language


def define_encoder(args, probing_input_dim, probing_output_dim, v_actor_func_with_hidden, model, device,
                   vfunc_params_shapes):
    # create rnn weight encoder
    if args.encoder_type == "interactive":
        rnn_weight_encoder = InteractiveEncoder(probing_input_dim=probing_input_dim,
                                                probing_output_dim=probing_output_dim,
                                                vfunc=v_actor_func_with_hidden,
                                                hidden_sizes=[args.encoder_mlp_hidden_size] * args.encoder_mlp_num_layers,
                                                activation=torch.nn.ReLU,
                                                device=device,
                                                hidden_size_rnn=args.encoder_core_hidden_size,
                                                probing_sequence_length=args.probing_sequence_length,
                                                num_parallel_ps=args.num_parallel_probing_sequences,
                                                encoding_size=args.encoder_mlp_hidden_size,
                                                output_size=args.rnn_embedding_size,
                                                num_rnn_layers=args.encoder_core_num_layers,
                                                recurrent_vfunc=True,
                                                probing_input_softmax=args.probing_input_softmax,
                                                initial_hidden_state=model.get_initial_hidden_states(1, device=device),
                                                dropout=args.dropout,
                                                residual=args.encoder_residual,
                                                static_weight=args.static_weight,
                                                open_recurrent_connections=args.encoder_open_gate_init,
                                                rnn_cell_type=args.encoder_rnn_type)
    elif args.encoder_type == "noninteractive":
        rnn_weight_encoder = InteractiveEncoder(probing_input_dim=probing_input_dim,
                                                probing_output_dim=probing_output_dim,
                                                vfunc=v_actor_func_with_hidden,
                                                hidden_sizes=[args.encoder_mlp_hidden_size] * args.encoder_mlp_num_layers,
                                                activation=torch.nn.ReLU,
                                                device=device,
                                                hidden_size_rnn=args.encoder_core_hidden_size,
                                                probing_sequence_length=args.probing_sequence_length,
                                                num_parallel_ps=args.num_parallel_probing_sequences,
                                                encoding_size=args.encoder_mlp_hidden_size,
                                                output_size=args.rnn_embedding_size,
                                                num_rnn_layers=args.encoder_core_num_layers,
                                                recurrent_vfunc=True,
                                                probing_input_softmax=args.probing_input_softmax,
                                                initial_hidden_state=model.get_initial_hidden_states(1, device=device),
                                                dropout=args.dropout,
                                                residual=args.encoder_residual,
                                                interactive_weight=0.,
                                                static_weight=1.,
                                                open_recurrent_connections=args.encoder_open_gate_init)
    elif args.encoder_type == "nf":
        rnn_weight_encoder = NFLSTMParameterEncoder(vfunc_params_shapes,
                                                    num_np_channels=[args.encoder_core_hidden_size] *
                                                                    args.encoder_core_num_layers,
                                                mlp_hidden_sizes=[args.encoder_mlp_hidden_size] *
                                                                 args.encoder_mlp_num_layers,
                                                output_size=args.rnn_embedding_size)
    elif args.encoder_type == "flat":
        rnn_weight_encoder = FlatParameterEncoder(vfunc_params_shapes,
                                              [args.encoder_core_hidden_size] * args.encoder_core_num_layers +
                                              [args.rnn_embedding_size, ])
    elif args.encoder_type == "parameter_transformer":
        rnn_weight_encoder = ParameterTransformerEncoder(parameter_shapes=vfunc_params_shapes,
                                                         num_heads=args.encoder_num_heads,
                                                         num_transformer_layers=args.encoder_core_num_layers,
                                                         transformer_size=args.encoder_core_hidden_size,
                                                         output_size=args.rnn_embedding_size)
    elif args.encoder_type == "layerwise_statistics":
        rnn_weight_encoder = ParameterStatisticsEncoder(
            parameter_shapes=vfunc_params_shapes,
            mlp_hidden_sizes=[args.encoder_mlp_hidden_size] * args.encoder_mlp_num_layers +
                             [args.rnn_embedding_size, ],
            quantiles=(0., 0.25, 0.5, 0.75, 1.),
            per_layer=True
        )

    return rnn_weight_encoder


def check_rnn_model(state, reference_model, example):
    device = state.device
    args = state.args
    model = state.rnn_model
    actor_params_shapes = state.rnn_params_shapes
    v_actor_func_with_hidden = state.vfunc
    og_parameters = example["parameters"]
    sequences = example["sequences"]

    og_parameters = og_parameters.to(device)
    sequences = sequences.to(device)

    parameters = og_parameters.unsqueeze(0).repeat(sequences.shape[0], 1)
    parameters = unflatten_params(parameters, actor_params_shapes)
    if args.one_hot_tokens:
        dummy_input = F.one_hot(sequences[:, :-1], args.probing_input_dim).float()
    else:
        dummy_input = sequences[:, :-1].float()
    torch.nn.utils.vector_to_parameters(og_parameters, reference_model.parameters())
    reference_output = reference_model(dummy_input)
    hidden_states = model.get_initial_hidden_states(sequences.shape[0], device=device)
    actor_output_with_hidden = v_actor_func_with_hidden(parameters, dummy_input.unsqueeze(1), hidden_states)
    assert torch.allclose(reference_output[0], actor_output_with_hidden[0], atol=1e-4)


def training_iteration(batch,
                       state,
                       diagnostics=False):
    device = state.device
    args = state.args
    combined_model = state.combined_model

    sequence_batch = batch["sequences"].to(device)
    logits_batch = batch["logits"].to(device)

    if args.one_hot_tokens:
        target_sequence_batch = sequence_batch[..., 1:]
    else:
        target_sequence_batch = torch.argmax(logits_batch, dim=-1)

    emulator_output, rnn_embedding, diagnostic_dict = combined_model(batch, diagnostics=diagnostics)

    emulator_logits = torch.log_softmax(emulator_output, dim=-1)
    og_output_probs = torch.distributions.Categorical(logits=logits_batch.view(-1, args.probing_output_dim)).probs
    if args.criterion == "kl":
        losses = F.kl_div(emulator_logits.view(-1, args.probing_output_dim), og_output_probs,
                          reduction="none").mean(-1)
    elif args.criterion == "kl_reverse":
        losses = F.kl_div(og_output_probs.log(), emulator_logits.view(-1, args.probing_output_dim),
                          reduction="none", log_target=True).mean(-1)
    elif args.criterion == "js_div":
        losses = (F.kl_div(emulator_logits.view(-1, args.probing_output_dim), og_output_probs,
                           reduction="none") +
                  F.kl_div(og_output_probs.log(), emulator_logits.view(-1, args.probing_output_dim),
                           reduction="none", log_target=True)) / 2
    elif args.criterion == "total_variation":
        losses = 0.5 * (og_output_probs - torch.exp(emulator_logits.view(-1, args.probing_output_dim))).abs().sum(-1)
    elif args.criterion == "ce":
        losses = F.cross_entropy(emulator_output.view(-1, args.probing_output_dim), target_sequence_batch.reshape(-1),
                                 reduction="none")
    elif args.criterion == "cosine":
        losses = 1 - F.cosine_similarity(emulator_logits.view(-1, args.probing_output_dim), torch.log(og_output_probs), dim=-1)
    elif args.criterion == "l2":
        losses = (emulator_logits.view(-1, args.probing_output_dim) - torch.log(og_output_probs)).pow(2).mean(-1)
    losses = losses.view(rnn_embedding.shape[0], -1).mean(-1)
    losses = losses

    if diagnostics:
        emulator_logits = emulator_logits.view_as(logits_batch)
        diagnostic_dict["og_output_probs"] = og_output_probs.view_as(emulator_logits).detach().cpu()
        diagnostic_dict["emulator_output_probs"] = torch.softmax(emulator_logits, dim=-1).detach().cpu()
        diagnostic_dict["task"] = batch["task"].detach().cpu()

    if diagnostics:
        return losses, diagnostic_dict
    return losses


@fix_seed
def evaluate_with_dataset(dataset,
                          state: ModelState,
                          diagnostics=False):
    """Evaluates the current behavior cloning on the validation set."""
    rnn_weight_encoder = state.rnn_weight_encoder
    emulator = state.emulator
    args = state.args

    rnn_weight_encoder.eval()
    emulator.eval()

    all_losses = []

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.rnn_batch_size, shuffle=False)
    diagnostics_dict = None
    for i, batch in enumerate(dataloader):
        if diagnostics:
            losses, new_diagnostics_dict = training_iteration(batch,
                                                              state=state,
                                                              diagnostics=True)
            if diagnostics_dict is None:
                diagnostics_dict = new_diagnostics_dict
            else:
                for key in new_diagnostics_dict:
                    diagnostics_dict[key] = torch.cat([diagnostics_dict[key], new_diagnostics_dict[key]], dim=0)
        else:
            losses = training_iteration(batch)

        all_losses.append(losses.detach().cpu())
    all_losses = torch.cat(all_losses, dim=0)
    eval_dict = {"validation loss": all_losses.mean().item()}

    if diagnostics:
        return eval_dict, diagnostics_dict

    rnn_weight_encoder.train()
    emulator.train()
    return eval_dict


def create_eval_logging_dict(state,
                             dataset,
                             num_examples=5):
    args = state.args
    eval_dict, diagnostics_dict = evaluate_with_dataset(dataset,
                                                        state=state,
                                                        diagnostics=True)
    indices = torch.linspace(0, len(dataset) - 1, num_examples).long()

    distribution_images = []
    task_list = []
    for idx in indices:
        distributions_fig, _ = visualize_output_distributions(diagnostics_dict["emulator_output_probs"][idx, 0],
                                                              diagnostics_dict["og_output_probs"][idx, 0])
        distributions_pil = wandb.Image(figure2PIL(distributions_fig))
        distribution_images.append(distributions_pil)
        task_list.append(diagnostics_dict["task"][idx].tolist())

    if args.dataset_task == "sequential_mnist":
        visualize_probing_inputs = visualize_probing_inputs_sequential_mnist
    else:
        visualize_probing_inputs = visualize_probing_inputs_formal_language

    eval_dict["distributions"] = distribution_images
    if args.dataset_task == "modulo_addition_masked":
        columns = ["result"] + [f"{i}" for i in range(args.num_tokens)]
        eval_dict["task"] = wandb.Table(columns=columns, data=task_list)
    elif args.dataset_task == "alternating_crossroads":
        eval_dict["task"] = wandb.Table(columns=["target"], data=task_list)
    elif args.dataset_task == "multitarget_crossroads":
        columns = ["target"] + [f"{i}" for i in range(args.num_tokens)]
        eval_dict["task"] = wandb.Table(columns=columns, data=task_list)
    elif args.dataset_task == "sequential_mnist":
        eval_dict["task"] = wandb.Table(columns=["target"], data=task_list)
    else:
        eval_dict["task"] = wandb.Table(columns=[f"{i}" for i in range(args.num_tokens)], data=task_list)

    if "probing_actions" in diagnostics_dict.keys():
        probing_inputs_images = []
        probing_outputs_images = []
        for idx in indices[:5]:
            probing_inputs_fig, _ = visualize_probing_inputs(diagnostics_dict["probing_inputs"][idx],
                                                                         state=state)
            probing_inputs_pil = wandb.Image(figure2PIL(probing_inputs_fig))
            probing_inputs_images.append(probing_inputs_pil)

            probing_outputs_fig, _ = visualize_probing_outputs(diagnostics_dict["probing_outputs"][idx])
            probing_outputs_pil = wandb.Image(figure2PIL(probing_outputs_fig))
            probing_outputs_images.append(probing_outputs_pil)
        eval_dict["probing inputs"] = probing_inputs_images
        eval_dict["probing outputs"] = probing_outputs_images

    return eval_dict


def main(args):
    log_with_wandb = args.wandb_logging > 0
    args.server = socket.gethostname()
    hyperparameters = dict(vars(args))

    logger = Logger(enabled=log_with_wandb,
                    print_logs_to_console=not log_with_wandb,
                    project="recurrent_behavior_cloning",
                    tags=[],
                    config=hyperparameters)

    if args.seed >= 0:
        seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu") if args.cuda else 'cpu'

    dataset_name = args.dataset
    dataset_path = os.path.join('datasets', dataset_name)
    if not os.path.exists(dataset_path):
        print(f"Did not find the dataset directory {dataset_path}.")

    eval_ratio = 0.15
    ood_ratio = 0.15
    if args.dataset_task == "formal_languages":
        dataset_type = lambda mode: ParametersFormalLanguageDataset(dataset_path,
                                                                    sequences_per_rnn=args.sequence_batch_size,
                                                                    eval_ratio=eval_ratio, ood_ratio=ood_ratio,
                                                                    mode=mode,
                                                                    task_criterion=lambda x: task_entropy_criterion(x))
    elif args.dataset_task == "sequential_mnist":
        dataset_type = lambda mode: ParametersSequentialMNISTDataset(dataset_path,
                                                                     sequences_per_rnn=args.sequence_batch_size,
                                                                     eval_ratio=eval_ratio, ood_ratio=ood_ratio,
                                                                     mode=mode)

    train_dataset = dataset_type('train')
    eval_dataset = dataset_type('eval')
    ood_dataset = dataset_type('ood')

    get_language = define_language(args)

    reference_model = RNNFast(hidden_size=args.rnn_hidden_size,
                              input_size=args.probing_input_dim,
                              output_size=args.probing_output_dim,
                              num_layers=args.rnn_num_layers,
                              input_batch=True)

    model = RNNFunctionalizable(hidden_size=args.rnn_hidden_size,
                                input_size=args.probing_input_dim,
                                output_size=args.probing_output_dim,
                                num_layers=args.rnn_num_layers,
                                input_batch=True)

    actor_params_dict = dict(model.named_parameters())
    actor_func_with_hidden = lambda parameters, x, hidden: torch.func.functional_call(model, parameters,
                                                                                      args=(x, hidden))

    actor_params_shapes = {k: p.shape for k, p in actor_params_dict.items()}
    actor_params_reference = list(reference_model.named_parameters())
    converted_names = convert_parameter_names([k for k, _ in actor_params_reference])
    actor_params_shapes = [(k, actor_params_shapes[k]) for k in converted_names]

    v_actor_func_with_hidden = torch.vmap(actor_func_with_hidden, in_dims=(0, 0, (0, 0)), out_dims=(0, (0, 0)))

    obs_dim = actor_params_shapes[0][1][-1]
    probing_output_dim = actor_params_shapes[-1][1][0]

    rnn_weight_encoder = define_encoder(args,
                                    probing_input_dim=obs_dim,
                                    probing_output_dim=probing_output_dim,
                                    v_actor_func_with_hidden=v_actor_func_with_hidden,
                                    model=model,
                                    device=device,
                                    vfunc_params_shapes=actor_params_shapes)
    rnn_weight_encoder.to(device)
    encoder_num_params = num_params(rnn_weight_encoder)

    emulator = EmulatorLSTM(rnn_embedding_size=args.rnn_embedding_size,
                            obs_dim=obs_dim,
                            probing_output_dim=probing_output_dim,
                            hidden_size=args.emulator_hidden_size,
                            num_layers=args.emulator_num_layers,
                            only_condition_bos=args.emulator_only_condition_bos)
    if args.emulator_open_gate_init:
        emulator.open_forget_gates()
    emulator.to(device)
    bc_actor_num_params = num_params(emulator)

    logger().log({
        "num bc_actor params": bc_actor_num_params,
        "num encoder params": encoder_num_params,
    }, step=0)

    optimizer = torch.optim.AdamW(
        list(rnn_weight_encoder.parameters()) +
        list(emulator.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    step = 0
    epoch = 0
    meter = MultiMeter()
    min_validation_loss = float("inf")
    min_return_diff = float("inf")
    min_ood_loss_step = 0

    model_state = ModelState(
        rnn_weight_encoder=rnn_weight_encoder,
        emulator=emulator,
        combined_model=None,
        device=device,
        rnn_params_shapes=actor_params_shapes,
        vfunc=v_actor_func_with_hidden,
        rnn_model=model,
        args=args,
        get_language=get_language
    )

    combined_model = CombinedEncoderEmulator(rnn_weight_encoder,
                                             emulator,
                                             args=args,
                                             state=model_state)
    logger().watch(combined_model, log="all", log_freq=1000)
    model_state.combined_model = combined_model

    check_rnn_model(model_state,
                    reference_model=reference_model,
                    example=train_dataset[0])

    while True:
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.rnn_batch_size, shuffle=True)
        for batch in iter(dataloader):
            losses = training_iteration(batch,
                                        state=model_state)

            loss = losses.mean()

            optimizer.zero_grad()
            loss.backward()

            if args.gradient_clipping > 0.:
                torch.nn.utils.clip_grad_norm_(emulator.parameters(), args.gradient_clipping)
                torch.nn.utils.clip_grad_norm_(rnn_weight_encoder.parameters(), args.gradient_clipping)

            optimizer.step()

            meter.update({"training loss": loss.item()})

            if step % 100 == 0:
                logging_dict = {"training loss": meter["training loss"].avg}
                logger().log(logging_dict, step=step)
                meter.reset()

            if step % 1000 == 0:
                eval_dict = create_eval_logging_dict(state=model_state,
                                                     dataset=eval_dataset)
                logger().log(eval_dict, step=step)

                if ood_dataset is not None:
                    ood_eval_dict = create_eval_logging_dict(state=model_state,
                                                             dataset=ood_dataset)
                    log_ood_eval_dict = {f"ood {key}": value for key, value in ood_eval_dict.items()}
                    logger().log(log_ood_eval_dict, step=step)

                train_idxs = torch.linspace(0, len(train_dataset) - 1,
                                            args.num_evaluation_policies).long()
                returns = [train_dataset[idx]["returns"].item() for idx in train_idxs]
                train_idxs = [idx for _, idx in sorted(zip(returns, train_idxs), key=lambda pair: pair[0])]
                train_idxs = torch.stack(train_idxs, dim=0)
                return_difference = evaluate_clones(train_idxs,
                                                    train_dataset,
                                                    evaluation_state=model_state)
                logger().log({"train returns": plt}, step=step)

                eval_idxs = torch.linspace(0, len(eval_dataset) - 1,
                                           args.num_evaluation_policies).long()
                returns = [eval_dataset[idx]["returns"].item() for idx in eval_idxs]
                eval_idxs = [idx for _, idx in sorted(zip(returns, eval_idxs), key=lambda pair: pair[0])]
                eval_idxs = torch.stack(eval_idxs, dim=0)
                return_difference, fig = evaluate_clones(eval_idxs,
                                                         eval_dataset,
                                                         evaluation_state=model_state)
                if abs(return_difference) < min_return_diff:
                    min_return_diff = abs(return_difference)
                logger().log({"eval returns": fig,
                              "return difference": return_difference,
                              "min return difference": min_return_diff}, step=step)

                if ood_dataset is not None:
                    ood_eval_idxs = torch.linspace(0, len(ood_dataset) - 1,
                                                   args.num_evaluation_policies).long()
                    returns = [ood_dataset[idx]["returns"].item() for idx in ood_eval_idxs]
                    ood_eval_idxs = [idx for _, idx in sorted(zip(returns, ood_eval_idxs), key=lambda pair: pair[0])]
                    ood_eval_idxs = torch.stack(ood_eval_idxs, dim=0)
                    ood_return_difference, fig = evaluate_clones(ood_eval_idxs,
                                                                 ood_dataset,
                                                                 evaluation_state=model_state)
                    logger().log({"ood eval returns": fig,
                                  "ood return difference": ood_return_difference}, step=step)

                fig_return, fig_task = visualize_embedding_space(model_state,
                                                                 datasets=[eval_dataset, ood_dataset])
                logger().log({"embedding space returns": fig_return,
                              "embedding space tasks": fig_task}, step=step)

                validation_loss = eval_dict["validation loss"]

                if validation_loss < min_validation_loss:
                    min_validation_loss = validation_loss
                    if args.save_model and log_with_wandb:
                        print("saving models")
                        torch.save(
                            rnn_weight_encoder.cpu().state_dict(),
                            os.path.join(logger().run.dir, "rnn_weight_encoder.pt")
                        )
                        rnn_weight_encoder.to(device)
                        torch.save(
                            emulator.cpu().state_dict(),
                            os.path.join(logger().run.dir, "bc_actor.pt")
                        )
                        emulator.to(device)
                    else:
                        print("not saving models")
                    min_ood_loss_step = step
                elif args.early_stopping > 0 and step - min_ood_loss_step > args.early_stopping:
                    logger().log({"early stopping": True}, step=step)
                    return

            if step >= args.total_training_steps:
                return

            step += 1
        epoch += 1


if __name__ == "__main__":
    args = parse_args()
    if args.config != "":
        try:
            config = encoder_configs[args.config]
        except KeyError:
            raise ValueError(f"Config {args.config} not found in encoder_configs.")

        for key, value in config.items():
            args.__dict__[key] = value
    args = conditional_args(args)
    main(args)