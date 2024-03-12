import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from utilities import unflatten_params, hash64
from modules_neural_functionals import prepare_lstm_params_for_np
from dataset_creation_sequential_mnist import TiledMNIST, convert_seq_to_image


def evaluate_on_formal_language_task(
        batch,
        state,
        clone_rnn: bool,
        num_episodes=100):
    """Evaluates a rnn in the environment. If clone_rnn is False, then the given rnn is evaluated. If it is
    True, the rnn is cloned based on the given parameters and the clone is evaluated."""

    rnn_weight_encoder = state.rnn_weight_encoder
    emulator = state.emulator
    args = state.args
    rnn_params_shapes = state.rnn_params_shapes
    model = state.rnn_model
    vfunc = state.vfunc
    get_language = state.get_language
    device = state.device

    rnn_weight_encoder.eval()
    emulator.eval()

    if len(batch["parameters"].shape) == 1:
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}
    rnn_params_batch = batch["parameters"].to(device)
    rnn_params_batch = unflatten_params(rnn_params_batch, rnn_params_shapes)
    if args.encoder_type == "nf" and clone_rnn:
        rnn_params_batch = prepare_lstm_params_for_np(rnn_params_batch)
    sequences = batch["sequences"].to(device)
    og_logits = batch["logits"].to(device)
    task = batch["task"][0]
    language = get_language(task)

    if clone_rnn:
        rnn_weight_embedding = rnn_weight_encoder(rnn_params_batch)
        if args.normalize_embedding:
            rnn_weight_embedding = rnn_weight_embedding / torch.norm(rnn_weight_embedding, dim=-1, keepdim=True)
        rnn_weight_embedding = rnn_weight_embedding.repeat(num_episodes, 1)
    else:
        rnn_params_batch = {k: p.repeat(*([num_episodes] + [1] * (len(p.shape) - 1)))
                              for k, p in rnn_params_batch.items()}

    sequence_length = args.sequence_length
    hidden_states = None
    sequence = [torch.ones(num_episodes, 1, device=device, dtype=torch.long) * language.bos_token]
    sequence_logits = []
    for j in range(sequence_length - 1):
        if args.one_hot_tokens:
            input = F.one_hot(sequence[-1], args.probing_input_dim).float()
        else:
            input = sequence[-1].unsqueeze(-1).float()
        if clone_rnn:
            output, hidden_states = emulator(input, rnn_weight_embedding, hidden_states)
        else:
            if hidden_states is None:
                hidden_states = model.get_initial_hidden_states(input.shape[0], device=device)
            output, hidden_states = vfunc(rnn_params_batch, input.unsqueeze(1), hidden_states)
        logits = F.log_softmax(output, dim=2)
        logits = logits.squeeze(1)
        if not clone_rnn and j == 0:
            og_logits = og_logits[0, :num_episodes, 0]
            # check if logits and og_logits are the same
            assert torch.allclose(logits[:og_logits.shape[0]], og_logits, atol=1e-2)

        # get one sample from each distribution
        sample = torch.multinomial(torch.exp(logits), num_samples=1)
        sequence.append(sample)
        sequence_logits.append(logits)
    sequence = torch.cat(sequence, dim=1)
    sequence_logits = torch.stack(sequence_logits, dim=1)
    accepted, undefined = language.accept_sequence(sequence.cpu())
    num_correct = accepted.sum().item()
    num_finished = accepted.shape[0] - undefined.sum().item()
    if num_finished > 0:
        mean_reward = num_correct / num_finished
    else:
        mean_reward = 0.

    rnn_weight_encoder.train()
    emulator.train()

    return mean_reward


def evaluate_on_mnist_task(
        batch,
        state,
        clone_rnn: bool,
        num_episodes=100):
    """Evaluates a rnn in the environment. If clone_rnn is False, then the given rnn is evaluated. If it is
    True, the rnn is cloned based on the given parameters and the clone is evaluated."""

    # since the validation task is exactly the same as in during the creation of the dataset, if we don't clone the
    # rnn, we can just return the accuracy that is saved in the dataset. If we still want to check if everything is
    # working correctly, we can comment out the next two lines
    if not clone_rnn:
        return batch["returns"][0].item()

    rnn_weight_encoder = state.rnn_weight_encoder
    emulator = state.emulator
    args = state.args
    rnn_params_shapes = state.rnn_params_shapes
    vfunc = state.vfunc
    device = state.device
    task = batch["task"][0]
    model = state.rnn_model

    rnn_weight_encoder.eval()
    emulator.eval()

    indices = torch.arange(60_000)
    hash_indices = hash64(indices)
    eval_indices = indices[hash_indices % 10 >= 8]

    transform = transforms.Compose([
        transforms.RandomRotation([task, task]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = TiledMNIST(root="./data", train=True, download=True,
                         transform=transform,
                         tile_w=args.tile_size, tile_h=args.tile_size)
    eval_dataset = torch.utils.data.Subset(dataset, eval_indices)

    if len(batch["parameters"].shape) == 1:
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}
    rnn_params_batch = batch["parameters"].to(device)
    rnn_params_batch = unflatten_params(rnn_params_batch, rnn_params_shapes)
    if args.encoder_type == "nf" and clone_rnn:
        rnn_params_batch = prepare_lstm_params_for_np(rnn_params_batch)
    sequences = batch["sequences"].to(device)
    og_logits = batch["logits"].to(device)

    if clone_rnn:
        rnn_weight_embedding = rnn_weight_encoder(rnn_params_batch)
        if args.normalize_embedding:
            rnn_weight_embedding = rnn_weight_embedding / torch.norm(rnn_weight_embedding, dim=-1, keepdim=True)

    correct_predictions = 0
    total_predictions = 0
    losses = []
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1000, shuffle=True)

    hidden_states = model.get_initial_hidden_states(rnn_params_batch['linear.weight'].shape[0], device=device)

    for batch_idx, (data, target) in enumerate(eval_dataloader):
        data, target = data.to(device), target.to(device)
        if clone_rnn:
            logits, _ = emulator(data, rnn_weight_embedding)
        else:
            logits, _ = vfunc(rnn_params_batch, data.unsqueeze(0), hidden_states)
            logits = logits.squeeze(0)

        repeated_target = target.unsqueeze(1).repeat(1, logits.shape[1])

        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), repeated_target.flatten())
        losses.append(loss.item())

        predictions = logits[:, -1].argmax(dim=1)
        correct_predictions += (predictions == target).sum().item()
        total_predictions += target.shape[0]
        break # to save time. For full validation set, remove this line

    eval_accuracy = correct_predictions / total_predictions

    rnn_weight_encoder.train()
    emulator.train()

    return eval_accuracy


def evaluate_clones(eval_idxs,
                    dataset,
                    evaluation_state):
    plt.clf()
    original_returns = []
    cloned_returns = []

    args = evaluation_state.args
    if args.dataset_task == "sequential_mnist":
        evaluation_on_task_func = evaluate_on_mnist_task
    else:
        evaluation_on_task_func = evaluate_on_formal_language_task

    for idx in eval_idxs:
        batch = dataset[idx]

        original_return = evaluation_on_task_func(batch,
                                                  evaluation_state,
                                                  clone_rnn=False,
                                                  num_episodes=64)
        original_returns.append(original_return)
        cloned_return_mean = evaluation_on_task_func(batch,
                                                     evaluation_state,
                                                     clone_rnn=True,
                                                     num_episodes=64)
        cloned_returns.append(cloned_return_mean)
    original_returns = torch.tensor(original_returns)
    cloned_returns = torch.tensor(cloned_returns)
    return_difference = (original_returns - cloned_returns).abs().mean().item()

    # plot original returns and cloned mean returns in the same graph
    plt.plot(torch.arange(len(eval_idxs)), original_returns, label="original", color="red")
    plt.plot(torch.arange(len(eval_idxs)), cloned_returns, label="cloned", color="blue")
    plt.ylabel("returns")
    plt.legend()
    fig = plt.gcf()
    return return_difference, fig


def visualize_embedding_space(state,
                              datasets):
    """Visualizes the embedding space of the rnn encoder using the validation set"""
    args = state.args
    rnn_weight_encoder = state.rnn_weight_encoder
    device = state.device
    rnn_params_shapes = state.rnn_params_shapes

    rnn_weight_encoder.eval()

    plt.clf()
    embeddings_list = []
    returns_list = []
    tasks_list = []
    for dataset in datasets:
        embeddings = []
        returns = []
        tasks = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

        for batch in iter(dataloader):
            rnn_params_batch = batch["parameters"].to(device)
            sequence_batch = batch["sequences"].to(device)
            logits_batch = batch["logits"].to(device)
            returns_batch = batch["returns"]
            task_batch = batch["task"]
            rnn_params_batch = unflatten_params(rnn_params_batch,
                                                  rnn_params_shapes)
            if args.encoder_type == "nf":
                rnn_params_batch = prepare_lstm_params_for_np(rnn_params_batch)

            rnn_weight_embedding = rnn_weight_encoder(rnn_params_batch)

            if args.normalize_embedding:
                rnn_weight_embedding = rnn_weight_embedding / torch.norm(rnn_weight_embedding, dim=-1, keepdim=True)

            embeddings.append(rnn_weight_embedding.detach().cpu())
            returns.append(returns_batch)
            tasks.append(task_batch)

        embeddings = torch.cat(embeddings, dim=0)
        returns = torch.cat(returns, dim=0)[:, 0]
        tasks = torch.cat(tasks, dim=0)

        embeddings_list.append(embeddings)
        returns_list.append(returns)
        tasks_list.append(tasks)

    # create PCA of embeddings
    all_embeddings = torch.cat(embeddings_list, dim=0)
    pca = PCA(n_components=2)
    pca.fit(embeddings)

    pca_embeddings_list = [pca.transform(embeddings) for embeddings in embeddings_list]

    # plot embeddings_pca, color by returns
    colormaps = ["viridis", "plasma", "inferno", "magma"]
    fig_return, ax = plt.subplots()
    for i, (embeddings_pca, returns) in enumerate(zip(pca_embeddings_list, returns_list)):
        ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=returns, cmap=colormaps[i])

    # plot embeddings_pca, color by task
    fig_task, ax = plt.subplots()
    for i, (embeddings_pca, tasks) in enumerate(zip(pca_embeddings_list, tasks_list)):
        tasks = 1 + tasks - tasks.min(dim=-1, keepdims=True)[0]
        tasks_as_distribution = tasks.float() / tasks.sum(-1, keepdims=True)
        entropy = -(tasks_as_distribution * torch.log(tasks_as_distribution + 1e-6)).sum(-1)
        ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=entropy, cmap=colormaps[i])

    rnn_weight_encoder.train()
    return fig_return, fig_task


def visualize_probing_inputs_formal_language(probing_inputs, state):
    plt.clf()
    fig, ax = plt.subplots()
    probing_inputs = probing_inputs.view(probing_inputs.shape[0], -1)
    plt.imshow(probing_inputs.t())
    return fig, ax


def visualize_probing_inputs_sequential_mnist(probing_inputs, state):
    tile_size = state.args.tile_size
    plt.clf()
    fig, ax = plt.subplots()
    probing_image = convert_seq_to_image(probing_inputs.transpose(0, 1), tile_size=tile_size)
    probing_image = probing_image.view(-1, probing_image.shape[-1])
    plt.imshow(probing_image)
    return fig, ax


def visualize_probing_outputs(probing_outputs):
    plt.clf()
    fig, ax = plt.subplots()
    probing_outputs = probing_outputs.view(probing_outputs.shape[0], -1)
    plt.imshow(probing_outputs.t())
    return fig, ax


def visualize_output_distributions(emulator_dist, og_dist):
    # the distributions have the shape seq_len x num_tokens
    # this function creates two imshow plots
    # the top one shows the original distribution over tokens for each time step
    # the bottom one shows the distribution created by the emulator

    # create the two plots
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # plot the original distribution, and set the range from 0 to 1
    ax1.imshow(og_dist.t(), vmin=0, vmax=1)
    ax1.set_ylabel("original distribution")

    # plot the emulator distribution
    ax2.imshow(emulator_dist.t(), vmin=0, vmax=1)
    ax2.set_ylabel("emulated distribution")

    return fig, (ax1, ax2)
