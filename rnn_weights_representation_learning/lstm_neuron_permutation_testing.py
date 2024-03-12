import torch
from modules_rnn_models import RNNFunctionalizable

from modules_rnn_encoders import InteractiveEncoder, FlatParameterEncoder, NFLSTMParameterEncoder
from modules_neural_functionals import prepare_lstm_params_for_np


def permuteLSTMNeurons(parameters, layer, permutation=None, switching_dict=None):
    layer_size = parameters[f'lstm_cells.{layer}.i2h.weight'].shape[1] // 4
    device = parameters[f'lstm_cells.{layer}.i2h.weight'].device
    if permutation is None:
        permutation = torch.randperm(layer_size, device=device)
    num_lstm_layers = (len(parameters) - 2) // 4
    permutation_x4 = (permutation.unsqueeze(0).repeat(4, 1) +
                      torch.arange(4, device=device).unsqueeze(1) * layer_size).flatten()

    if switching_dict is None:
        switching_dict = {
            "current_i2h_weights": True,
            "current_h2h_weights": True,
            "current_i2h_biases": True,
            "current_h2h_biases": True,
            "next_i2h_weights": True,
            "next_h2h_weights": True,
        }

    # switch columns in current layer's weights
    if switching_dict["current_i2h_weights"]:
        parameters[f'lstm_cells.{layer}.i2h.weight'] = parameters[f'lstm_cells.{layer}.i2h.weight'][:, permutation_x4].clone()

    if switching_dict["current_h2h_weights"]:
        parameters[f'lstm_cells.{layer}.h2h.weight'] = parameters[f'lstm_cells.{layer}.h2h.weight'][:, permutation_x4].clone()

    # switch columns in current layer's biases
    if switching_dict["current_i2h_biases"]:
        parameters[f'lstm_cells.{layer}.i2h.bias'] = parameters[f'lstm_cells.{layer}.i2h.bias'][:, permutation_x4].clone()

    if switching_dict["current_h2h_biases"]:
        parameters[f'lstm_cells.{layer}.h2h.bias'] = parameters[f'lstm_cells.{layer}.h2h.bias'][:, permutation_x4].clone()

    # switch rows in next layer's weights
    if switching_dict["next_i2h_weights"]:
        if layer < num_lstm_layers - 1:
            parameters[f'lstm_cells.{layer + 1}.i2h.weight'] = parameters[f'lstm_cells.{layer + 1}.i2h.weight'][:, :, permutation].clone()
        else:
            parameters[f'linear.weight'] = parameters[f'linear.weight'][:, :, permutation].clone()

    if switching_dict["next_h2h_weights"]:
        parameters[f'lstm_cells.{layer}.h2h.weight'] = parameters[f'lstm_cells.{layer}.h2h.weight'][:, :, permutation].clone()

    return parameters


def check_permutation_invariance(parameter_encoder, parameters):
    if isinstance(parameter_encoder, NFLSTMParameterEncoder):
        og_output = parameter_encoder(prepare_lstm_params_for_np(parameters))
    else:
        og_output = parameter_encoder(parameters)
    og_params = parameters
    print(og_output)

    num_layers = (len(parameters) - 2) // 4

    switching_dict = {
        "current_i2h_weights": True,
        "current_h2h_weights": True,
        "current_i2h_biases": True,
        "current_h2h_biases": True,
        "next_i2h_weights": True,
        "next_h2h_weights": True,
    }
    switching_keys = list(switching_dict.keys())

    for layer in range(num_layers):
        for parameter_block in range(7):
            permuted_params = {k: p.clone() for k, p in og_params.items()}
            output_should_be_same = True
            for perm_layer in range(num_layers):
                if parameter_block < 6 and perm_layer != layer:
                    switching_dict = {k: i != parameter_block for i, k in enumerate(switching_keys)}
                    output_should_be_same = False
                else:
                    switching_dict = {k: True for k in switching_keys}
                print(f"switching dict for layer {perm_layer}: {switching_dict}")
                permuted_params = permuteLSTMNeurons(permuted_params,
                                                     layer=layer,
                                                     switching_dict=switching_dict)
            if isinstance(parameter_encoder, NFLSTMParameterEncoder):
                permuted_params = prepare_lstm_params_for_np(permuted_params)
            permuted_output = parameter_encoder(permuted_params)

            output_is_same = torch.allclose(og_output, permuted_output, atol=1e-5)
            print(f"output is same: {output_is_same}, output should be same: {output_should_be_same}\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 6
    hidden_size = 32
    output_size = 7
    num_layers = 2
    batch_size = 4
    sequence_length = 12

    model = RNNFunctionalizable(hidden_size=hidden_size, input_size=input_size,
                                      output_size=output_size,
                                      num_layers=num_layers, input_batch=True)
    model.to(device)

    actor_params_dict = dict(model.named_parameters())
    actor_params_shapes = [(k, p.shape) for k, p in actor_params_dict.items()]
    actor_func = lambda parameters, x: torch.func.functional_call(model, parameters, x)
    actor_func_with_hidden = lambda parameters, x, hidden: torch.func.functional_call(model, parameters,
                                                                                      args=(x, hidden))
    actor_params = actor_params_dict
    vfunc = torch.vmap(actor_func, in_dims=(0, 0), out_dims=(0, (0, 0)))
    vfunc_with_hidden = torch.vmap(actor_func_with_hidden, in_dims=(0, 0, (0, 0)), out_dims=(0, (0, 0)))

    og_parameters = {k: p.unsqueeze(0).repeat(batch_size, 1, 1) if len(p.shape) == 2
                     else p.unsqueeze(0).repeat(batch_size, 1) for k, p in actor_params.items()}
    og_parameters = {k: p + torch.randn_like(p) for k, p in og_parameters.items()}
    dummy_input = torch.randn(batch_size, sequence_length, input_size, device=device)
    hidden_states = model.get_initial_hidden_states(batch_size, device=device)
    og_actor_output = vfunc_with_hidden(og_parameters, dummy_input.unsqueeze(1), hidden_states)[0]

    switching_dict = {
        "current_i2h_weights": True,
        "current_h2h_weights": True,
        "current_i2h_biases": True,
        "current_h2h_biases": True,
        "next_i2h_weights": True,
        "next_h2h_weights": True,
    }
    switching_keys = list(switching_dict.keys())

    for layer in range(num_layers):
        for parameter_block in range(7):
            permuted_params = {k: p.clone() for k, p in og_parameters.items()}
            output_should_be_same = True
            for perm_layer in range(num_layers):
                if parameter_block < 6 and perm_layer != layer:
                    switching_dict = {k: i != parameter_block for i, k in enumerate(switching_keys)}
                    output_should_be_same = False
                else:
                    switching_dict = {k: True for k in switching_keys}
                print(f"switching dict for layer {perm_layer}: {switching_dict}")
                permuted_params = permuteLSTMNeurons(permuted_params,
                                                     layer=layer,
                                                     switching_dict=switching_dict)
            permuted_output = vfunc_with_hidden(permuted_params, dummy_input.unsqueeze(1), hidden_states)[0]

            output_is_same = torch.allclose(og_actor_output, permuted_output, atol=1e-5)
            assert output_is_same == output_should_be_same
            print(f"output is same: {output_is_same}, output should be same: {output_should_be_same}\n")

    print("\n\nTesting Interative Encoder")
    policy_encoder = InteractiveEncoder(probing_input_dim=input_size,
                                        probing_output_dim=output_size,
                                        vfunc=vfunc_with_hidden,
                                        hidden_sizes=[64] * 2,
                                        activation=torch.nn.ReLU,
                                        device=device,
                                        hidden_size_rnn=128,
                                        probing_sequence_length=10,
                                        num_parallel_ps=4,
                                        encoding_size=64,
                                        output_size=16,
                                        num_rnn_layers=2,
                                        recurrent_vfunc=True,
                                        probing_input_softmax=True,
                                        initial_hidden_state=model.get_initial_hidden_states(1, device=device),
                                        dropout=False,
                                        residual=False,
                                        static_weight=0.)
    policy_encoder.to(device)
    check_permutation_invariance(policy_encoder, og_parameters)

    print("\n\nTesting Flat Encoder")
    policy_encoder = FlatParameterEncoder(actor_params_shapes,
                                          [64, 64, 16])
    policy_encoder.to(device)
    check_permutation_invariance(policy_encoder, og_parameters)

    print("\n\nTesting NF Encoder")
    policy_encoder = NFLSTMParameterEncoder(actor_params_shapes,
                                            num_np_channels=[16, 16, 16],
                                            mlp_hidden_sizes=[64, 64],
                                            output_size=16)
    policy_encoder.to(device)
    check_permutation_invariance(policy_encoder, og_parameters)


if __name__ == "__main__":
    main()