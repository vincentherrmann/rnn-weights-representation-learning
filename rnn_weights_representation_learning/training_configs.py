from copy import deepcopy

general_configs = {
    # rnn
    "rnn_num_layers": 2,
    "rnn_hidden_size": 32,
    # training
    "total_training_steps": 100_000,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "criterion": "total_variation",
    "gradient_clipping": 0.1,
    "rnn_batch_size": 64,
    "sequence_batch_size": 1,
    "normalize_embedding": 1,
    "dropout": 0.,
    # encoder
    "rnn_embedding_size": 16,
    "encoder_open_gate_init": 0,
    # actor
    "emulator_num_layers": 2,
    "emulator_hidden_size": 256,
    "emulator_only_condition_bos": 1,
    "emulator_open_gate_init": 0
}

encoder_configs = {
    "layerwise_statistics": {
        "encoder_type": "layerwise_statistics",
        "encoder_core_hidden_size": 768,
        "encoder_core_num_layers": 3,
    },
    "flat": {
        "encoder_type": "flat",
        "encoder_core_hidden_size": 128,
        "encoder_core_num_layers": 3,
    },
    "parameter_transformer": {
        "encoder_type": "parameter_transformer",
        "encoder_core_num_layers": 6,
        "encoder_core_hidden_size": 128,
        "encoder_num_heads": 2,
    },
    "nf": {
        "encoder_type": "nf",
        "encoder_core_hidden_size": 48,
        "encoder_core_num_layers": 4,
        "encoder_mlp_hidden_size": 128,
        "encoder_mlp_num_layers": 1,
    },
    "noninteractive_1": {
        "encoder_type": "noninteractive",
        "encoder_mlp_hidden_size": 128,
        "encoder_mlp_num_layers": 1,
        "encoder_core_hidden_size": 256,
        "encoder_core_num_layers": 2,
        "encoder_residual": 1,
        "encoder_rnn_type": "lstm",
        "num_parallel_probing_sequences": 1,
        "probing_sequence_length": 22,
    },
    "interactive_1": {
        "encoder_type": "interactive",
        "encoder_mlp_hidden_size": 128,
        "encoder_mlp_num_layers": 1,
        "encoder_core_hidden_size": 256,
        "encoder_core_num_layers": 2,
        "encoder_residual": 1,
        "encoder_rnn_type": "lstm",
        "num_parallel_probing_sequences": 1,
        "probing_sequence_length": 22,
    }
}

encoder_configs["noninteractive_2"] = deepcopy(encoder_configs["noninteractive_1"])
encoder_configs["noninteractive_2"]["num_parallel_probing_sequences"] = 2

encoder_configs["noninteractive_4"] = deepcopy(encoder_configs["noninteractive_1"])
encoder_configs["noninteractive_4"]["num_parallel_probing_sequences"] = 4

encoder_configs["noninteractive_8"] = deepcopy(encoder_configs["noninteractive_1"])
encoder_configs["noninteractive_8"]["num_parallel_probing_sequences"] = 8

encoder_configs["interactive_2"] = deepcopy(encoder_configs["interactive_1"])
encoder_configs["interactive_2"]["num_parallel_probing_sequences"] = 2

encoder_configs["interactive_4"] = deepcopy(encoder_configs["interactive_1"])
encoder_configs["interactive_4"]["num_parallel_probing_sequences"] = 4

encoder_configs["interactive_8"] = deepcopy(encoder_configs["interactive_1"])
encoder_configs["interactive_8"]["num_parallel_probing_sequences"] = 8

encoder_configs["interactive_length_7"] = deepcopy(encoder_configs["interactive_1"])
encoder_configs["interactive_length_7"]["probing_sequence_length"] = 7

encoder_configs["interactive_length_12"] = deepcopy(encoder_configs["interactive_1"])
encoder_configs["interactive_length_12"]["probing_sequence_length"] = 12

encoder_configs["interactive_length_42"] = deepcopy(encoder_configs["interactive_1"])
encoder_configs["interactive_length_42"]["probing_sequence_length"] = 42

for key in encoder_configs:
    encoder_configs[key].update(general_configs)
