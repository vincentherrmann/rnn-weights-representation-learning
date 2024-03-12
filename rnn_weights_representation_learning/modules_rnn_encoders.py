import torch
import torch.nn as nn
import numpy as np

from modules_neural_functionals import TreePointwise, NPLinearLSTM
from utilities import tree_map


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity, dropout=0.0):
    layers = []
    for j in range(len(sizes) - 1):
        if dropout > 0:
            layers += [nn.Dropout(p=dropout)]
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]

    return nn.Sequential(*layers)


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, attention=False, dropout=0., num_heads=1, residual=False, type='lstm'):
        super().__init__()
        self.type = type
        if type == 'lstm':
            self.cell = nn.LSTMCell(input_size, hidden_size)
        elif type == 'gru':
            self.cell = nn.GRUCell(input_size, hidden_size)

        self.attention = None
        if attention:
            self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.residual = residual

    def forward(self, x, h):
        # x: (batch_size, sequence_length, input_size)
        # h: (batch_size, hidden_size) if gru,
        #    ((batch_size, hidden_size), (batch_size, hidden_size)) if lstm

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        if self.type == 'lstm':
            rnn_output, c = self.cell(x[:, -1], h)
            h_new = (rnn_output, c)
        elif self.type == 'gru':
            rnn_output = self.cell(x[:, -1], h)
            h_new = rnn_output

        if self.attention is not None:
            rnn_output = rnn_output.unsqueeze(1)
            rnn_output, _ = self.attention(rnn_output, x, x)
            rnn_output = rnn_output.squeeze(1)

        rnn_output = self.dropout(rnn_output)
        if self.residual:
            rnn_output = rnn_output + x[:, -1]

        return rnn_output, h_new

    def init_hidden(self, batch_size, device):
        if self.type == 'lstm':
            return (torch.zeros(batch_size, self.cell.hidden_size, device=device),
                    torch.zeros(batch_size, self.cell.hidden_size, device=device))
        elif self.type == 'gru':
            return torch.zeros(batch_size, self.cell.hidden_size, device=device)

    def open_recurrent_connections(self):
        if self.type == 'lstm':
            self.cell.bias_ih[1 * self.cell.hidden_size:2 * self.cell.hidden_size].data.fill_(1.0)
            self.cell.bias_hh[2 * self.cell.hidden_size:2 * self.cell.hidden_size].data.fill_(1.0)
        elif self.type == 'gru':
            self.cell.bias_ih[:1 * self.cell.hidden_size].data.fill_(-1.0)
            self.cell.bias_hh[:1 * self.cell.hidden_size].data.fill_(-1.0)


class InteractiveEncoder(nn.Module):
    def __init__(self,
                 probing_input_dim: None,
                 probing_output_dim: None,
                 vfunc: None,
                 hidden_sizes: None,
                 device: None,
                 activation: None = nn.ReLU,
                 output_size: int = 1,
                 probing_sequence_length: int = 10,
                 num_parallel_ps: int = 10,
                 hidden_size_rnn: int = 1,
                 encoding_size: int = 256,
                 num_rnn_layers: int = 1,
                 recurrent_vfunc: bool = False,
                 probing_input_softmax: bool = False,
                 initial_hidden_state: torch.Tensor = None,
                 probing_input_decoder: torch.nn.Module = None,
                 dropout: float = 0.0,
                 residual: bool = False,
                 static_weight: float = 0.0,
                 interactive_weight: float = 1.0,
                 open_recurrent_connections: bool = False,
                 rnn_cell_type: str = 'lstm'):
        super().__init__()
        self.device = device
        self.probing_output_dim = probing_output_dim
        self.probing_input_dim = probing_input_dim
        self.hidden_size_rnn = hidden_size_rnn
        self.length_abstract_rollout = probing_sequence_length
        self.num_parallel_ps = num_parallel_ps
        self.vfunc = vfunc
        self.encoding_size = encoding_size
        self.recurrence = recurrent_vfunc
        self.initial_hidden_state = initial_hidden_state
        self.probing_input_softmax = probing_input_softmax
        self.static_weight = static_weight
        self.interactive_weight = interactive_weight

        # LSTM
        self.rnn_cells = torch.nn.ModuleList()
        for i in range(num_rnn_layers):
            new_cell = RNNCell(input_size=hidden_size_rnn,
                                          hidden_size=hidden_size_rnn,
                                          dropout=dropout,
                                          residual=residual,
                                          type=rnn_cell_type)
            if open_recurrent_connections:
                new_cell.open_recurrent_connections()
            self.rnn_cells.append(new_cell)
        self.bos = torch.nn.Parameter(torch.zeros(1, hidden_size_rnn))
        self.eos = torch.nn.Parameter(torch.zeros(1, hidden_size_rnn))
        self.h0 = torch.nn.Parameter(torch.zeros(num_rnn_layers, 1, hidden_size_rnn))
        self.c0 = torch.nn.Parameter(torch.zeros(num_rnn_layers, 1, hidden_size_rnn))

        self.probing_encodings = torch.nn.Parameter(torch.randn(probing_sequence_length + 1,
                                                                hidden_size_rnn))

        # transform probing output to lstm input
        probing_output_transform_size = [self.probing_output_dim * self.num_parallel_ps] + [hidden_size_rnn]
        self.probing_output_transform = mlp(probing_output_transform_size, activation, output_activation=nn.Sigmoid,
                                            dropout=dropout)

        # transform lstm output to probing input encodings
        proj_up_size = [hidden_size_rnn] + [self.encoding_size * self.num_parallel_ps]
        self.proj_up = mlp(proj_up_size, activation, nn.ReLU)

        # transform probing state encoding to probing states
        if probing_input_decoder is None:
            probing_input_transform_size = [self.encoding_size] + list(hidden_sizes) + [self.probing_input_dim]
            self.probing_input_decoder = mlp(probing_input_transform_size, activation,
                                             output_activation=nn.Identity if self.probing_input_softmax else nn.Sigmoid,
                                             dropout=dropout)
        else:
            self.probing_input_decoder = probing_input_decoder

        # transform last lstm output to embedding
        output_transform_size = [hidden_size_rnn] + list(hidden_sizes) + [output_size]
        self.output_transform = mlp(output_transform_size, activation, dropout=dropout)

    def forward(self, parameters, diagnostics=False):
        bs_parameters = parameters['linear.weight'].shape[0]

        if diagnostics:
            diagnostics_dict = {
                "probing_inputs": [],
                "probing_outputs": []
            }

        # initialize
        rnn_hidden_states = [cell.init_hidden(bs_parameters, self.device) for cell in self.rnn_cells]
        bos = self.bos.repeat(bs_parameters, 1)
        eos = self.eos.repeat(bs_parameters, 1)
        if self.recurrence:
            if self.initial_hidden_state is not None:
                vfunc_hidden_state = tree_map(lambda x: x.unsqueeze(0).repeat(*[bs_parameters, self.num_parallel_ps] +
                                                                           [1] * (x.ndim - 1)).to(self.device),
                                           self.initial_hidden_state)
            else:
                vfunc_hidden_state = None
        rnn_inputs = [bos.unsqueeze(1)] + [torch.zeros(bs_parameters, 0, self.hidden_size_rnn, device=self.device)
                                           for _ in range(len(self.rnn_cells))]

        # probing rollout
        for probing_step in range(self.length_abstract_rollout + 1):
            # rnn
            for i, lstm_cell in enumerate(self.rnn_cells):
                rnn_output, rnn_hidden_states_new = lstm_cell(rnn_inputs[i], rnn_hidden_states[i])
                rnn_hidden_states[i] = rnn_hidden_states_new
                rnn_inputs[i + 1] = torch.cat([rnn_inputs[i + 1], rnn_output.unsqueeze(1)], dim=1)
            if probing_step == self.length_abstract_rollout:
                rnn_inputs[0] = torch.cat([rnn_inputs[0], eos.unsqueeze(1)], dim=1)
                break

            # generate probing inputs
            static_encodings = self.probing_encodings[probing_step].unsqueeze(0).repeat(bs_parameters, 1)
            proj_up_input = self.interactive_weight * rnn_output + self.static_weight * static_encodings
            proj_up = self.proj_up(proj_up_input).reshape([-1, self.encoding_size])
            probing_inputs = self.probing_input_decoder(proj_up)
            probing_inputs = probing_inputs.reshape([bs_parameters, self.num_parallel_ps] + list(probing_inputs.shape[1:]))
            if self.probing_input_softmax:
                probing_inputs = torch.softmax(probing_inputs, dim=-1)
            if diagnostics:
                diagnostics_dict["probing_inputs"].append(probing_inputs.detach().cpu())

            # give the probing inputs to the vfunc and get the probing outputs
            if self.recurrence:
                probing_inputs = probing_inputs.unsqueeze(2)
                probing_outputs, vfunc_hidden_state = self.vfunc(parameters, probing_inputs,
                                                                 vfunc_hidden_state)
            else:
                probing_outputs = self.vfunc(parameters, probing_inputs)
            if diagnostics:
                diagnostics_dict["probing_outputs"].append(probing_outputs.squeeze(-2).detach().cpu())

            # transform probing outputs to lstm inputs
            probing_outputs = probing_outputs.reshape([-1, self.probing_output_dim*self.num_parallel_ps])
            new_rnn_input = self.probing_output_transform(probing_outputs).unsqueeze(1)
            rnn_inputs[0] = torch.cat([rnn_inputs[0], new_rnn_input], dim=1)

        # final rnn step (with eos input)
        for i, lstm_cell in enumerate(self.rnn_cells):
            rnn_output, rnn_hidden_states_new = lstm_cell(rnn_inputs[i], rnn_hidden_states[i])
            rnn_hidden_states[i] = rnn_hidden_states_new
            rnn_inputs[i + 1] = torch.cat([rnn_inputs[i + 1], rnn_output.unsqueeze(1)], dim=1)
        r = self.output_transform(rnn_output)

        if diagnostics:
            diagnostics_dict["probing_inputs"] = torch.stack(diagnostics_dict["probing_inputs"], dim=1)
            diagnostics_dict["probing_outputs"] = torch.stack(diagnostics_dict["probing_outputs"], dim=1)
            return r, diagnostics_dict
        return r


class NFLSTMParameterEncoder(nn.Module):
    def __init__(self, parameter_shapes, num_np_channels, mlp_hidden_sizes, output_size):
        super().__init__()

        self.parameter_names = [k for k, p in parameter_shapes]
        np_layers = []
        for i in range(len(num_np_channels)):
            if i > 0:
                np_layers.append(TreePointwise(nn.ReLU()))
            np_layers.append(NPLinearLSTM(parameter_shapes=parameter_shapes,
                                          in_channels=4 if i == 0 else num_np_channels[i - 1],
                                          out_channels=num_np_channels[i],
                                          io_embed=True))

        self.np_model = torch.nn.Sequential(*np_layers)
        # print number of parameters of the np_model
        num_params_np = sum(p.numel() for p in self.np_model.parameters() if p.requires_grad)
        print(f'Number of parameters of the NP model: {num_params_np}')

        mlp_hidden_sizes = mlp_hidden_sizes + [output_size]
        mlp_input_size = len(parameter_shapes) * num_np_channels[-1]
        mlp_layers = []
        for i in range(len(mlp_hidden_sizes)):
            if i > 0:
                mlp_layers.append(torch.nn.ReLU())
            mlp_layers.append(torch.nn.Linear(mlp_input_size if i == 0 else mlp_hidden_sizes[i - 1],
                                              mlp_hidden_sizes[i]))
        self.mlp_model = torch.nn.Sequential(*mlp_layers)
        # print number of parameters of the mlp_model
        num_params_mlp = sum(p.numel() for p in self.mlp_model.parameters() if p.requires_grad)
        print(f'Number of parameters of the MLP model: {num_params_mlp}')

    def forward(self, parameters, diagnostics=False):
        np_out = self.np_model(parameters)

        # flatten and take mean symmetry dimensions
        mean_np_out = []
        for k in self.parameter_names:
            o = np_out[k]
            mean_np_out.append(o.view(o.shape[0], o.shape[1], -1).mean(dim=-1))
        mean_np_out = torch.cat(mean_np_out, dim=1)

        mlp_out = self.mlp_model(mean_np_out)
        if diagnostics:
            return mlp_out, {}
        return mlp_out


class FlatParameterEncoder(nn.Module):
    def __init__(self, parameter_shapes, mlp_hidden_sizes):
        super().__init__()
        self.parameter_names = [k for k, p in parameter_shapes]
        num_params = sum([np.prod(p) for (k, p) in parameter_shapes])
        mlp_layers = []
        for i in range(len(mlp_hidden_sizes)):
            if i > 0:
                mlp_layers.append(torch.nn.ReLU())
            mlp_layers.append(torch.nn.Linear(num_params if i == 0 else mlp_hidden_sizes[i - 1],
                                              mlp_hidden_sizes[i]))
        self.mlp_model = torch.nn.Sequential(*mlp_layers)

    def forward(self, parameters, diagnostics=False):
        parameters = [parameters[k].view(parameters[k].shape[0], -1) for k in self.parameter_names]
        parameters = torch.cat(parameters, dim=1)
        mlp_out = self.mlp_model(parameters)
        if diagnostics:
            return mlp_out, {}
        return mlp_out


class ParameterStatisticsEncoder(nn.Module):
    def __init__(self,
                 parameter_shapes,
                 mlp_hidden_sizes,
                 quantiles=(0., 0.25, 0.5, 0.75, 1.),
                 per_layer=True):
        super().__init__()
        num_statistics = 2 + len(quantiles)
        self.register_buffer("quantiles", torch.tensor(quantiles))
        self.per_layer = per_layer
        self.parameter_names = [k for k, p in parameter_shapes]

        if self.per_layer:
            input_size = len(parameter_shapes) * num_statistics
        else:
            input_size = num_statistics

        mlp_layers = []
        for i in range(len(mlp_hidden_sizes)):
            if i > 0:
                mlp_layers.append(torch.nn.ReLU())
            mlp_layers.append(torch.nn.Linear(input_size if i == 0 else mlp_hidden_sizes[i - 1],
                                              mlp_hidden_sizes[i]))
        self.mlp_model = torch.nn.Sequential(*mlp_layers)

    def forward(self, parameters, diagnostics=False):
        parameters = [parameters[name].view(parameters[name].shape[0], -1) for name in self.parameter_names]

        if not self.per_layer:
            parameters = [torch.cat(parameters, dim=1)]

        all_statistics = []
        for p in parameters:
            statistics = []
            statistics.append(torch.mean(p, dim=1).unsqueeze(1))
            statistics.append(torch.var(p, dim=1).unsqueeze(1))
            statistics.append(torch.quantile(p, self.quantiles, dim=1).t())
            statistics = torch.cat(statistics, dim=1)
            all_statistics.append(statistics)
        all_statistics = torch.cat(all_statistics, dim=1)

        mlp_out = self.mlp_model(all_statistics)
        if diagnostics:
            return mlp_out, {}
        return mlp_out


class ParameterTransformerEncoder(nn.Module):
    def __init__(self,
                 parameter_shapes,
                 num_heads,
                 num_transformer_layers,
                 transformer_size,
                 output_size):
        super().__init__()
        # the parameters should have the structure
        #  'lstm_cells.0.i2h.weight': torch.Size([4*lstm_size_0, input_size]),
        #  'lstm_cells.0.i2h.bias': torch.Size([4*lstm_size_0),
        #  'lstm_cells.0.h2h.weight': torch.Size([4*lstm_size_0, lstm_size_0]),
        #  'lstm_cells.0.h2h.bias': torch.Size([4*lstm_size_0]),
        #  'lstm_cells.1.i2h.weight': torch.Size([4*lstm_size_1, lstm_size_0]),
        #  'lstm_cells.1.i2h.bias': torch.Size([4*lstm_size_1]),
        #  'lstm_cells.1.h2h.weight': torch.Size([4*lstm_size_1, lstm_size_1]),
        #  'lstm_cells.1.h2h.bias': torch.Size([4*lstm_size_1]),
        #  ...
        #  'linear.weight': torch.Size([output_size, lstm_size_last]),
        #  'linear.bias': torch.Size([output_size])
        #
        # we concatenate the weights and bias pairs along the input dimension, and for each such pair we create a
        # learneable transformation scaling it to transformer_size

        self.sequence_length = 0
        self.input_transformations = nn.ModuleDict()
        self.parameter_pair_names = []
        for name, shape in parameter_shapes:
            if 'bias' in name:
                continue
            self.parameter_pair_names.append([name, name.replace('.weight', '.bias')])
            transform_name = name.replace('.weight', '').replace('.', '_')
            self.input_transformations[transform_name] = nn.Linear(shape[-1] + 1, transformer_size)
            self.sequence_length += shape[-2]

        transformer_layer = torch.nn.TransformerEncoderLayer(d_model=transformer_size,
                                                             dim_feedforward=4*transformer_size,
                                                             nhead=num_heads,
                                                             batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        self.output_projection = nn.Linear(transformer_size, output_size)
        self.compression_token = nn.Parameter(torch.zeros(1, 1, transformer_size))
        self.positional_encoding = nn.Parameter(torch.randn(1, self.sequence_length + 1, transformer_size))

    def forward(self, parameters, diagnostics=False):
        neuron_sequence = []
        for weight_name, bias_name in self.parameter_pair_names:
            neurons = torch.cat([parameters[weight_name], parameters[bias_name].unsqueeze(-1)], dim=-1)
            neuron_sequence.append(self.input_transformations[weight_name.replace('.weight', '').replace('.', '_')](neurons))
        neuron_sequence.append(self.compression_token.repeat(neurons.shape[0], 1, 1))
        neuron_sequence = torch.cat(neuron_sequence, dim=1)
        neuron_sequence = neuron_sequence + self.positional_encoding

        transformer_output = self.transformer(neuron_sequence)
        output = self.output_projection(transformer_output[:, -1, :])
        if diagnostics:
            return output, {}
        return output