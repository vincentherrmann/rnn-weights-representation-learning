import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange

from utilities import tree_map


def set_init_(*layers):
    in_chan = 0
    for layer in layers:
        if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
            in_chan += layer.in_channels
        elif isinstance(layer, nn.Linear):
            in_chan += layer.in_features
        else:
            raise NotImplementedError(f"Unknown layer type {type(layer)}")
    bd = math.sqrt(1 / in_chan)
    for layer in layers:
        nn.init.uniform_(layer.weight, -bd, bd)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias, -bd, bd)


def shape_params_symmetry(params):
    """Reshape so last 2 dims have symmetry, channel dims have all nonsymmetry.
    E.g., for conv weights we reshape (B, C, out, in, h, w) -> (B, C * h * w, out, in)
    """
    original_shapes = {k: p.shape for k, p in params.items()}
    reshaped_params = {}
    for name, parameter in params.items():
        if len(parameter.shape) <= 4:
            reshaped_params[name] = parameter
        else:
            reshaped_params[name] = rearrange(parameter, "b c o i h w -> b (c h w) o i")
    return reshaped_params, original_shapes


def unshape_params_symmetry(params, original_shapes):
    """Reverse shape_params_symmetry"""
    weights = params[::2]
    bias = params[1::2]
    unreshaped_params = []
    for i, weight in enumerate(weights):
        if len(original_shapes[i]) <= 4:  # mlp weight matrix:
            unreshaped_params.append(weight)
        else:
            h, w = original_shapes[i][-2:]
            unreshaped_params.append(rearrange(weight, "b (c h w) o i -> b c o i h w", h=h, w=w))
        unreshaped_params.append(bias[i])
    return unreshaped_params


def prepare_lstm_params_for_np(params):
    batch_dim = params['lstm_cells.0.i2h.weight'].dim() > 2
    channeled_parameters = {key: p.view(p.shape[0], 4, -1, *p.shape[2:]) if batch_dim else p.view(4, -1, *p.shape[1:])
                            for key, p in params.items() if key.startswith('lstm_cells')}
    channeled_parameters['linear.weight'] = params['linear.weight'].unsqueeze(-3).repeat(*[1, 4, 1, 1] if batch_dim
                                                else [4, 1, 1])
    channeled_parameters['linear.bias'] = params['linear.bias'].unsqueeze(-2).repeat(*[1, 4, 1] if batch_dim
                                                else [4, 1])
    return channeled_parameters


class TreePointwise(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        return tree_map(f=lambda x: self.op(x), tree=x)


class NPLinear(nn.Module):
    """Assume permutation symmetry of input and output layers, as well as hidden."""
    def __init__(self, parameter_shapes, in_channels, out_channels, io_embed=False):
        super().__init__()
        self.c_in, self.c_out = in_channels, out_channels
        self.parameter_shapes = parameter_shapes
        L = len(parameter_shapes) // 2
        weight_shapes = parameter_shapes[::2]
        bias_shapes = parameter_shapes[1::2]
        filter_facs = [int(np.prod(shapes[2:])) for shapes in weight_shapes]
        n_rc_inp = L + sum(filter_facs)
        for i in range(L):
            fac_i = filter_facs[i]
            # pointwise
            self.add_module(f"layer_{i}", nn.Conv2d(fac_i * in_channels, fac_i * out_channels, 1))
            # broadcasts over rows and columns
            self.add_module(f"layer_{i}_rc", nn.Linear(n_rc_inp * in_channels, fac_i * out_channels))

            # broadcasts over rows or columns
            row_in, col_in = fac_i * in_channels, (fac_i + 1) * in_channels
            if i > 0:
                fac_im1 = filter_facs[i - 1]
                row_in += (fac_im1 + 1) * in_channels
            if i < L - 1:
                fac_ip1 = filter_facs[i + 1]
                col_in += fac_ip1 * in_channels
            self.add_module(f"layer_{i}_r", nn.Conv1d(row_in, fac_i * out_channels, 1))
            self.add_module(f"layer_{i}_c", nn.Conv1d(col_in, fac_i * out_channels, 1))

            # pointwise
            self.add_module(f"bias_{i}", nn.Conv1d(col_in, out_channels, 1))
            self.add_module(f"bias_{i}_rc", nn.Linear(n_rc_inp * in_channels, out_channels))
            set_init_(
                getattr(self, f"layer_{i}"),
                getattr(self, f"layer_{i}_rc"),
                getattr(self, f"layer_{i}_r"),
                getattr(self, f"layer_{i}_c"),
            )
            set_init_(getattr(self, f"bias_{i}"), getattr(self, f"bias_{i}_rc"))
        self.io_embed = io_embed
        if io_embed:
            # initialize learned position embeddings to break input and output symmetry
            n_in = weight_shapes[0][1]
            n_out = weight_shapes[-1][0]
            self.in_embed = nn.Parameter(torch.randn(1, filter_facs[0] * in_channels, 1, n_in))
            self.out_embed = nn.Parameter(torch.randn(1, filter_facs[-1] * in_channels, n_out, 1))
            self.out_bias_embed = nn.Parameter(torch.randn(1, in_channels, n_out))

    def forward(self, parameters):
        parameters, original_shapes = shape_params_symmetry(parameters)
        batch_size = parameters[0].shape[0]
        if self.io_embed:
            parameters = (parameters[0] + self.in_embed,
                          *parameters[1:-2],
                          parameters[-2] + self.out_embed,
                          parameters[-1] + self.out_bias_embed)
        weights = parameters[::2]
        biases = parameters[1::2]
        row_means = [w.mean(dim=-2) for w in weights]
        col_means = [w.mean(dim=-1) for w in weights]
        rowcol_means = [w.mean(dim=(-2, -1)) for w in weights]  # (B, C_in)
        bias_means = [b.mean(dim=-1) for b in biases]  # (B, C_in)
        wb_means = torch.cat(rowcol_means + bias_means, dim=-1)  # (B, 2 * C_in * n_layers)

        wsfeat = list(zip(weights, biases))
        out_parameters = []
        for i, (weight, bias) in enumerate(wsfeat):
            # pointwise
            z1 = getattr(self, f"layer_{i}")(weight)  # (B, C_out, nrow, ncol)
            # wb all layers
            z2 = getattr(self, f"layer_{i}_rc")(wb_means)[..., None, None]
            row_bdcst = [row_means[i]]  # (B, C_in, ncol)
            col_bdcst = [col_means[i], bias]  # (B, 2 * C_in, nrow)
            if i > 0:
                ncols = row_bdcst[0].shape[-1]
                if ncols == col_means[i-1].shape[-1]:
                    row_bdcst.extend([col_means[i-1], biases[i-1]])  # (B, C_in, ncol)
                else:
                    # if the number of columns in the previous layer is different, there might be a flattening of the
                    # feature map. In this case, we repeat the previous row_means and bias to match the number of
                    # columns in the current layer.
                    if ncols % col_means[i-1].shape[-1] == 0:
                        nrepeat = ncols // col_means[i-1].shape[-1]
                        wr = col_means[i-1].repeat(1, 1, nrepeat)
                        br = biases[i-1].repeat(1, 1, nrepeat)
                        row_bdcst.extend([wr, br])
                    else:
                        raise ValueError("Error in the shapes during the broadcast of the previous layer")
            if i < len(weights) - 1:
                nrows = col_bdcst[0].shape[-1]
                if nrows == row_means[i+1].shape[-1]:
                    col_bdcst.append(row_means[i+1]) # (B, C_in, nrow)
                else:
                    # if the number of columns in the next layer is different, there might be a flattening of the
                    # feature map. In this case, we take the reshape the next following row_means and take the mean over
                    # the feature map dimensions.
                    if row_means[i+1].shape[-1] % nrows == 0:
                        r = row_means[i+1].view(batch_size, row_means[i+1].shape[1], -1, nrows).mean(dim=-2)
                        col_bdcst.append(r)
                    else:
                        raise ValueError("Error in the shapes during the broadcast of the next layer")
            row_bdcst = torch.cat(row_bdcst, dim=-2)
            col_bdcst = torch.cat(col_bdcst, dim=-2)
            z3 = getattr(self, f"layer_{i}_r")(row_bdcst).unsqueeze(-2)  # (B, C_out, 1, ncol)
            z4 = getattr(self, f"layer_{i}_c")(col_bdcst).unsqueeze(-1)  # (B, C_out, nrow, 1)
            out_parameters.append(z1 + z2 + z3 + z4)

            u1 = getattr(self, f"bias_{i}")(col_bdcst)  # (B, C_out, nrow)
            u2 = getattr(self, f"bias_{i}_rc")(wb_means).unsqueeze(-1)  # (B, C_out, 1)
            out_parameters.append(u1 + u2)
        out_parameters = unshape_params_symmetry(out_parameters, original_shapes)
        return out_parameters

    def __repr__(self):
        return f"NPLinear(in_channels={self.c_in}, out_channels={self.c_out})"


class NPLinearLSTM(nn.Module):
    """Assume permutation symmetry of input and output layers, as well as hidden."""
    def __init__(self, parameter_shapes, in_channels, out_channels, io_embed=False):
        super().__init__()
        self.c_in, self.c_out = in_channels, out_channels
        self.parameter_shapes = parameter_shapes
        # parameter shapes should have the structure
        # [l0_i2h_weight, l0_i2h_bias, l0_h2h_weight, l0_h2h_bias, l1_i2h_weight, ...]
        weight_shapes = [shape for k, shape in parameter_shapes if k.endswith("weight")]

        # i2h (and output) layers
        i2h_weight_shapes = weight_shapes[::2]
        i2h_L = len(i2h_weight_shapes)
        self.num_lstm_layers = i2h_L - 1
        for l in range(i2h_L):
            # pointwise (convolution with kernel size 1)
            self.add_module(f"i2h_layer_{l}", nn.Conv2d(in_channels, out_channels, 1))
            # broadcasts over rows and columns
            self.add_module(f"i2h_layer_{l}_rc", nn.Linear(len(parameter_shapes) * in_channels,  out_channels))

            # broadcasts over rows or columns
            # row_in: curr_i2h_rows, (prev_i2h_cols, prev_i2h_bias, prev_h2h_cols, prev_h2h_bias, p_h2h_rows)
            # col_in: curr_i2h_cols, curr_i2h_bias, (curr_h2h_cols, curr_h2h_bias, curr_h2h_rows, next_i2h_rows)
            row_in, col_in = 1 * in_channels, 2 * in_channels
            if l > 0:
                row_in += 5 * in_channels
            if l < i2h_L - 1:
                col_in += 4 * in_channels
            self.add_module(f"i2h_layer_{l}_r", nn.Conv1d(row_in, out_channels, 1))
            self.add_module(f"i2h_layer_{l}_c", nn.Conv1d(col_in, out_channels, 1))

            # pointwise
            self.add_module(f"i2h_bias_{l}", nn.Conv1d(col_in, out_channels, 1))
            self.add_module(f"i2h_bias_{l}_rc", nn.Linear(len(parameter_shapes) * in_channels, out_channels))
            set_init_(
                getattr(self, f"i2h_layer_{l}"),
                getattr(self, f"i2h_layer_{l}_rc"),
                getattr(self, f"i2h_layer_{l}_r"),
                getattr(self, f"i2h_layer_{l}_c"),
            )
            set_init_(
                getattr(self, f"i2h_bias_{l}"),
                getattr(self, f"i2h_bias_{l}_rc")
            )
        # h2h layers
        h2h_weight_shapes = weight_shapes[1::2]
        h2h_L = len(h2h_weight_shapes)
        for l in range(h2h_L):
            # pointwise (convolution with kernel size 1)
            self.add_module(f"h2h_layer_{l}", nn.Conv2d(in_channels, out_channels, 1))
            # broadcasts over rows and columns
            self.add_module(f"h2h_layer_{l}_rc", nn.Linear(len(parameter_shapes) * in_channels, out_channels))

            # broadcasts over rows or columns
            # row_in: curr_h2h_rows, curr_2h2_cols, curr_h2h_bias, curr_i2h_cols, curr_i2h_bias
            # col_in: curr_h2h_cols, curr_h2h_bias, next_h2h_rows(, next_i2h_rows)
            row_in, col_in = 5 * in_channels, 3 * in_channels
            if l < h2h_L - 1:
                col_in += in_channels
            self.add_module(f"h2h_layer_{l}_r", nn.Conv1d(row_in, out_channels, 1))
            self.add_module(f"h2h_layer_{l}_c", nn.Conv1d(col_in, out_channels, 1))

            # pointwise
            self.add_module(f"h2h_bias_{l}", nn.Conv1d(col_in, out_channels, 1))
            self.add_module(f"h2h_bias_{l}_rc", nn.Linear(len(parameter_shapes) * in_channels, out_channels))
            set_init_(
                getattr(self, f"h2h_layer_{l}"),
                getattr(self, f"h2h_layer_{l}_rc"),
                getattr(self, f"h2h_layer_{l}_r"),
                getattr(self, f"h2h_layer_{l}_c"),
            )
            set_init_(
                getattr(self, f"h2h_bias_{l}"),
                getattr(self, f"h2h_bias_{l}_rc")
            )

        self.io_embed = io_embed
        if io_embed:
            # initialize learned position embeddings to break input and output symmetry
            n_in = i2h_weight_shapes[0][1]
            n_out = i2h_weight_shapes[-1][0]
            self.in_embed = nn.Parameter(torch.randn(1, in_channels, 1, n_in))
            self.out_embed = nn.Parameter(torch.randn(1, in_channels, n_out, 1))
            self.out_bias_embed = nn.Parameter(torch.randn(1, in_channels, n_out))

    def forward(self, parameters):
        # parameters: (B, C_in, nrow, ncol)
        parameters, _ = shape_params_symmetry(parameters)
        if self.io_embed:
            parameters['lstm_cells.0.i2h.weight'] = parameters['lstm_cells.0.i2h.weight'] + self.in_embed
            parameters['linear.weight'] = parameters['linear.weight'] + self.out_embed
            parameters['linear.bias'] = parameters['linear.bias'] + self.out_bias_embed
        weights = []
        biases = []
        for l in range(self.num_lstm_layers):
            weights.append(parameters[f'lstm_cells.{l}.i2h.weight'])
            weights.append(parameters[f'lstm_cells.{l}.h2h.weight'])
            biases.append(parameters[f'lstm_cells.{l}.i2h.bias'])
            biases.append(parameters[f'lstm_cells.{l}.h2h.bias'])
        weights.append(parameters['linear.weight'])
        biases.append(parameters['linear.bias'])
        row_means = [w.mean(dim=-2) for w in weights]
        col_means = [w.mean(dim=-1) for w in weights]
        rowcol_means = [w.mean(dim=(-2, -1)) for w in weights]  # (B, C_in)
        bias_means = [b.mean(dim=-1) for b in biases]  # (B, C_in)
        wb_means = torch.cat(rowcol_means + bias_means, dim=-1)  # (B, 2 * C_in * n_layers)

        weights_i2h = weights[0::2]
        weights_h2h = weights[1::2]
        biases_i2h = biases[0::2]
        biases_h2h = biases[1::2]
        row_means_i2h = row_means[0::2]
        row_means_h2h = row_means[1::2]
        col_means_i2h = col_means[0::2]
        col_means_h2h = col_means[1::2]

        # i2h
        i2h_out = []
        for l in range(len(weights_i2h)):
            z1 = getattr(self, f"i2h_layer_{l}")(weights_i2h[l])  # (B, C_out, nrow, ncol)
            z2 = getattr(self, f"i2h_layer_{l}_rc")(wb_means)[..., None, None]
            # row_in: curr_i2h_rows, (prev_i2h_cols, prev_i2h_bias, prev_h2h_cols, prev_h2h_bias, p_h2h_rows)
            # col_in: curr_i2h_cols, curr_i2h_bias, (curr_h2h_cols, curr_h2h_bias, curr_h2h_rows, next_i2h_rows)
            row_bdcst = [row_means_i2h[l]]  # (B, C_in, ncol)
            col_bdcst = [col_means_i2h[l], biases_i2h[l]]  # (B, 2 * C_in, nrow)
            if l > 0:
                row_bdcst.extend([col_means_i2h[l-1], biases_i2h[l-1],
                                  col_means_h2h[l-1], biases_h2h[l-1],
                                  row_means_h2h[l-1]])  # (B, C_in, ncol)
            if l < len(weights_i2h) - 1:
                col_bdcst.extend([col_means_h2h[l], biases_h2h[l],
                                  row_means_h2h[l],
                                  row_means_i2h[l+1]])  # (B, C_in, nrow)
            col_bdcst = torch.cat(col_bdcst, dim=-2)
            z3 = getattr(self, f"i2h_layer_{l}_r")(torch.cat(row_bdcst, dim=-2)).unsqueeze(-2)  # (B, C_out, 1, ncol)
            z4 = getattr(self, f"i2h_layer_{l}_c")(col_bdcst).unsqueeze(-1)  # (B, C_out, nrow, 1)
            i2h_out.append(z1 + z2 + z3 + z4)

            u1 = getattr(self, f"i2h_bias_{l}")(col_bdcst)  # (B, C_out, nrow)
            u2 = getattr(self, f"i2h_bias_{l}_rc")(wb_means).unsqueeze(-1)  # (B, C_out, 1)
            i2h_out.append(u1 + u2)

        # h2h
        h2h_out = []
        for l in range(len(weights_h2h)):
            z1 = getattr(self, f"h2h_layer_{l}")(weights_h2h[l])
            z2 = getattr(self, f"h2h_layer_{l}_rc")(wb_means)[..., None, None]
            # row_in: curr_h2h_rows, curr_2h2_cols, curr_h2h_bias, curr_i2h_cols, curr_i2h_bias
            # col_in: curr_h2h_cols, curr_h2h_bias, next_h2h_rows(, next_i2h_rows)
            row_bdcst = [row_means_h2h[l], col_means_h2h[l], biases_h2h[l], col_means_i2h[l], biases_i2h[l]]
            col_bdcst = [col_means_h2h[l], biases_h2h[l], row_means_h2h[l]]
            if l < len(weights_h2h) - 1:
                col_bdcst.append(row_means_i2h[l+1])
            col_bdcst = torch.cat(col_bdcst, dim=-2)
            z3 = getattr(self, f"h2h_layer_{l}_r")(torch.cat(row_bdcst, dim=-2)).unsqueeze(-2)
            z4 = getattr(self, f"h2h_layer_{l}_c")(col_bdcst).unsqueeze(-1)
            h2h_out.append(z1 + z2 + z3 + z4)

            u1 = getattr(self, f"h2h_bias_{l}")(col_bdcst)
            u2 = getattr(self, f"h2h_bias_{l}_rc")(wb_means).unsqueeze(-1)
            h2h_out.append(u1 + u2)

        # [l0_i2h_weight, l0_i2h_bias, l0_h2h_weight, l0_h2h_bias, l1_i2h_weight, ...]
        out_parameters = {}
        for i in range(len(h2h_out) // 2):
            out_parameters[f"lstm_cells.{i}.i2h.weight"] = i2h_out[2*i]
            out_parameters[f"lstm_cells.{i}.i2h.bias"] = i2h_out[2*i+1]
            out_parameters[f"lstm_cells.{i}.h2h.weight"] = h2h_out[2*i]
            out_parameters[f"lstm_cells.{i}.h2h.bias"] = h2h_out[2*i+1]
        out_parameters["linear.weight"] = i2h_out[-2]
        out_parameters["linear.bias"] = i2h_out[-1]
        return out_parameters

    def __repr__(self):
        return f"NPLinear(in_channels={self.c_in}, out_channels={self.c_out})"


def permuteLSTMNeurons(parameters, layer, neuron_a, neuron_b, switching_dict=None):
    layer = layer
    layer_size = parameters[layer*2].shape[0] // 4
    neuron_a = neuron_a + torch.arange(4) * layer_size
    neuron_b = neuron_b + torch.arange(4) * layer_size

    i2h_weights = parameters[::4][:4]
    i2h_biases = parameters[1::4][:4]
    h2h_weights = parameters[2::4]
    h2h_biases = parameters[3::4]

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
        temp = i2h_weights[layer][neuron_b].clone()
        i2h_weights[layer][neuron_b] = i2h_weights[layer][neuron_a].clone()
        i2h_weights[layer][neuron_a] = temp

    if switching_dict["current_h2h_weights"]:
        temp = h2h_weights[layer][neuron_b].clone()
        h2h_weights[layer][neuron_b] = h2h_weights[layer][neuron_a].clone()
        h2h_weights[layer][neuron_a] = temp

    # switch columns in current layer's biases
    if switching_dict["current_i2h_biases"]:
        temp = i2h_biases[layer][neuron_b].clone()
        i2h_biases[layer][neuron_b] = i2h_biases[layer][neuron_a].clone()
        i2h_biases[layer][neuron_a] = temp

    if switching_dict["current_h2h_biases"]:
        temp = h2h_biases[layer][neuron_b].clone()
        h2h_biases[layer][neuron_b] = h2h_biases[layer][neuron_a].clone()
        h2h_biases[layer][neuron_a] = temp

    # switch rows in next layer's weights
    if switching_dict["next_i2h_weights"]:
        temp = i2h_weights[layer + 1][:, neuron_b[0]].clone()
        i2h_weights[layer + 1][:, neuron_b[0]] = i2h_weights[layer + 1][:, neuron_a[0]].clone()
        i2h_weights[layer + 1][:, neuron_a[0]] = temp

    if switching_dict["next_h2h_weights"]:
        temp = h2h_weights[layer][:, neuron_b[0]].clone()
        h2h_weights[layer][:, neuron_b[0]] = h2h_weights[layer][:, neuron_a[0]].clone()
        h2h_weights[layer][:, neuron_a[0]] = temp

    return parameters
