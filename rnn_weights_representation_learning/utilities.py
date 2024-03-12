import torch
import numpy as np
import random
import io
import matplotlib.pyplot as plt
from PIL import Image
from dataclasses import dataclass
from argparse import Namespace
try:
    WANDB_AVAILABLE = True
    import wandb
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class ModelState:
    rnn_weight_encoder: torch.nn.Module
    emulator: torch.nn.Module
    combined_model: torch.nn.Module
    device: str
    rnn_params_shapes: list
    rnn_model: torch.nn.Module
    vfunc: ...
    args: ...
    get_language: ... = None


@dataclass
class SupervisedModelState:
    policy_encoder: torch.nn.Module
    predictor: torch.nn.Module
    clip_loss: torch.nn.Module
    device: str
    use_sequence_encoder: bool
    rnn_params_shapes: list
    rnn_model: torch.nn.Module
    v_actor_func: ...
    args: ...
    get_language: ... = None
    get_target_func: ... = None
    get_losses_func: ... = None


def unflatten_params(params_vec, params_shapes):
    # convert the flat vector of parameters to a dict of named parameters with the correct shapes
    is_batch = len(params_vec.shape) == 2
    if not is_batch:
        params_vec = params_vec.unsqueeze(0)
    batch_size = params_vec.shape[0]

    flat_shapes = [np.prod(shape) for _, shape in params_shapes]

    split_params_vec = params_vec.split(flat_shapes, dim=-1)
    unflattened_params = [p.view(batch_size, *shape) for p, (_, shape) in zip(split_params_vec, params_shapes)]

    if not is_batch:
        unflattened_params = [p.squeeze(0) for p in unflattened_params]

    parameter_dict = {name: param for (name, _), param in zip(params_shapes, unflattened_params)}
    return parameter_dict


def convert_parameter_names(parameter_names):
    # convert the names of the parameters as they appear in the RNNPolicyFast model to the names of the
    # RNNPolicyFunctionalizable model

    # concretely, 'lstm.weight_ih_l0' -> 'lstm_cells.0.i2h.weight', etc.
    new_parameter_names = []

    for name in parameter_names:
        if name.startswith('lstm.'):
            layer = int(name.split('_')[-1][1:])
            if name.startswith('lstm.weight_ih_l'):
                new_name = f'lstm_cells.{layer}.i2h.weight'
            elif name.startswith('lstm.weight_hh_l'):
                new_name = f'lstm_cells.{layer}.h2h.weight'
            elif name.startswith('lstm.bias_ih_l'):
                new_name = f'lstm_cells.{layer}.i2h.bias'
            elif name.startswith('lstm.bias_hh_l'):
                new_name = f'lstm_cells.{layer}.h2h.bias'
        elif name.startswith('linear.weight'):
            new_name = 'linear.weight'
        elif name.startswith('linear.bias'):
            new_name = 'linear.bias'
        else:
            raise Exception(f"Unknown parameter name: {name}")
        new_parameter_names.append(new_name)

    return new_parameter_names


def tree_flatten(tree, is_leaf=None, reference_tree=None):
    def traverse_leaves(node, is_leaf=None, reference_node=None):
        leaves = []
        if is_leaf is not None and is_leaf(node):
            return [node]
        elif type(node) is list or type(node) is tuple:
            for i in range(len(node)):
                leaves.extend(traverse_leaves(node[i], is_leaf,
                                              reference_node[i] if reference_node is not None else None))
        elif type(node) is dict:
            keys = node.keys() if reference_node is None else reference_node.keys()
            for key in keys:
                child = node[key]
                leaves.extend(traverse_leaves(child, is_leaf,
                                              reference_node[key] if reference_node is not None else None))
        else:
            return [node]
        return leaves

    def traverse_structure(node, is_leaf=None, reference_node=None):
        if is_leaf is not None and is_leaf(node):
            return None
        elif type(node) is list or type(node) is tuple:
            structure = [traverse_structure(node[i], is_leaf, reference_node[i] if reference_node is not None else None)
                         for i in range(len(node))]
            if type(node) is tuple:
                structure = tuple(structure)
        elif type(node) is dict:
            keys = node.keys() if reference_node is None else reference_node.keys()
            structure = {key: traverse_structure(node[key], is_leaf,
                                                 reference_node[key] if reference_node is not None else None)
                         for key in keys}
        else:
            return None
        return structure

    leaves = traverse_leaves(tree, is_leaf, reference_tree)
    structure = traverse_structure(tree, is_leaf, reference_tree)
    return leaves, structure


def tree_fill(leaves, structure, is_leaf=None):
    if is_leaf is not None and is_leaf(structure):
        return leaves.pop(0)
    if type(structure) is list:
        filled_structure = [tree_fill(leaves, child, is_leaf) for child in structure]
    elif type(structure) is tuple:
        filled_structure = tuple([tree_fill(leaves, child, is_leaf) for child in structure])
    elif type(structure) is dict:
        filled_structure = {key: tree_fill(leaves, child, is_leaf) for key, child in structure.items()}
    else:
        return leaves.pop(0)
    return filled_structure


def tree_map(f, tree, rest=[], is_leaf=None):
    leaves, structure = tree_flatten(tree, is_leaf)
    all_leaves = [leaves] + [tree_flatten(r, is_leaf, tree)[0] for r in rest]
    if len(rest) > 0:
        assert len(leaves) == len(all_leaves[1])
    processed_leaves = [f(*xs) for xs in zip(*all_leaves)]
    processed_tree = tree_fill(processed_leaves, structure)
    return processed_tree


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fix_seed(func):
    """The wrapped function will be executed with the same seed,
    the original seed will be restored after execution."""
    def wrapper(*args, **kwargs):
        # sample a new seed
        new_seed = random.randint(0, 2**32 - 1)
        if "seed" in kwargs:
            seed_everything(kwargs["seed"])
            del kwargs["seed"]
        else:
            seed_everything(1234)
        result = func(*args, **kwargs)
        seed_everything(new_seed)
        return result

    return wrapper


HASH_KEY_1 = torch.tensor(0x58a849af6cbf585a, dtype=torch.long)
HASH_KEY_2 = torch.tensor(0x5ca118cb4d828a5e, dtype=torch.long)
def hash64(x):
    # Convert the input to a 64-bit unsigned integer tensor
    x = torch.tensor(x, dtype=torch.long)
    # Use the PyTorch's bitwise XOR operator to generate a hash value
    # We use two random 64-bit integers as the hash key
    # Note: the key can be any fixed 64-bit integer or a set of integers
    h = torch.bitwise_xor(x ^ HASH_KEY_1.to(x.device), HASH_KEY_2.to(x.device))
    return h


def figure2PIL(fig=None):
    """Converts a matplotlib figure to a PIL Image and returns it"""
    if fig is None:
        fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


class AverageMeter(object):
    """Computes and stores the average, max and current value"""
    def __init__(self, keep_track_of_extrema=True):
        self.keep_track_of_extrema = keep_track_of_extrema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.min = float('inf')
        self.max = float('-inf')
        self.m = 0  # for running variance calculation (Knuth's algorithm)
        self.v = 0  # for running variance calculation
        self.var = 0
        self.count = 0
        self.just_reset = True

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.keep_track_of_extrema:
            if val < self.min:
                self.min = val
            if val > self.max:
                self.max = val
            if self.just_reset:
                self.just_reset = False
                self.m = val
            old_m = self.m
            self.m = old_m + (val - old_m) / self.count
            self.v = self.v + (val - old_m) * (val - self.m)
            if self.count - n > 0:
                self.var = self.v / (self.count - n)
        self.avg = self.sum / self.count


class MultiMeter(object):
    """Acts as average meter for multiple values at once"""
    def __init__(self, name_list=[], keep_track_of_extrema_list=None):
        self.meters = {}
        for i, name in enumerate(name_list):
            self.meters[name] = AverageMeter(keep_track_of_extrema_list[i]
                                             if keep_track_of_extrema_list is not None else True)

    def add_meter(self, key, keep_track_of_extrema=True):
        if key in self.meters.keys():
            raise Exception(f"The meter '{key}' already exists!")
        self.meters[key] = AverageMeter(keep_track_of_extrema)

    def reset(self, name_list=None):
        if name_list is None:
            for m in self.meters.values():
                m.reset()
        else:
            for name in name_list:
                self.meters[name].reset()

    def update(self, val_dict, n=1):
        for k, v in val_dict.items():
            if k not in self.meters.keys():
                self.add_meter(k)
            self.meters[k].update(v, n=n)

    def __getitem__(self, key):
        return self.meters[key]


class MockWandb(object):
    def __init__(self, print_to_console, config):
        self.print = print_to_console
        self.config = config

    def __getattr__(self, attr):
        try:
            return super(MockWandb, self).__getattr__(attr)
        except AttributeError:
            return self.__get_global_handler(attr)

    def __get_global_handler(self, name):
        # Do anything that you need to do before simulating the method call
        handler = self.__global_handler
        if name == "config":
            return self.config
        return handler

    def __global_handler(self, *args, **kwargs):
        if self.print:
            print(str(args))
            print(str(kwargs))


class Logger:
    def __init__(self, enabled=True, print_logs_to_console=False, config={}, **kwargs):
        self.enabled = enabled
        if self.enabled and WANDB_AVAILABLE:
            wandb.init(config=config, **kwargs)
        elif self.enabled:
            raise Exception("wandb is not available, but it is enabled!")
            self.enabled = False

        config = Namespace(**config)
        self.mock = MockWandb(print_to_console=print_logs_to_console, config=config)

    def __call__(self, *args, **kwargs):
        if self.enabled:
            return wandb
        else:
            return self.mock