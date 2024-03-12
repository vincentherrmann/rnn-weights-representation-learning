import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
import json
import matplotlib.pyplot as plt
from npy_append_array import NpyAppendArray
from torchvision import datasets, transforms

from utilities import hash64, seed_everything
from modules_rnn_models import RNNFast, PiecewiseLinearSchedule


def parse_args():
    parser = argparse.ArgumentParser()

    # global
    parser.add_argument("--seed", type=int, default=-1,
                        help="seed of the experiment")
    parser.add_argument("--use_cuda", type=int, default=1,
                        help="whether to use cuda")
    parser.add_argument("--store", type=int, default=0,
                        help="whether to store the dataset")
    parser.add_argument("--tile_size", type=int, default=4,
                        help="tile size")
    parser.add_argument("--num_training_runs", type=int, default=250,
                        help="number of training runs")
    parser.add_argument("--rnn_hidden_size", type=int, default=32,
                        help="policy hidden size")
    parser.add_argument("--rnn_num_layers", type=int, default=2,
                        help="policy num layers")
    args = parser.parse_args()
    return args


class TiledMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        # remove tile_w and tile_h from kwargs
        self.tile_w = kwargs.pop("tile_w", None)
        self.tile_h = kwargs.pop("tile_h", None)

        self.bos = torch.zeros(1, self.tile_w * self.tile_h)

        super().__init__(*args, **kwargs)
        #self.tile_w = kwargs.get("tile_w", 1)
        #self.tile_h = kwargs.get("tile_h", 1)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        num_tiles_w = image.shape[1] // self.tile_w
        num_tiles_h = image.shape[2] // self.tile_h
        image = image.view(num_tiles_w, self.tile_w, num_tiles_h, self.tile_h)
        image = image.transpose(1, 2).contiguous()
        image = image.view(num_tiles_w * num_tiles_h, self.tile_w * self.tile_h)

        image = torch.cat([self.bos, image], dim=0)
        return image, label


def convert_seq_to_image(sample_sequences, tile_size=4):
    batch_size = sample_sequences.shape[0]
    num_tiles_w = 28 // tile_size
    num_tiles_h = 28 // tile_size
    seq_length = 28*28 // tile_size**2
    sample_sequences = sample_sequences[:, -seq_length:]

    sample_sequences = sample_sequences.view(batch_size, num_tiles_w, num_tiles_h,
                                             tile_size, tile_size)
    sample_sequences = sample_sequences.transpose(2, 3).contiguous()
    images = sample_sequences.view(batch_size, tile_size * num_tiles_w, tile_size * num_tiles_h)
    return images


def evaluate_model(model, eval_dataset, num_save=100, max_num_iterations=12, device='cpu'):
    with torch.no_grad():
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        losses = []
        saved_data = []
        saved_logits = []
        saved_targets = []
        num_saved = 0
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1000, shuffle=True)
        for batch_idx, (data, target) in enumerate(eval_dataloader):
            data, target = data.to(device), target.to(device)
            logits, _ = model(data)
            repeated_target = target.unsqueeze(1).repeat(1, logits.shape[1])

            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), repeated_target.flatten())
            losses.append(loss.item())

            predictions = logits[:, -1].argmax(dim=1)
            correct_predictions += (predictions == target).sum().item()
            total_predictions += target.shape[0]

            if num_saved < num_save:
                saved_data.append(data[:num_save])
                saved_logits.append(logits[:num_save])
                saved_targets.append(target[:num_save])
                num_saved += min(target.shape[0], num_save)
            if batch_idx >= max_num_iterations:
                break
        model.train()
        eval_loss = np.mean(losses)
        eval_accuracy = correct_predictions / total_predictions
        saved_data = torch.cat(saved_data, dim=0)
        saved_logits = torch.cat(saved_logits, dim=0)
        saved_targets = torch.cat(saved_targets, dim=0)
    return eval_loss, eval_accuracy, saved_data, saved_logits, saved_targets



def main(args):
    if args.seed >= 0:
        seed_everything(args.seed)
    num_training_runs = args.num_training_runs
    snapshot_steps = [0, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000]
    batch_size = 32
    indices = torch.arange(60_000)
    hash_indices = hash64(indices)
    train_indices = indices[hash_indices % 10 < 8]
    eval_indices = indices[hash_indices % 10 >= 8]

    if args.use_cuda > 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    dataset_name = f"sequential_mnist_tile_{args.tile_size}x{args.tile_size}_model_{args.rnn_num_layers}x{args.rnn_hidden_size}"
    if args.seed >= 0:
        dataset_name += f"_seed_{args.seed}"

    if args.store:
        if not os.path.exists("datasets"):
            os.mkdir("datasets")

        if not os.path.exists(f"datasets/{dataset_name}"):
            os.mkdir(f"datasets/{dataset_name}")
        else:
            raise ValueError(f"Dataset {dataset_name} already exists")
            return

        sequence_storage = NpyAppendArray(f"datasets/{dataset_name}/sequences.npy", delete_if_exists=True)
        logits_storage = NpyAppendArray(f"datasets/{dataset_name}/logits.npy", delete_if_exists=True)
        parameter_storage = NpyAppendArray(f"datasets/{dataset_name}/parameters.npy", delete_if_exists=True)
        num_step_storage = NpyAppendArray(f"datasets/{dataset_name}/num_step.npy", delete_if_exists=True)
        accuracy_losses_wd_storage = NpyAppendArray(f"datasets/{dataset_name}/accuracy_losses.npy", delete_if_exists=True)
        task_storage = NpyAppendArray(f"datasets/{dataset_name}/tasks.npy", delete_if_exists=True)

        args_dict = vars(args)
        with open(f"datasets/{dataset_name}/args.json", 'w') as f:
            json.dump(args_dict, f)


    tic = time.time()
    for n in range(num_training_runs):
        print(f"Run {n+1}/{num_training_runs}")

        task = torch.rand(1).item() * 360
        print(f"task: rotate by {task} degrees")

        transform = transforms.Compose([
            transforms.RandomRotation([task, task]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = TiledMNIST(root="./data", train=True, download=True,
                             transform=transform,
                             tile_w=args.tile_size, tile_h=args.tile_size)
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        eval_dataset = torch.utils.data.Subset(dataset, eval_indices)

        model = RNNFast(hidden_size=args.rnn_hidden_size, input_size=args.tile_size**2,
                              output_size=10,
                              num_layers=args.rnn_num_layers, input_batch=True)
        model = model.to(device)
        possible_weight_decay = [0., 0.001, 0.01, 0.1]
        weight_decay = possible_weight_decay[np.random.randint(len(possible_weight_decay))]
        print("weight decay:", weight_decay)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=weight_decay)
        lr_schedule = PiecewiseLinearSchedule(timesteps=[snapshot_steps[0], snapshot_steps[-2], snapshot_steps[-1]],
                                              values=[0.01, 0.003, 0.0003])

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule(0)

        step = 0
        for epoch in range(10_000):
            dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                repeated_target = target.unsqueeze(1).repeat(1, output.shape[1])
                loss = F.cross_entropy(output.view(-1, output.shape[-1]), repeated_target.flatten())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # set learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_schedule(step)

                if step in snapshot_steps:
                    train_loss, train_accuracy, saved_data, saved_logits, saved_targets = evaluate_model(model,
                                                                                                         train_dataset,
                                                                                                         num_save=100,
                                                                                                         device=device)
                    eval_loss, eval_accuracy, _, _, _ = evaluate_model(model,
                                                                       eval_dataset,
                                                                       num_save=100,
                                                                       device=device)
                    print(f'step {step}, train_loss: {train_loss}, eval_loss: {eval_loss}, eval accuracy: {eval_accuracy*100:.2f}%')
                    if args.store:
                        sequences = saved_data.cpu().to(torch.float16)
                        sequence_storage.append(sequences.unsqueeze(0).detach().cpu().numpy())
                        logits = saved_logits.cpu().to(torch.float16)
                        logits_storage.append(logits.unsqueeze(0).detach().cpu().numpy())
                        parameter_storage.append(torch.cat([p.detach().cpu().flatten() for p in model.parameters()]).unsqueeze(0).cpu().numpy())

                        num_step = np.array([n, step])
                        num_step_storage.append(num_step.reshape(1, -1))

                        accuracy_losses_wd = np.array([eval_accuracy, train_loss, eval_loss, weight_decay])
                        accuracy_losses_wd_storage.append(accuracy_losses_wd.reshape(1, -1))

                        task_storage.append(np.array(task).reshape(1, -1))
                        print("stored")
                    else:
                        print("not stored")
                    pass

                if step >= snapshot_steps[-1] + 1:
                    break
                step += 1
            if step >= snapshot_steps[-1] + 1:
                break
        toc = time.time()
        print(f"Time elapsed: {toc - tic} seconds")
        tic = toc

    # transformations: rotate by 15 degress, to tensor, normalize
    transform = transforms.Compose([
        transforms.RandomRotation([15, 15]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = TiledMNIST(root="./data", train=True, download=True,
                         transform=transform,
                         tile_w=args.tile_size, tile_h=args.tile_size)

    example = dataset[0][0]
    print(example.shape)

    image = convert_seq_to_image(example.unsqueeze(0), tile_size=args.tile_size)
    print(image.shape)

    plt.imshow(image[0].numpy())
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)


