import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
import json
from npy_append_array import NpyAppendArray

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
    parser.add_argument("--num_tokens", type=int, default=2,
                        help="number of tokens")
    parser.add_argument("--sequence_length", type=int, default=22,
                        help="sequence length")
    parser.add_argument("--max_n", type=int, default=10,
                        help="max number of occurrences of the first token")
    parser.add_argument("--random_seq_prob", type=float, default=0.,
                        help="probability of using a random sequence")
    parser.add_argument("--num_training_runs", type=int, default=250,
                        help="number of training runs")
    parser.add_argument("--one_hot_input", type=int, default=1,
                        help="whether to use one-hot input")
    parser.add_argument("--rnn_hidden_size", type=int, default=32,
                        help="policy hidden size")
    parser.add_argument("--rnn_num_layers", type=int, default=2,
                        help="policy num layers")
    args = parser.parse_args()
    return args


class FormalLanguageGeneration(nn.Module):
    """
    Abstract class for formal language generation
    :param num_tokens: number of tokens (alphabet size)
    :param sequence_length: length of the sequences
    :param device: device to use
    """
    def __init__(self, num_tokens, sequence_length, device):
        super().__init__()
        self.num_tokens = num_tokens
        self.seq_length = sequence_length
        self.epsilon = 1e-6
        self.bos_token = num_tokens
        self.eos_token = num_tokens + 1
        self.total_tokens = num_tokens + 2
        self.device = device

    def get_sequence(self, batch_size):
        raise NotImplementedError

    def forward(self, batch_size):
        """
        Returns a batch of sequences from the language and the auto-regressive probabilities of the tokens
        """
        sequence = self.get_sequence(batch_size)
        probabilities = self.get_deterministic_probabilities(sequence)
        return sequence, probabilities

    def get_deterministic_probabilities(self, sequence):
        prob_matrix = F.one_hot(sequence[:, 1:], self.total_tokens).float() + self.epsilon
        prob_matrix = prob_matrix / prob_matrix.sum(dim=1, keepdim=True)
        return prob_matrix

    def get_uniform_probabilities(self, allowed_tokens):
        prob_matrix = allowed_tokens.float() + self.epsilon
        prob_matrix = prob_matrix / prob_matrix.sum(dim=1, keepdim=True)
        return prob_matrix

    def accept_sequence(self, sequences):
        raise NotImplementedError

    def get_readme(self):
        raise NotImplementedError


class RandomLanguage(FormalLanguageGeneration):
    """
    A language in which each token is sampled uniformly at random
    """
    def get_sequence(self, batch_size):
        sequence = torch.randint(self.num_tokens, (batch_size, self.seq_length-1), device=self.device).long()
        sequence = torch.cat([torch.ones(batch_size, 1, dtype=torch.long, device=self.device) * self.bos_token,
                              sequence], dim=1)
        return sequence

    def accept_sequence(self, sequences):
        return torch.ones(sequences.shape[0], device=self.device).bool()

    def get_readme(self):
        return "Random language"


class RelativeOccurrenceLanguage(FormalLanguageGeneration):

    def __init__(self, occurrence_differences, sequence_length, max_n=20, device="cpu"):
        super().__init__(len(occurrence_differences), sequence_length, device=device)
        self.occurrence_differences = torch.LongTensor(occurrence_differences).to(self.device)
        self.max_n = max_n
        self.min_n = max(0, -min(occurrence_differences))

    def get_sequence(self, batch_size):
        n = torch.randint(self.min_n, self.max_n, (batch_size,), device=self.device).long()
        num_occurrences = self.occurrence_differences.unsqueeze(0) + n.unsqueeze(1)

        occurrence_thresholds = num_occurrences.cumsum(dim=1)
        occurrence_thresholds = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long, device=self.device),
                                           occurrence_thresholds], dim=1)
        ramp = torch.arange(self.seq_length - 1, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        sequence = (ramp[:, :, None] >= occurrence_thresholds[:, None, :]).long()
        sequence = sequence.sum(dim=-1) - 1

        sequence[sequence == self.num_tokens] = self.eos_token
        # add a bos tokens
        sequence = torch.cat([torch.ones(batch_size, 1, dtype=torch.long, device=self.device) * self.bos_token,
                              sequence], dim=1)
        return sequence

    def accept_sequence(self, sequences):
        # sequence: batch_size x seq_length
        contains_eos = (sequences == self.eos_token).long().sum(dim=1) > 0

        sequences = sequences
        first_eos_token = (sequences == self.eos_token).long().argmax(dim=1)
        # set all tokens after the first eos token to eos token
        sequences[torch.arange(sequences.shape[0], device=self.device), first_eos_token] = self.eos_token
        # get the number of occurrences of each token
        counts = F.one_hot(sequences, self.total_tokens).cumsum(dim=1)[:, :, :self.num_tokens]
        # get the number of occurrences at the first eos token
        counts_at_eos = torch.gather(counts, dim=1,
                                     index=first_eos_token[:, None, None].repeat(1, 1, self.num_tokens)).squeeze(1)
        adjusted_counts = counts_at_eos - self.occurrence_differences[None]
        # sequence is correct if the adjusted counts are the same across the second dimension
        is_correct = (adjusted_counts[:, 0:1] == adjusted_counts).long().sum(dim=1) == self.num_tokens
        is_correct = torch.logical_and(is_correct, contains_eos)
        is_undefined = torch.logical_not(contains_eos)
        return is_correct, is_undefined

    def get_readme(self):
        example_task = self.occurrence_differences.tolist()
        example_strings = []
        tokens = 'abcdefghijklmnopqrstuvwxyz'
        for n in [self.min_n, self.min_n + 1, self.min_n + 2]:
            num_occurrences = (self.occurrence_differences + n).tolist()
            example_strings.append(f"{''.join([tokens[i] * num_occurrences[i] for i in range(len(num_occurrences))])}")

        description_string = f"""
            Example: For the task defined by the occurrence differences {example_task}, the following strings are valid:
            '{example_strings[0]}', '{example_strings[1]}', '{example_strings[2]}'.
        """
        return description_string


def evaluate_rnn(policy, language, num_samples=100, one_hot_input=True, device='cpu',
                 max_sequence_length=100):
    with torch.no_grad():
        policy.eval()
        num_tokens = language.total_tokens

        hidden_states = None
        sequence = [torch.ones(num_samples, 1, dtype=torch.long, device=language.device) * language.bos_token]
        sequence_logits = []
        for j in range(max_sequence_length-1):
            if one_hot_input:
                input = F.one_hot(sequence[-1].to(device), num_tokens).float()
            else:
                input = sequence[-1].to(device).unsqueeze(-1).float()
            output, hidden_states = policy(input, hidden_states)
            logits = F.log_softmax(output, dim=2)
            logits = logits.squeeze(1)
            # get one sample from each distribution
            sample = torch.multinomial(torch.exp(logits), num_samples=1)
            sequence.append(sample)
            sequence_logits.append(logits)
        sequence = torch.cat(sequence, dim=1)
        sequence_logits = torch.stack(sequence_logits, dim=1)

        accept, undefined = language.accept_sequence(sequence)
        num_correct = accept.sum().item()
        num_finished = undefined.shape[0] - undefined.sum().item()
        policy.train()
    return num_correct, sequence.cpu(), sequence_logits.cpu(), num_finished


def main(args):
    if args.seed >= 0:
        seed_everything(args.seed)
    num_training_runs = args.num_training_runs
    snapshot_steps = [0, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000]
    batch_size = 32
    num_tokens = args.num_tokens
    length = args.sequence_length
    one_hot_input = args.one_hot_input > 0
    if args.use_cuda > 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    dataset_name = f"formal_languages"
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

        sequence_storage = NpyAppendArray(f"datasets/{dataset_name}/sequences.npy",
                                          delete_if_exists=True)
        logits_storage = NpyAppendArray(f"datasets/{dataset_name}/logits.npy",
                                        delete_if_exists=True)
        parameter_storage = NpyAppendArray(f"datasets/{dataset_name}/parameters.npy",
                                           delete_if_exists=True)
        num_step_correct_storage = NpyAppendArray(f"datasets/{dataset_name}/num_step_correct.npy",
                                                  delete_if_exists=True)
        task_storage = NpyAppendArray(f"datasets/{dataset_name}/tasks.npy",
                                      delete_if_exists=True)

        args_dict = vars(args)
        with open(f"datasets/{dataset_name}/args.json", 'w') as f:
            json.dump(args_dict, f)

    random_language = RandomLanguage(num_tokens=num_tokens, sequence_length=length, device=device)

    tic = time.time()
    for n in range(num_training_runs):
        print(f"Run {n+1}/{num_training_runs}")

        task = torch.randint(low=-3, high=3, size=(num_tokens - 1,)).tolist()
        task = [0] + task
        print("Occurrence differences:", task)
        language = RelativeOccurrenceLanguage(occurrence_differences=task,
                                              sequence_length=args.sequence_length,
                                              max_n=args.max_n,
                                              device=device)

        rnn = RNNFast(hidden_size=args.rnn_hidden_size,
                      input_size=language.total_tokens if one_hot_input else 1,
                      output_size=language.total_tokens,
                      num_layers=args.rnn_num_layers,
                      input_batch=True)
        rnn = rnn.to(device)
        optimizer = torch.optim.AdamW(rnn.parameters(), lr=0.01, weight_decay=0.0001)
        lr_schedule = PiecewiseLinearSchedule(timesteps=[snapshot_steps[0], snapshot_steps[-2], snapshot_steps[-1]],
                                              values=[0.01, 0.003, 0.0003])

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule(0)

        for step in range(snapshot_steps[-1] + 1):
            sequence, probabilities = language(batch_size)

            if args.random_seq_prob > 0.:
                r_sequence, r_prob_matrix = random_language(batch_size)
                mask = torch.rand(batch_size) < args.random_seq_prob
                sequence[mask] = r_sequence[mask]
                probabilities[mask] = r_prob_matrix[mask]

            sequence = sequence.to(device)
            probabilities = probabilities.to(device)
            if one_hot_input:
                input_sequence = F.one_hot(sequence[:, :-1], language.total_tokens).float()
            else:
                input_sequence = sequence[:, :-1].unsqueeze(-1).float()
            labels = sequence[:, 1:]
            output, _ = rnn(input_sequence)
            loss = F.cross_entropy(output.reshape(-1, output.shape[-1]), labels.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule(step)

            if step in snapshot_steps:
                num_correct, _, _, num_finished = evaluate_rnn(rnn, language,
                                                               1000,
                                                               one_hot_input=one_hot_input,
                                                               device=device,
                                                               max_sequence_length=length)
                print(f'step {step}, num correct: {num_correct} / {num_finished}, loss {loss.item()}')
                _, sequences, logits, _ = evaluate_rnn(rnn, language,
                                                       100,
                                                       one_hot_input=one_hot_input,
                                                       device=device,
                                                       max_sequence_length=length)
                if args.store:
                    sequences = sequences.cpu().to(torch.int16)
                    sequence_storage.append(sequences.unsqueeze(0).detach().cpu().numpy())
                    logits = logits.cpu().to(torch.float16)
                    logits_storage.append(logits.unsqueeze(0).detach().cpu().numpy())
                    parameter_storage.append(torch.cat([p.detach().cpu().flatten()
                                                        for p in rnn.parameters()]).unsqueeze(0).cpu().numpy())

                    num_step_correct = np.array([n, step, num_correct])
                    num_step_correct_storage.append(num_step_correct.reshape(1, -1))

                    task_storage.append(np.array(task).reshape(1, -1))
                    print("stored")
                else:
                    print("not stored")
                pass
        toc = time.time()
        print(f"Time elapsed: {toc - tic} seconds")
        tic = toc


if __name__ == "__main__":
    args = parse_args()
    main(args)