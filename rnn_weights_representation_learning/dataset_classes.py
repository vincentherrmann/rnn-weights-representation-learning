import torch
import numpy as np
import os
from utilities import fix_seed


def task_entropy_criterion(tasks, inverse=False):
    tasks = 1 + tasks - tasks.min(axis=-1, keepdims=True)
    tasks_as_distribution = tasks.astype(np.float32) / tasks.sum(axis=-1, keepdims=True)
    entropy = -(tasks_as_distribution * np.log(tasks_as_distribution + 1e-6)).sum(axis=-1)
    if inverse:
        return 1 / entropy
    return entropy


def task_random_criterion(tasks):
    return np.random.rand(tasks.shape[0])


class ParametersSequencesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, sequences_per_rnn, eval_ratio=0.1, ood_ratio=0.1, mode='train',
                 task_criterion=lambda x: x.reshape(x.shape[0], -1).sum(-1), downstream_training_on_ood=False):
        super().__init__()
        # join dataset path with rnn dataset name
        self.parameters = np.load(os.path.join(dataset_path, "parameters.npy"), mmap_mode='r')
        self.sequences = np.load(os.path.join(dataset_path, "sequences.npy"), mmap_mode='r')
        self.outputs = np.load(os.path.join(dataset_path, "logits.npy"), mmap_mode='r')
        self.tasks = np.load(os.path.join(dataset_path, "tasks.npy"), mmap_mode='r')

        # find unique task vectors
        self.unique_tasks, self.unique_tasks_num = np.unique(self.tasks,
                                                             return_counts=True, axis=0)
        task_criteria = task_criterion(self.unique_tasks)
        sorted_tasks = self.unique_tasks[np.argsort(task_criteria)]

        num_train_tasks = int(self.unique_tasks.shape[0] * (1 - eval_ratio - ood_ratio))
        num_eval_tasks = int(self.unique_tasks.shape[0] * eval_ratio)
        num_ood_tasks = self.unique_tasks.shape[0] - num_train_tasks - num_eval_tasks

        self.ood_tasks = sorted_tasks[-num_ood_tasks:]
        get_random_idxs = fix_seed(lambda x: np.random.permutation(x))
        rand_idxs = get_random_idxs(num_train_tasks + num_eval_tasks, seed=1)
        self.train_tasks = sorted_tasks[rand_idxs[:num_train_tasks]]
        self.eval_tasks = sorted_tasks[rand_idxs[num_train_tasks:]]

        if downstream_training_on_ood:
            downstream_rand_idxs = get_random_idxs(num_ood_tasks, seed=2)
            new_num_ood = num_ood_tasks // 2
            self.train_tasks = np.concatenate([self.train_tasks,
                                               self.ood_tasks[downstream_rand_idxs[new_num_ood:]]])
            self.ood_tasks = self.ood_tasks[downstream_rand_idxs[:new_num_ood]]

        if mode == 'train':
            dataset_tasks = self.train_tasks
        elif mode == 'eval':
            dataset_tasks = self.eval_tasks
        elif mode == 'ood':
            dataset_tasks = self.ood_tasks

        task_mask = (self.tasks[:, None] == dataset_tasks[None, :]).all(-1).any(-1)
        self.allowed_idxs = np.arange(self.parameters.shape[0])[task_mask]

        assert sequences_per_rnn <= self.sequences.shape[1]
        self.sequences_per_rnn = sequences_per_rnn

    def __getitem__(self, idx):
        lookup_idx = self.allowed_idxs[idx]
        parameters = self.parameters[lookup_idx]
        sequences = self.sequences[lookup_idx]
        logits = self.outputs[lookup_idx]
        task = self.tasks[lookup_idx]

        # sample transitions
        transitions_idxs = np.random.choice(sequences.shape[0], self.sequences_per_rnn, replace=False)
        sampled_sequences = sequences[transitions_idxs]
        sampled_logits = logits[transitions_idxs]

        return_dict = {
            'parameters': torch.from_numpy(parameters),
            'sequences': torch.from_numpy(sampled_sequences),
            'logits': torch.from_numpy(sampled_logits),
            'task': torch.from_numpy(task)
        }

        return return_dict

    def __len__(self):
        return len(self.allowed_idxs)


class ParametersFormalLanguageDataset(ParametersSequencesDataset):
    def __init__(self, dataset_path, sequences_per_rnn, eval_ratio=0.1, ood_ratio=0.1, mode='train', min_return=0,
                 task_criterion=lambda x: x.reshape(x.shape[0], -1).sum(-1), downstream_training_on_ood=False):
        super().__init__(dataset_path=dataset_path,
                         sequences_per_rnn=sequences_per_rnn,
                         eval_ratio=eval_ratio,
                         ood_ratio=ood_ratio,
                         mode=mode,
                         task_criterion=task_criterion,
                         downstream_training_on_ood=downstream_training_on_ood)

        num_step_correct = np.load(os.path.join(dataset_path, "num_step_correct.npy"), mmap_mode='r')
        self.rnn_idx = num_step_correct[:, 0]
        self.rnn_step = num_step_correct[:, 1]
        self.rnn_returns = num_step_correct[:, 2] / 1000

        if min_return > 0:
            return_mask = self.rnn_returns[self.allowed_idxs] >= min_return
            self.allowed_idxs = self.allowed_idxs[return_mask]

    def __getitem__(self, idx):
        return_dict = super().__getitem__(idx)

        lookup_idx = self.allowed_idxs[idx]
        returns = self.rnn_returns[lookup_idx]
        step = self.rnn_step[lookup_idx]

        return_dict['parameters'] = return_dict['parameters'].float()
        return_dict['sequences'] = return_dict['sequences'].long()
        return_dict['logits'] = return_dict['logits'].float()
        return_dict['task'] = return_dict['task'].long()
        return_dict['returns'] = torch.FloatTensor([returns])
        return_dict['step'] = torch.LongTensor([step])

        return return_dict


class ParametersSequentialMNISTDataset(ParametersSequencesDataset):
    def __init__(self, dataset_path, sequences_per_rnn, eval_ratio=0.1, ood_ratio=0.1, mode='train', min_return=0,
                 task_criterion=lambda x: x.reshape(x.shape[0], -1).sum(-1), downstream_training_on_ood=False):
        super().__init__(dataset_path=dataset_path,
                         sequences_per_rnn=sequences_per_rnn,
                         eval_ratio=eval_ratio,
                         ood_ratio=ood_ratio,
                         mode=mode,
                         task_criterion=task_criterion,
                         downstream_training_on_ood=downstream_training_on_ood)

        num_step_correct = np.load(os.path.join(dataset_path, "num_step.npy"), mmap_mode='r')
        self.rnn_idx = num_step_correct[:, 0]
        self.rnn_step = num_step_correct[:, 1]

        accuracy_losses = np.load(os.path.join(dataset_path, "accuracy_losses.npy"), mmap_mode='r')
        self.eval_accuracy = accuracy_losses[:, 0]
        self.train_loss = accuracy_losses[:, 1]
        self.eval_loss = accuracy_losses[:, 2]
        self.weight_decay = accuracy_losses[:, 3]

        if min_return > 0:
            return_mask = self.eval_accuracy[self.allowed_idxs] >= min_return
            self.allowed_idxs = self.allowed_idxs[return_mask]

    def __getitem__(self, idx):
        return_dict = super().__getitem__(idx)

        lookup_idx = self.allowed_idxs[idx]
        returns = self.eval_accuracy[lookup_idx]
        eval_losses = self.eval_loss[lookup_idx]
        generalization_gaps = eval_losses - self.train_loss[lookup_idx]
        weight_decays = self.weight_decay[lookup_idx]
        step = self.rnn_step[lookup_idx]

        return_dict['parameters'] = return_dict['parameters'].float()
        return_dict['sequences'] = return_dict['sequences'].float()
        return_dict['logits'] = return_dict['logits'].float()
        return_dict['task'] = return_dict['task'].float()
        return_dict['returns'] = torch.FloatTensor([returns])
        return_dict['eval_losses'] = torch.FloatTensor([eval_losses])
        return_dict['generalization_gaps'] = torch.FloatTensor([generalization_gaps])
        return_dict['weight_decays'] = torch.FloatTensor([weight_decays])
        return_dict['step'] = torch.LongTensor([step])

        return return_dict