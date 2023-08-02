""" Obtain preprocessed data """

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# Dataset and DataLoader
class TaskDataset(Dataset):
    def __init__(self, data, labels):
        self.X = torch.squeeze(torch.from_numpy(data).float())
        self.y = torch.squeeze(torch.from_numpy(labels).float())

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def get_dataloader(data, labels, batch_size=64, shuffle=True, drop_last=True):
    dataset = TaskDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def splitcifar10_data_split(split_cifar10_dir, task_ind=0, train_size_per_class=4000):
    class_0 = task_ind * 2
    class_1 = task_ind * 2 + 1

    trainset_0 = np.load(os.path.join(split_cifar10_dir, 'cifar10_train_' + str(class_0) + '.npz'))
    trainset_1 = np.load(os.path.join(split_cifar10_dir, 'cifar10_train_' + str(class_1) + '.npz'))

    # split each class into train_l : train_u : val
    if train_size_per_class > trainset_0['features'].shape[0]:
        raise ValueError

    data_train = np.concatenate(
        [trainset_0['features'][0:train_size_per_class], trainset_1['features'][0:train_size_per_class]], axis=0)
    data_val = np.concatenate(
        [trainset_0['features'][train_size_per_class:], trainset_1['features'][train_size_per_class:]], axis=0)
    labels_train = np.concatenate(
        [trainset_0['labels'][0:train_size_per_class], trainset_1['labels'][0:train_size_per_class]], axis=0)
    labels_train = labels_train - task_ind * 2
    labels_val = np.concatenate(
        [trainset_0['labels'][train_size_per_class:], trainset_1['labels'][train_size_per_class:]], axis=0)
    labels_val = labels_val - task_ind * 2
    return data_train, labels_train, data_val, labels_val


def get_splitcifar10_loaders(data_dir, task_ind):
    data_train, labels_train, data_val, labels_val = splitcifar10_data_split(data_dir, task_ind=task_ind)
    task_train_loader = get_dataloader(data_train, labels_train, shuffle=True)
    task_val_loader = get_dataloader(data_val, labels_val, shuffle=False)
    return task_train_loader, task_val_loader


def get_celeba_loaders(data_dir, task_ind):
    trainset = np.load(os.path.join(data_dir, 'celeba_train_features.npz'))
    valset = np.load(os.path.join(data_dir, 'celeba_val_features.npz'))
    task_train_loader = get_dataloader(trainset['features'], trainset['labels'][:, task_ind], shuffle=True)
    task_val_loader = get_dataloader(valset['features'], valset['labels'][:, task_ind], shuffle=False)
    return task_train_loader, task_val_loader


def get_splitcifar10_test_data(data_dir, task_ind=0):
    class_0 = task_ind * 2
    class_1 = task_ind * 2 + 1
    testset_0 = np.load(os.path.join(data_dir, f'cifar10_test_{class_0}.npz'))
    testset_1 = np.load(os.path.join(data_dir, f'cifar10_test_{class_1}.npz'))
    data_test = np.concatenate([testset_0['features'], testset_1['features']], axis=0)
    labels_test = np.concatenate([testset_0['labels'], testset_1['labels']], axis=0)
    labels_test = labels_test - task_ind * 2
    return data_test, labels_test


def get_celeba_test_data(data_dir, task_ind):
    testset = np.load(os.path.join(data_dir, 'celeba_test_features.npz'))
    data_test = np.squeeze(testset['features'])
    labels_test = np.squeeze(testset['labels'][:, task_ind])
    return data_test, labels_test
