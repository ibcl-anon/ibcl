""" Obtain preprocessed data """

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image


# Dataset and DataLoader
class TaskDataset(Dataset):
    def __init__(self, data, labels):
        self.X = torch.squeeze(torch.from_numpy(data).float())
        self.y = torch.squeeze(torch.from_numpy(labels).float())

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# Test data of TinyImageNet does not have ground-truth labels, so use val data as test
class TinyImageNet(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        if is_train:
            self.data_dir = os.path.join(root_dir, 'train')
            self.image_files = [os.path.join(self.data_dir, class_dir, 'images', image) for class_dir in os.listdir(self.data_dir)
                                for image in os.listdir(os.path.join(self.data_dir, class_dir, 'images'))]
            self.classes = {cls: i for i, cls in enumerate(os.listdir(self.data_dir))}
        else:
            self.data_dir = os.path.join(root_dir, 'val')
            with open(os.path.join(self.data_dir, 'val_annotations.txt'), 'r') as f:
                lines = f.readlines()
            self.image_files = [os.path.join(self.data_dir, 'images', line.split('\t')[0]) for line in lines]
            self.train_data_dir = os.path.join(root_dir, 'train')
            self.classes = {cls: i for i, cls in enumerate(os.listdir(self.train_data_dir))}
            self.image_maps = {line.split('\t')[0]: line.split('\t')[1] for i, line in enumerate(lines)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        if os.name == 'nt':  # if windows os
            token = '\\'
        else:
            token = '/'

        class_name = img_path.split(token)[-3] if 'train' in self.data_dir else self.image_maps[img_path.split(token)[-1]]
        label = self.classes[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label


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


def get_splitcifar10_test_data(data_dir, task_ind=0):
    class_0 = task_ind * 2
    class_1 = task_ind * 2 + 1
    testset_0 = np.load(os.path.join(data_dir, f'cifar10_test_{class_0}.npz'))
    testset_1 = np.load(os.path.join(data_dir, f'cifar10_test_{class_1}.npz'))
    data_test = np.concatenate([testset_0['features'], testset_1['features']], axis=0)
    labels_test = np.concatenate([testset_0['labels'], testset_1['labels']], axis=0)
    labels_test = labels_test - task_ind * 2
    return data_test, labels_test


def get_celeba_loaders(data_dir, task_ind):
    trainset = np.load(os.path.join(data_dir, 'celeba_train_features.npz'))
    valset = np.load(os.path.join(data_dir, 'celeba_val_features.npz'))
    task_train_loader = get_dataloader(trainset['features'], trainset['labels'][:, task_ind], shuffle=True)
    task_val_loader = get_dataloader(valset['features'], valset['labels'][:, task_ind], shuffle=False)
    return task_train_loader, task_val_loader


def get_celeba_test_data(data_dir, task_ind):
    testset = np.load(os.path.join(data_dir, 'celeba_test_features.npz'))
    data_test = np.squeeze(testset['features'])
    labels_test = np.squeeze(testset['labels'][:, task_ind])
    return data_test, labels_test


def splitcifar100_data_split(split_cifar100_dir, task_ind=0, train_size_per_class=400):

    # aquarium fish, beaver, dolphin, flatfish, otter, ray, seal, shark, trout, whale
    animal_classes = [1, 4, 30, 32, 55, 67, 72, 73, 91, 95]
    # bicycle, bus, lawn_mower, motorcycle, pickup_truck, rocket, streetcar, tank, tractor, train
    non_animal_classes = [8, 13, 41, 48, 58, 69, 81, 85, 89, 90]

    class_0 = animal_classes[task_ind]
    class_1 = non_animal_classes[task_ind]

    trainset_0 = np.load(os.path.join(split_cifar100_dir, 'cifar100_train_' + str(class_0) + '.npz'))
    trainset_1 = np.load(os.path.join(split_cifar100_dir, 'cifar100_train_' + str(class_1) + '.npz'))

    trainset_0_size = trainset_0['features'].shape[0]

    # split each class into train: val
    if train_size_per_class > trainset_0_size:
        raise ValueError

    data_train = np.concatenate(
        [trainset_0['features'][0:train_size_per_class], trainset_1['features'][0:train_size_per_class]], axis=0)
    data_val = np.concatenate(
        [trainset_0['features'][train_size_per_class:], trainset_1['features'][train_size_per_class:]], axis=0)
    labels_train = np.concatenate(
        [trainset_0['labels'][0:train_size_per_class], trainset_1['labels'][0:train_size_per_class]], axis=0)

    labels_train_mask = np.isin(labels_train, np.array(non_animal_classes).astype(labels_train.dtype))
    labels_train = labels_train_mask.astype(labels_train.dtype)
    # print(f'Labels train: {labels_train}')

    labels_val = np.concatenate(
        [trainset_0['labels'][train_size_per_class:], trainset_1['labels'][train_size_per_class:]], axis=0)
    labels_val_mask = np.isin(labels_val, np.array(non_animal_classes).astype(labels_val.dtype))
    labels_val = labels_val_mask.astype(labels_val.dtype)
    # print(f'Labels val: {labels_val}')

    return data_train, labels_train, data_val, labels_val


def get_splitcifar100_loaders(data_dir, task_ind):
    data_train, labels_train, data_val, labels_val = splitcifar100_data_split(data_dir, task_ind=task_ind)
    task_train_loader = get_dataloader(data_train, labels_train, batch_size=32, shuffle=True)
    task_val_loader = get_dataloader(data_val, labels_val, batch_size=32, shuffle=False)
    return task_train_loader, task_val_loader


def get_splitcifar100_test_data(data_dir, task_ind=0):

    # aquarium fish, beaver, dolphin, flatfish, otter, ray, seal, shark, trout, whale
    animal_classes = [1, 4, 30, 32, 55, 67, 72, 73, 91, 95]
    # bicycle, bus, lawn_mower, motorcycle, pickup_truck, rocket, streetcar, tank, tractor, train
    non_animal_classes = [8, 13, 41, 48, 58, 69, 81, 85, 89, 90]
    class_0 = animal_classes[task_ind]
    class_1 = non_animal_classes[task_ind]

    testset_0 = np.load(os.path.join(data_dir, f'cifar100_test_{class_0}.npz'))
    testset_1 = np.load(os.path.join(data_dir, f'cifar100_test_{class_1}.npz'))
    data_test = np.concatenate([testset_0['features'], testset_1['features']], axis=0)
    labels_test = np.concatenate([testset_0['labels'], testset_1['labels']], axis=0)
    labels_test_mask = np.isin(labels_test, np.array(non_animal_classes).astype(labels_test.dtype))
    labels_test = labels_test_mask.astype(labels_test.dtype)
    return data_test, labels_test


def tinyimagenet_split(data_dir, task_ind=0, train_size_per_class=450):

    animal_classes = [
        0,  # 'n01443537', goldfish, Carassius auratus
        1,  # 'n01629819', European fire salamander, Salamandra salamandra
        2,  # 'n01641577', bullfrog, Rana catesbeiana
        3,  # 'n01644900', tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui
        4,  # 'n01698640', American alligator, Alligator mississipiensis
        5,  # 'n01742172', boa constrictor, Constrictor constrictor
        11,  # 'n01855672', goose
        12,  # 'n01882714', koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus
        21,  # 'n02056570', king penguin, Aptenodytes patagonica
        22  # 'n02058221', albatross, mollymawk
    ]
    non_animal_classes = [
        195,  # 'n09246464', cliff, drop, drop-off
        193,  # 'n07920052', espresso
        192,  # 'n07875152', potpie
        191,  # 'n07873807', pizza, pizza pie
        190,  # 'n07871810', meatloaf, meat loaf
        188,  # 'n07753592', banana
        186,  # 'n07747607', orange
        173,  # 'n04562935', water tower
        170,  # 'n04532670', via duct
        164  # 'n04465501', tractor
    ]

    class_0 = animal_classes[task_ind]
    class_1 = non_animal_classes[task_ind]

    trainset_0 = np.load(os.path.join(data_dir, 'tinyimagenet_train_' + str(class_0) + '.npz'))
    trainset_1 = np.load(os.path.join(data_dir, 'tinyimagenet_train_' + str(class_1) + '.npz'))

    trainset_0_size = trainset_0['features'].shape[0]
    if train_size_per_class > trainset_0_size:
        raise ValueError

    data_train = np.concatenate(
        [trainset_0['features'][0:train_size_per_class], trainset_1['features'][0:train_size_per_class]], axis=0)
    data_val = np.concatenate(
        [trainset_0['features'][train_size_per_class:], trainset_1['features'][train_size_per_class:]], axis=0)
    labels_train = np.concatenate(
        [trainset_0['labels'][0:train_size_per_class], trainset_1['labels'][0:train_size_per_class]], axis=0)

    labels_train_mask = np.isin(labels_train, np.array(non_animal_classes).astype(labels_train.dtype))
    labels_train = labels_train_mask.astype(labels_train.dtype)
    # print(f'Labels train: {labels_train}')

    labels_val = np.concatenate(
        [trainset_0['labels'][train_size_per_class:], trainset_1['labels'][train_size_per_class:]], axis=0)
    labels_val_mask = np.isin(labels_val, np.array(non_animal_classes).astype(labels_val.dtype))
    labels_val = labels_val_mask.astype(labels_val.dtype)
    # print(f'Labels val: {labels_val}')

    return data_train, labels_train, data_val, labels_val


def get_tinyimagenet_loaders(data_dir, task_ind=0):
    data_train, labels_train, data_val, labels_val = tinyimagenet_split(data_dir, task_ind=task_ind)
    task_train_loader = get_dataloader(data_train, labels_train, batch_size=32, shuffle=True)
    task_val_loader = get_dataloader(data_val, labels_val, batch_size=32, shuffle=False)
    return task_train_loader, task_val_loader


def get_tinyimagenet_test_data(data_dir, task_ind=0):

    animal_classes = [
        0,  # 'n01443537', goldfish, Carassius auratus
        1,  # 'n01629819', European fire salamander, Salamandra salamandra
        2,  # 'n01641577', bullfrog, Rana catesbeiana
        3,  # 'n01644900', tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui
        4,  # 'n01698640', American alligator, Alligator mississipiensis
        5,  # 'n01742172', boa constrictor, Constrictor constrictor
        11,  # 'n01855672', goose
        12,  # 'n01882714', koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus
        21,  # 'n02056570', king penguin, Aptenodytes patagonica
        22  # 'n02058221', albatross, mollymawk
    ]
    non_animal_classes = [
        195,  # 'n09246464', cliff, drop, drop-off
        193,  # 'n07920052', espresso
        192,  # 'n07875152', potpie
        191,  # 'n07873807', pizza, pizza pie
        190,  # 'n07871810', meatloaf, meat loaf
        188,  # 'n07753592', banana
        186,  # 'n07747607', orange
        173,  # 'n04562935', water tower
        170,  # 'n04532670', via duct
        164  # 'n04465501', tractor
    ]

    class_0 = animal_classes[task_ind]
    class_1 = non_animal_classes[task_ind]

    testset_0 = np.load(os.path.join(data_dir, f'tinyimagenet_test_{class_0}.npz'))
    testset_1 = np.load(os.path.join(data_dir, f'tinyimagenet_test_{class_1}.npz'))
    data_test = np.concatenate([testset_0['features'], testset_1['features']], axis=0)
    labels_test = np.concatenate([testset_0['labels'], testset_1['labels']], axis=0)
    labels_test_mask = np.isin(labels_test, np.array(non_animal_classes).astype(labels_test.dtype))
    labels_test = labels_test_mask.astype(labels_test.dtype)
    return data_test, labels_test


def get_20newsgroup_loaders(data_dir, task_ind=0, val_data_size=50):
    X_train = torch.load(os.path.join(data_dir, 'X_train_reduced.pt'))
    y_train = torch.load(os.path.join(data_dir, 'y_train.pt'))

    computer_classes = [1, 2, 3, 4, 5]
    non_computer_classes = [6, 7, 8, 9, 10]
    class_0 = computer_classes[task_ind]
    class_1 = non_computer_classes[task_ind]

    X_task_0 = X_train[y_train == class_0]
    X_task_1 = X_train[y_train == class_1]
    n_0, n_1 = len(X_task_0), len(X_task_1)
    X_train_task = np.concatenate([X_task_0[: n_0 - val_data_size], X_task_1[: n_1 - val_data_size]])
    X_val_task = np.concatenate([X_task_0[n_0 - val_data_size:], X_task_1[n_1 - val_data_size:]])
    y_train_task = np.array([0] * (n_0 - val_data_size) + [1] * (n_1 - val_data_size))
    y_val_task = np.array([0] * val_data_size + [1] * val_data_size)

    train_loader = get_dataloader(X_train_task, y_train_task, batch_size=32, shuffle=True)
    val_loader = get_dataloader(X_val_task, y_val_task, batch_size=32, shuffle=False)

    return train_loader, val_loader


def get_20newsgroup_test_data(data_dir, task_ind=0):
    X_test = torch.load(os.path.join(data_dir, 'X_test_reduced.pt'))
    y_test = torch.load(os.path.join(data_dir, 'y_test.pt'))

    computer_classes = [1, 2, 3, 4, 5]
    non_computer_classes = [6, 7, 8, 9, 10]
    class_0 = computer_classes[task_ind]
    class_1 = non_computer_classes[task_ind]

    X_task_0 = X_test[y_test == class_0]
    X_task_1 = X_test[y_test == class_1]
    X_test_task = np.concatenate([X_task_0, X_task_1])
    y_test_task = np.array([0] * len(X_task_0) + [1] * len(X_task_1))

    return X_test_task, y_test_task
