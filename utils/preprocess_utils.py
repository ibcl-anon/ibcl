''' Utils for preprocess data for both Split-CIFAR-10 and CelebA'''

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os


# Extract features by pretrained ResNet18
def feature_extraction(dataloader):
    resnet = models.resnet18(pretrained=True)
    resnet.eval()
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  # use the last hidden layer of resnet
    features, labels= [], []
    with torch.no_grad():
        for data in dataloader:
            images, labels_batch = data
            outputs = torch.squeeze(feature_extractor(images))
            features.append(outputs.numpy())
            labels.append(labels_batch.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


# Download raw CIFAR 10 data
def download_cifar10(root_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 50000 training data
    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2)

    # 10000 testing data
    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    return trainloader, testloader


# Dwonload raw CelebA data
# If data is too large to be downloaded, plese refer to https://github.com/pytorch/vision/issues/2262
def download_celeba(root_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CelebA(root=root_dir, split='train', target_type='attr', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2)
    valset = torchvision.datasets.CelebA(root=root_dir, split='valid', target_type='attr', download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
    testset = torchvision.datasets.CelebA(root=root_dir, split='test', target_type='attr', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    return trainloader, valloader, testloader


# Split CIFAR10 by 10 classes and save
def save_split_cifar10(data_dir, train_features, train_labels, test_features, test_labels):

    for label in range(10):
        # Train data and labels
        train_indices = np.where(train_labels == label)[0]
        class_train_features = train_features[train_indices]
        class_train_labels = train_labels[train_indices]

        # Test data and labels
        test_indices = np.where(test_labels == label)[0]
        class_test_features = test_features[test_indices]
        class_test_labels = test_labels[test_indices]

        # Save train and test data
        np.savez(os.path.join(data_dir, 'cifar10_train_' + str(label) + '.npz'), features=class_train_features,
                 labels=class_train_labels)
        np.savez(os.path.join(data_dir, 'cifar10_test_' + str(label) + '.npz'), features=class_test_features,
                 labels=class_test_labels)
    return


# Save CelebA data
def save_celeba(data_dir, train_features, train_labels, val_features, val_labels, test_features, test_labels):
    np.savez(os.path.join(data_dir, 'celeba_train_features.npz'), features=train_features, labels=train_labels)
    np.savez(os.path.join(data_dir, 'celeba_val_features.npz'), features=val_features, labels=val_labels)
    np.savez(os.path.join(data_dir, 'celeba_test_features.npz'), features=test_features, labels=test_labels)
