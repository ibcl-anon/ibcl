''' Utils for preprocess data for both Split-CIFAR-10 and CelebA'''

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import random
import os
import tqdm
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from utils.dataloader_utils import TinyImageNet


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# Extract features by pretrained ResNet18
def feature_extraction(dataloader):
    resnet = models.resnet18(pretrained=True)
    resnet.eval()
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  # use the last hidden layer of resnet
    features, labels = [], []
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


# Download raw CIFAR 100 data and augment
def download_cifar100(root_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 50000 training data
    trainset = torchvision.datasets.CIFAR100(root=root_dir, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2)
    # 10000 testing data
    testset = torchvision.datasets.CIFAR100(root=root_dir, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    return trainset, testset, trainloader, testloader


# def augment_cifar100(root_dir, dataset, train=True):
#     transform_augment = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(15),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
#     ])
#
#     augmented_data = []
#     augmented_labels = []
#
#     for idx, (image, label) in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
#         # Add the original image to the augmented dataset
#         original_img_array = np.array(image).reshape(3, 32, 32).transpose(1, 2, 0)
#         augmented_data.append(original_img_array)
#         augmented_labels.append(label)
#
#         # Augment and add the image 9 more times to reach a total of 10 versions per original image
#         for _ in range(9):
#             augmented_img = transform_augment(image)
#             img_array = np.array(augmented_img).reshape(3, 32, 32).transpose(1, 2, 0)
#             augmented_data.append(img_array)
#             augmented_labels.append(label)
#
#     # Convert the augmented data and labels into numpy arrays
#     augmented_data_np = np.array(augmented_data).reshape(-1, 3 * 32 * 32)
#     augmented_labels_np = np.array(augmented_labels)
#
#     # Get dataloader
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-100 normalization
#     ])
#     dataset = AugmentedCIFAR100Dataset(augmented_data_np, augmented_labels_np, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)  # Avoid pickling by num_workers=0
#     return dataloader


def save_split_cifar100(data_dir, train_features, train_labels, test_features, test_labels):

    for label in range(100):
        # Train data and labels
        train_indices = np.where(train_labels == label)[0]
        class_train_features = train_features[train_indices]
        class_train_labels = train_labels[train_indices]

        # Test data and labels
        test_indices = np.where(test_labels == label)[0]
        class_test_features = test_features[test_indices]
        class_test_labels = test_labels[test_indices]

        # Save train and test data
        np.savez(os.path.join(data_dir, 'cifar100_train_' + str(label) + '.npz'), features=class_train_features,
                 labels=class_train_labels)
        np.savez(os.path.join(data_dir, 'cifar100_test_' + str(label) + '.npz'), features=class_test_features,
                 labels=class_test_labels)
    return


# Get TinyImageNet data, need manual download from http://cs231n.stanford.edu/tiny-imagenet-200.zip
def get_tinyimagenet(root_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 500 per class - need to split into 450 : 50
    trainset = TinyImageNet(root_dir=os.path.join(root_dir, 'tiny-imagenet-200'), is_train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=False, num_workers=2)

    # 50 per class
    # TinyImageNet's original test data does not have ground-truth labels, so use the val data as test data
    testset = TinyImageNet(root_dir=os.path.join(root_dir, 'tiny-imagenet-200'), is_train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    return trainset, testset, trainloader, testloader


def save_tinyimagenet(data_dir, train_features, train_labels, test_features, test_labels):

    for label in range(200):
        # Train data and labels
        train_indices = np.where(train_labels == label)[0]
        class_train_features = train_features[train_indices]
        class_train_labels = train_labels[train_indices]

        # Test data and labels
        test_indices = np.where(test_labels == label)[0]
        class_test_features = test_features[test_indices]
        class_test_labels = test_labels[test_indices]

        # Save train and test data
        np.savez(os.path.join(data_dir, 'tinyimagenet_train_' + str(label) + '.npz'), features=class_train_features,
                 labels=class_train_labels)
        np.savez(os.path.join(data_dir, 'tinyimagenet_test_' + str(label) + '.npz'), features=class_test_features,
                 labels=class_test_labels)
    return


def download_20newsgroups():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target
    return X_train, y_train, X_test, y_test


def preprocess_20newsgroups(X_train, X_test):
    # TFIDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train).todense()
    X_test_tfidf = vectorizer.transform(X_test).todense()

    # The dimensionality of TFIDF vectors is high, reduce it by PCA
    pca = PCA(n_components=512)
    X_train_reduced = pca.fit_transform(X_train_tfidf)  # 11314 data points
    X_test_reduced = pca.transform(X_test_tfidf)  # 7532 data points

    return X_train_reduced, X_test_reduced


def save_20newsgroups(X_train_reduced, y_train, X_test_reduced, y_test, data_dir):
    torch.save(X_train_reduced, os.path.join(data_dir, 'X_train_reduced.pt'))
    torch.save(y_train, os.path.join(data_dir, 'y_train.pt'))
    torch.save(X_test_reduced, os.path.join(data_dir, 'X_test_reduced.pt'))
    torch.save(y_test, os.path.join(data_dir, 'y_test.pt'))
    return
