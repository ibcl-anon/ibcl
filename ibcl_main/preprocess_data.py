''' Preprocess data for both Split-CIFAR-10 and CelebA'''

import argparse
import os
from utils.preprocess_utils import feature_extraction, download_celeba, download_cifar10, save_celeba, save_split_cifar10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", help="celeba or cifar10", default='cifar10')
    parser.add_argument("--raw_data_dir", help="directory to the raw data", default=os.path.join('data', 'cifar-10'))
    parser.add_argument("--proc_data_dir", help="directory to the preprocessed data", default=os.path.join('data', 'cifar-10-features'))
    args = parser.parse_args()

    if not os.path.exists(args.raw_data_dir):
        os.mkdir(args.raw_data_dir)
    if not os.path.exists(args.proc_data_dir):
        os.mkdir(args.proc_data_dir)

    if args.task_name == 'cifar10':
        trainloader, testloader = download_cifar10(args.raw_data_dir)
        train_features, train_labels = feature_extraction(trainloader)
        test_features, test_labels = feature_extraction(testloader)
        save_split_cifar10(args.proc_data_dir, train_features, train_labels, test_features, test_labels)
    elif args.task_name == 'celeba':
        trainloader, valloader, testloader = download_celeba(args.raw_data_dir)
        train_features, train_labels = feature_extraction(trainloader)
        val_features, val_labels = feature_extraction(valloader)
        test_features, test_labels = feature_extraction(testloader)
        save_celeba(args.proc_data_dir, train_features, train_labels, val_features, val_labels, test_features, test_labels)
    else:
        raise NotImplementedError
