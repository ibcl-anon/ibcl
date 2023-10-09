''' Preprocess data for both Split-CIFAR-10 and CelebA'''

import argparse
from utils.preprocess_utils import *
from utils.dataloader_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", help="valid task name", default='cifar100')
    parser.add_argument("--raw_data_dir", help="directory to the raw data", default=os.path.join('data', 'cifar-10'))
    parser.add_argument("--proc_data_dir", help="directory to the preprocessed data", default=os.path.join('data', 'cifar100_proc_data'))
    args = parser.parse_args()

    set_seed(42)

    if not os.path.exists(args.raw_data_dir):
        os.mkdir(args.raw_data_dir)
    if not os.path.exists(args.proc_data_dir):
        os.mkdir(args.proc_data_dir)

    if args.task_name == 'cifar10':
        trainloader, testloader = download_cifar10(args.raw_data_dir)
        print('Preprocessing CIFAR-10 data ...')
        train_features, train_labels = feature_extraction(trainloader)
        test_features, test_labels = feature_extraction(testloader)
        save_split_cifar10(args.proc_data_dir, train_features, train_labels, test_features, test_labels)
    elif args.task_name == 'celeba':
        trainloader, valloader, testloader = download_celeba(args.raw_data_dir)
        print('Preprocessing CelebA data ...')
        train_features, train_labels = feature_extraction(trainloader)
        val_features, val_labels = feature_extraction(valloader)
        test_features, test_labels = feature_extraction(testloader)
        save_celeba(args.proc_data_dir, train_features, train_labels, val_features, val_labels, test_features, test_labels)
    elif args.task_name == 'cifar100':
        trainset, testset, trainloader, testloader = download_cifar100(args.raw_data_dir)
        print('Preprocessing CIFAR100 data ...')
        train_features, train_labels = feature_extraction(trainloader)
        test_features, test_labels = feature_extraction(testloader)
        save_split_cifar100(args.proc_data_dir, train_features, train_labels, test_features, test_labels)
    elif args.task_name == 'tinyimagenet':
        trainset, testset, trainloader, testloader = get_tinyimagenet(args.raw_data_dir)
        print('Preprocessing TinyImageNet data ...')
        train_features, train_labels = feature_extraction(trainloader)
        test_features, test_labels = feature_extraction(testloader)
        save_tinyimagenet(args.proc_data_dir, train_features, train_labels, test_features, test_labels)
    elif args.task_name == '20newsgroup':
        X_train, y_train, X_test, y_test = download_20newsgroups()
        print('Preprocessing 20newsgroups data ...')
        X_train_reduced, X_test_reduced = preprocess_20newsgroups(X_train, X_test)
        save_20newsgroups(X_train_reduced, y_train, X_test_reduced, y_test, args.proc_data_dir)
    else:
        raise NotImplementedError
