import numpy as np
import torch
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", help="valid task name", default='cifar100')
    parser.add_argument("--data_dir", help="directory to the preprocessed data",
                        default=os.path.join('data', 'cifar100_proc_data'))
    parser.add_argument("--num_prefs_per_task", help="number of preferences per task",
                        default=10)

    args = parser.parse_args()

    if args.task_name == 'cifar10':
        num_tasks = 5
    elif args.task_name == 'celeba':
        num_tasks = 15
    elif args.task_name == 'cifar100':
        num_tasks = 10
    elif args.task_name == 'tinyimagenet':
        num_tasks = 10
    elif args.task_name == '20newsgroup':
        num_tasks = 5
    else:
        raise NotImplementedError

    num_prefs_per_task = int(args.num_prefs_per_task)
    data_dir = args.data_dir

    # Randomly generate prefs
    dict_prefs = {}
    for i in range(num_tasks):
        prefs = []
        if i == 0:
            prefs += [[1]]
        else:
            for j in range(num_prefs_per_task):
                pref = np.random.rand(i + 1)
                pref = pref / np.sum(pref)
                prefs += [pref]
        dict_prefs[i] = prefs

    torch.save(dict_prefs, os.path.join(data_dir, 'dict_prefs.pt'))