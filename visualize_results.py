import numpy as np
import torch
import os
import argparse
import matplotlib.pyplot as plt


# Generate lower triangular matrix of accs to compute CL metrics
# Each task has M prefs, each pref has K models, each model evaluates testing acc on i tasks encountered so far
# If model is probabilistic, generate 3 matrices - max, mean, min; else, generate 1 matrix
def generate_acc_matrix(dict_all_accs, dict_prefs, task_nums=5, deterministic=False):
    dict_matrix = {}
    for i in range(task_nums):
        task_prefs = dict_prefs[i]
        prefs_sum = np.sum(task_prefs, axis=0)
        task_accs = dict_all_accs[i]
        pref_weighted_max_accs = np.zeros(shape=(i+1))
        pref_weighted_mean_accs = np.zeros(shape=(i+1))
        pref_weighted_min_accs = np.zeros(shape=(i+1))
        pref_weighted_accs = np.zeros(shape=(i+1))
        for j in range(len(task_accs)):
            pref = np.array(task_prefs[j])  # shape = (i)
            pref_weights = []
            for l in range(i+1):
                if prefs_sum[l] == 0:
                    pref_weights += [1 / i]
                else:
                    pref_weights += [pref[l] / prefs_sum[l]]
            pref_weights = np.array(pref_weights)
            pref_accs = task_accs[j]
            pref_accs = np.array(pref_accs)
            if not deterministic:  # VCL, IBCL - probabilistic models
                # For each pref, get the max, mean, min performance of the sampled models
                pref_max_accs = np.max(pref_accs, axis=0)  # shape = (i)
                pref_mean_accs = np.mean(pref_accs, axis=0)
                pref_min_accs = np.min(pref_accs, axis=0)
                # Compute weighted sum - performance in addressing each pref
                pref_weighted_max_accs += np.multiply(pref_max_accs, pref_weights)
                pref_weighted_mean_accs += np.multiply(pref_mean_accs, pref_weights)
                pref_weighted_min_accs += np.multiply(pref_min_accs, pref_weights)
            else:
                pref_weighted_accs += np.multiply(pref_accs, pref_weights)
        if not deterministic:
            dict_matrix[i] = [pref_weighted_max_accs, pref_weighted_mean_accs, pref_weighted_min_accs]
        else:
            dict_matrix[i] = [pref_weighted_accs]
    return dict_matrix


# Compute average per task acc
def avg_per_task_acc(dict_matrix, task_nums=5, deterministic=False):
    dict_out = {}
    for i in range(task_nums):
        if not deterministic:
            pref_weighted_max_accs, pref_weighted_mean_accs, pref_weighted_min_accs = dict_matrix[i]
            avg_max_acc = np.mean(pref_weighted_max_accs)
            avg_mean_acc = np.mean(pref_weighted_mean_accs)
            avg_min_acc = np.mean(pref_weighted_min_accs)
            dict_out[i] = [avg_max_acc, avg_mean_acc, avg_min_acc]
        else:
            pref_weighted_accs = dict_matrix[i]
            avg_acc = np.mean(pref_weighted_accs)
            dict_out[i] = [avg_acc]
    return dict_out


# Compute peak per task acc
def peak_per_task_acc(dict_matrix, task_nums=5, deterministic=False):
    dict_out = {}
    for i in range(task_nums):
        if not deterministic:
            pref_weighted_max_accs, pref_weighted_mean_accs, pref_weighted_min_accs = dict_matrix[i]
            peak_max_acc = np.max(pref_weighted_max_accs)
            peak_mean_acc = np.max(pref_weighted_mean_accs)
            peak_min_acc = np.max(pref_weighted_min_accs)
            dict_out[i] = [peak_max_acc, peak_mean_acc, peak_min_acc]
        else:
            pref_weighted_accs = dict_matrix[i]
            peak_acc = np.max(pref_weighted_accs)
            dict_out[i] = [peak_acc]
    return dict_out


def compute_bt(accs):
    all_diffs = []
    for i in range(1, len(accs)):
        acc_diff = accs[i] - accs[i-1]
        all_diffs += [acc_diff]
    return np.mean(all_diffs)


# Compute average backward transfer
def avg_bt(dict_matrix, task_nums=5, deterministic=False):
    dict_out = {}
    for i in range(1, task_nums):
        if not deterministic:
            pref_weighted_max_accs, pref_weighted_mean_accs, pref_weighted_min_accs = dict_matrix[i]
            bt_max_acc = compute_bt(pref_weighted_max_accs)
            bt_mean_acc = compute_bt(pref_weighted_mean_accs)
            bt_min_acc = compute_bt(pref_weighted_min_accs)
            dict_out[i] = [bt_max_acc, bt_mean_acc, bt_min_acc]
        else:
            pref_weighted_accs = dict_matrix[i]
            bt_acc = compute_bt(pref_weighted_accs)
            dict_out[i] = [bt_acc]
    return dict_out


# Plot metrics
def plot_metrics(dict_metric, task_nums=5, metric_name='Avg per task accuracy', bt=False):
    if bt:
        task_range = list(range(1, task_nums))
    else:
        task_range = list(range(task_nums))
    task_range = [int(j) for j in task_range]
    plot_data = [dict_metric[j] for j in task_range]

    if bt:
        max_data = [np.amax(plot_data[j]) for j in range(task_nums-1)]
        mean_data = [np.mean(plot_data[j]) for j in range(task_nums-1)]
        min_data = [np.amin(plot_data[j]) for j in range(task_nums-1)]
    else:
        max_data = [np.amax(plot_data[j]) for j in range(task_nums)]
        mean_data = [np.mean(plot_data[j]) for j in range(task_nums)]
        min_data = [np.amin(plot_data[j]) for j in range(task_nums)]

    plt.plot(np.array(task_range) + 1, max_data, '-*', label=f'IBCL Pareto', color='blue', markersize=12)
    plt.plot(np.array(task_range) + 1, mean_data, '--', label=f'IBCL mean', color='blue')
    plt.fill_between(np.array(task_range) + 1, max_data, min_data, facecolor='tab:blue', alpha=0.5, label=f'IBCL range')

    if bt:
        plt.ylim([-0.5, 0.5])
    else:
        plt.ylim([0, 1])
    if task_nums <= 10:
        plt.xticks(np.array(task_range) + 1)
    else:
        plt.xticks(np.array(task_range[::5]) + 1)
    plt.xlabel('Task num')
    plt.ylabel(metric_name)
    plt.title('IBCL Performance')
    plt.legend(loc="lower left")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", help="celeba or cifar10 or rmnist", default='cifar10')
    parser.add_argument("--data_dir", help="directory to the preprocessed data",
                        default=os.path.join('data', 'cifar-10-features'))
    parser.add_argument("--alpha", help="alpha value of IBCL in (0, 1)", default=0.5)
    parser.add_argument("--discard_threshold", help="d threshold of sublinear fgcs", default=0.01)
    args = parser.parse_args()

    if args.task_name == 'cifar10':
        task_nums = 5
    elif args.task_name == 'celeba':
        task_nums = 15
    elif args.task_name == 'cifar100':
        task_nums = 10
    elif args.task_name == 'tinyimagenet':
        task_nums = 10
    elif args.task_name == '20newsgroup':
        task_nums = 5
    else:
        raise NotImplementedError

    # Load dict of accs
    if int(args.sublinear) == 0:
        dict_all_accs = torch.load(os.path.join(args.data_dir, f'dict_all_accs_{args.alpha}.pt'))
    else:
        dict_all_accs = torch.load(os.path.join(args.data_dir, f'dict_all_accs_{args.alpha}_{args.discard_threshold}.pt'))

    # Load dict of prefs
    dict_prefs = torch.load(os.path.join(args.data_dir, 'dict_prefs.pt'))

    # Compute acc matrix
    dict_matrix = generate_acc_matrix(dict_all_accs, dict_prefs, task_nums=task_nums, deterministic=False)

    # Compute metrics
    dict_avg_accs = avg_per_task_acc(dict_matrix, task_nums=task_nums)
    dict_peak_accs = peak_per_task_acc(dict_matrix, task_nums=task_nums)
    dict_avg_bts = avg_bt(dict_matrix, task_nums=task_nums)

    # Plot metrics
    plot_metrics(dict_avg_accs, task_nums=task_nums, metric_name='Avg per task accuracy', bt=False)
    plot_metrics(dict_peak_accs, task_nums=task_nums, metric_name='Peak per task accuracy', bt=False)
    plot_metrics(dict_avg_bts, task_nums=task_nums, metric_name='Avg per task backward transfer', bt=True)
