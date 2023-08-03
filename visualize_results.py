import numpy as np
import torch
import os
import argparse
import matplotlib.pyplot as plt


# Compute average per task acc
def avg_per_task_acc(dict_all_accs, task_nums=5):
    dict_out = {}

    for i in range(task_nums):
        task_accs = dict_all_accs[i]
        task_pareto_accs = []
        for pref_accs in task_accs:  # for each preference
            pref_pareto_avg_acc = np.mean(pref_accs)
            task_pareto_accs += [pref_pareto_avg_acc]
        dict_out[i] = task_pareto_accs

    return dict_out


# Compute peak per task acc
def peak_per_task_acc(dict_all_accs, task_nums=5):
    dict_out = {}

    for i in range(task_nums):
        task_accs = dict_all_accs[i]
        task_pareto_accs = []
        for pref_accs in task_accs:  # for each preference
            pref_peak_accs = []
            for sampled_model_accs in pref_accs:  # for each sampled model
                peak_model_acc = np.amax(sampled_model_accs)  # max across all tasks so far
                pref_peak_accs += [peak_model_acc]
            pref_pareto_peak_acc = np.amax(pref_peak_accs)
            task_pareto_accs += [pref_pareto_peak_acc]
        dict_out[i] = task_pareto_accs

    return dict_out


# Compute average backward transfer
def avg_bt(dict_all_accs, task_nums=5):
    dict_out = {}

    for i in range(1, task_nums):
        task_accs = dict_all_accs[i]
        task_pareto_bts = []
        for pref_accs in task_accs:  # for each preference
            pref_bts = []
            for sampled_model_accs in pref_accs:  # for each sampled model
                model_bt = 0
                for j in range(1, len(sampled_model_accs)):
                    model_bt += sampled_model_accs[j] - sampled_model_accs[j - 1]
                model_bt = model_bt / (len(sampled_model_accs) - 1)
                pref_bts += [model_bt]
            pref_pareto_bts = np.amax(pref_bts)
            task_pareto_bts += [pref_pareto_bts]
        dict_out[i] = task_pareto_bts

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
    plt.xticks(np.array(task_range) + 1)
    plt.xlabel('Task num')
    plt.ylabel(metric_name)
    plt.title('IBCL Performance')
    plt.legend(loc="lower left")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", help="celeba or cifar10", default='cifar10')
    parser.add_argument("--data_dir", help="directory to the preprocessed data",
                        default=os.path.join('data', 'cifar-10-features'))
    parser.add_argument("--alpha", help="alpha value of IBCL in (0, 1)", default=0.5)
    parser.add_argument("--sublinear", help="0 or 1, 1 means it is a sublinear model", default=0)
    args = parser.parse_args()

    if args.task_name == 'cifar10':
        task_nums = 5
    elif args.task_name == 'celeba':
        task_nums = 40
    else:
        raise NotImplementedError

    # Load dict of accs
    if int(args.sublinear) == 0:
        dict_all_accs = torch.load(os.path.join(args.data_dir, f'dict_all_accs_{args.alpha}.pt'))
    else:
        dict_all_accs = torch.load(os.path.join(args.data_dir, f'dict_all_accs_{args.alpha}_sublinear.pt'))

    # Compute metrics
    dict_avg_accs = avg_per_task_acc(dict_all_accs, task_nums=task_nums)
    dict_peak_accs = peak_per_task_acc(dict_all_accs, task_nums=task_nums)
    dict_avg_bts = avg_bt(dict_all_accs, task_nums=task_nums)

    # Plot metrics
    plot_metrics(dict_avg_accs, task_nums=task_nums, metric_name='Avg per task accuracy', bt=False)
    plot_metrics(dict_peak_accs, task_nums=task_nums, metric_name='Peak per task accuracy', bt=False)
    plot_metrics(dict_avg_bts, task_nums=task_nums, metric_name='Avg per task backward transfer', bt=True)
