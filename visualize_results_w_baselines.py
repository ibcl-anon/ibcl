import numpy as np
import torch
import os
import argparse
import matplotlib.pyplot as plt
from visualize_results import avg_per_task_acc, peak_per_task_acc, avg_bt


# Plot metrics of GEM + VCL + IBCL
def plot_metrics_w_baselines(dict_gem, dict_vcl, dict_ibcl, task_nums=5, metric_name='Avg per task accuracy', bt=False):

    dicts = [dict_gem, dict_vcl, dict_ibcl]
    curve_colors = ['orange', 'green', 'blue']
    fill_colors = ['gold', 'limegreen', 'tab:blue']

    if bt:
        task_range = list(range(1, task_nums))
    else:
        task_range = list(range(task_nums))
    task_range = [int(j) for j in task_range]

    plot_data_gem = [dicts[0][j] for j in range(task_nums)]
    plot_data_vcl = [dicts[1][j] for j in range(task_nums)]
    plot_data_ibcl = [dicts[2][j] for j in range(task_nums)]

    if bt:
        mean_data_gem = [np.mean(plot_data_gem[j]) for j in range(task_nums-1)]
        max_data_vcl = [np.amax(plot_data_vcl[j]) for j in range(task_nums-1)]
        mean_data_vcl = [np.mean(plot_data_vcl[j]) for j in range(task_nums-1)]
        min_data_vcl = [np.amin(plot_data_vcl[j]) for j in range(task_nums-1)]
        max_data_ibcl = [np.amax(plot_data_ibcl[j]) for j in range(task_nums-1)]
        mean_data_ibcl = [np.mean(plot_data_ibcl[j]) for j in range(task_nums-1)]
        min_data_ibcl = [np.amin(plot_data_ibcl[j]) for j in range(task_nums-1)]
    else:
        mean_data_gem = [np.mean(plot_data_gem[j]) for j in range(task_nums)]
        max_data_vcl = [np.amax(plot_data_vcl[j]) for j in range(task_nums)]
        mean_data_vcl = [np.mean(plot_data_vcl[j]) for j in range(task_nums)]
        min_data_vcl = [np.amin(plot_data_vcl[j]) for j in range(task_nums)]
        max_data_ibcl = [np.amax(plot_data_ibcl[j]) for j in range(task_nums)]
        mean_data_ibcl = [np.mean(plot_data_ibcl[j]) for j in range(task_nums)]
        min_data_ibcl = [np.amin(plot_data_ibcl[j]) for j in range(task_nums)]

    plt.plot(np.array(task_range) + 1, mean_data_gem, '-*', label=f'GEM', color=curve_colors[0], markersize=12)
    plt.plot(np.array(task_range) + 1, max_data_vcl, '-*', label=f'VCL Pareto', color=curve_colors[1], markersize=12)
    plt.plot(np.array(task_range) + 1, mean_data_vcl, '--', label=f'VCL mean', color=curve_colors[1])
    plt.fill_between(np.array(task_range) + 1, max_data_vcl, min_data_vcl, facecolor=fill_colors[1], alpha=0.5,
                     label=f'VCL range')
    plt.plot(np.array(task_range) + 1, max_data_ibcl, '-*', label=f'IBCL Pareto', color=curve_colors[2], markersize=12)
    plt.plot(np.array(task_range) + 1, mean_data_ibcl, '--', label=f'IBCL mean', color=curve_colors[2])
    plt.fill_between(np.array(task_range) + 1, max_data_ibcl, min_data_ibcl, facecolor=fill_colors[2], alpha=0.5, label=f'IBCL range')

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
    plt.title('GEM + VCL + IBCL Performance')
    plt.legend(loc="lower left")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", help="celeba or cifar10", default='cifar10')
    parser.add_argument("--data_dir", help="directory to the preprocessed data",
                        default=os.path.join('data', 'cifar-10-features'))
    parser.add_argument("--alpha", help="alpha value of IBCL in (0, 1)", default=0.5)
    args = parser.parse_args()

    if args.task_name == 'cifar10':
        task_nums = 5
    elif args.task_name == 'celeba':
        task_nums = 40
    else:
        raise NotImplementedError

    # Load dict of accs
    dict_all_accs_gem = torch.load(os.path.join(args.data_dir, f'dict_all_accs_gem.pt'))
    dict_all_accs_vcl = torch.load(os.path.join(args.data_dir, f'dict_all_accs_vcl.pt'))
    if int(args.sublinear) == 0:
        dict_all_accs_ibcl = torch.load(os.path.join(args.data_dir, f'dict_all_accs_{args.alpha}.pt'))
    else:
        dict_all_accs_ibcl = torch.load(os.path.join(args.data_dir, f'dict_all_accs_{args.alpha}_sublinear.pt'))

    # Compute metrics
    dict_avg_accs_gem = avg_per_task_acc(dict_all_accs_gem, task_nums=task_nums, deterministic=True)
    dict_peak_accs_gem = peak_per_task_acc(dict_all_accs_gem, task_nums=task_nums, deterministic=True)
    dict_avg_bts_gem = avg_bt(dict_all_accs_gem, task_nums=task_nums, deterministic=True)
    dict_avg_accs_vcl = avg_per_task_acc(dict_all_accs_vcl, task_nums=task_nums)
    dict_peak_accs_vcl = peak_per_task_acc(dict_all_accs_vcl, task_nums=task_nums)
    dict_avg_bts_vcl = avg_bt(dict_all_accs_vcl, task_nums=task_nums)
    dict_avg_accs_ibcl = avg_per_task_acc(dict_all_accs_ibcl , task_nums=task_nums)
    dict_peak_accs_ibcl = peak_per_task_acc(dict_all_accs_ibcl , task_nums=task_nums)
    dict_avg_bts_ibcl = avg_bt(dict_all_accs_ibcl , task_nums=task_nums)

    # Plot metrics
    plot_metrics_w_baselines(dict_avg_accs_gem, dict_avg_accs_vcl, dict_avg_accs_ibcl, task_nums=task_nums, metric_name='Avg per task accuracy', bt=False)
    plot_metrics_w_baselines(dict_peak_accs_gem, dict_peak_accs_vcl, dict_peak_accs_ibcl, task_nums=task_nums, metric_name='Peak per task accuracy', bt=False)
    plot_metrics_w_baselines(dict_avg_bts_gem, dict_avg_bts_vcl, dict_avg_bts_ibcl, task_nums=task_nums, metric_name='Avg per task backward transfer', bt=True)
