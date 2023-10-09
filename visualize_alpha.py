import matplotlib.pyplot as plt
from visualize_results_w_baselines import *


def plot_metric_alpha(dict_0, dict_1, dict_2, task_nums=10, bt=False, metric_name='avg per task acc'):

    plt.figure(figsize=(3, 2.5), dpi=150)

    dicts = [dict_0, dict_1, dict_2]
    curve_colors = ['purple', 'orangered', 'blue']
    fill_colors = ['violet', 'lightsalmon', 'tab:blue']

    if bt:
        task_range = list(range(1, task_nums))
        plot_range = list(range(task_nums - 1))
        plt.hlines(y=0.0, xmin=2, xmax=task_nums, colors='red', linestyles='-')
    else:
        task_range = list(range(task_nums))
        plot_range = list(range(task_nums))

    task_range = [int(j) for j in task_range]
    plot_range = [int(j) for j in plot_range]

    plot_data_0 = [dicts[0][j] for j in task_range]
    plot_data_1 = [dicts[1][j] for j in task_range]
    plot_data_2 = [dicts[2][j] for j in task_range]

    max_data_0 = [np.amax(plot_data_0[j]) for j in plot_range]
    mean_data_0 = [np.mean(plot_data_0[j]) for j in plot_range]
    min_data_0 = [np.amin(plot_data_0[j]) for j in plot_range]

    max_data_1 = [np.amax(plot_data_1[j]) for j in plot_range]
    mean_data_1 = [np.mean(plot_data_1[j]) for j in plot_range]
    min_data_1 = [np.amin(plot_data_1[j]) for j in plot_range]

    max_data_2 = [np.amax(plot_data_2[j]) for j in plot_range]
    mean_data_2 = [np.mean(plot_data_2[j]) for j in plot_range]
    min_data_2 = [np.amin(plot_data_2[j]) for j in plot_range]

    plt.plot(np.array(task_range) + 1, max_data_0, '-*', label=f'alpha=0.75 Pareto', color=curve_colors[0],
             markersize=8)
    plt.plot(np.array(task_range) + 1, mean_data_0, '--', label=f'alpha=0.75 mean', color=curve_colors[0])
    plt.fill_between(np.array(task_range) + 1, max_data_0, min_data_0, facecolor=fill_colors[0], alpha=0.5,
                     label=f'alpha=0.75 range')

    plt.plot(np.array(task_range) + 1, max_data_1, '-*', label=f'alpha=0.90 Pareto', color=curve_colors[1],
             markersize=8)
    plt.plot(np.array(task_range) + 1, mean_data_1, '--', label=f'alpha=0.90 mean', color=curve_colors[1])
    plt.fill_between(np.array(task_range) + 1, max_data_1, min_data_1, facecolor=fill_colors[1], alpha=0.5,
                     label=f'alpha=0.90 range')

    plt.plot(np.array(task_range) + 1, max_data_2, '-*', label=f'alpha=0.99 Pareto', color=curve_colors[2], markersize=8)
    plt.plot(np.array(task_range) + 1, mean_data_2, '--', label=f'alpha=0.99 mean', color=curve_colors[2])
    plt.fill_between(np.array(task_range) + 1, max_data_2, min_data_2, facecolor=fill_colors[2], alpha=0.5,
                     label=f'alpha=0.99 range')

    if bt:
        plt.ylim([-0.5, 0.5])
    else:
        plt.ylim([0.0, 1.0])

    plt.xticks(np.array(task_range) + 1)
    plt.xlabel('Task num')
    plt.ylabel(metric_name)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    plt.tight_layout()
    plt.show()

    return


if __name__ == '__main__':

    data_dir = '20newsgroup_proc_data'
    task_nums = 5

    # Load dict prefs
    dict_prefs = torch.load(os.path.join(data_dir, 'dict_prefs.pt'))

    dict_all_accs_0 = torch.load(os.path.join(data_dir, f'dict_all_accs_0.75.pt'))
    dict_all_accs_1 = torch.load(os.path.join(data_dir, f'dict_all_accs_0.9.pt'))
    dict_all_accs_2 = torch.load(os.path.join(data_dir, f'dict_all_accs_0.99.pt'))

    dict_matrix_0 = generate_acc_matrix(dict_all_accs_0, dict_prefs, task_nums=task_nums, deterministic=False)
    dict_matrix_1 = generate_acc_matrix(dict_all_accs_1, dict_prefs, task_nums=task_nums, deterministic=False)
    dict_matrix_2 = generate_acc_matrix(dict_all_accs_2, dict_prefs, task_nums=task_nums, deterministic=False)

    dict_avg_accs_0 = avg_per_task_acc(dict_matrix_0, task_nums=task_nums)
    dict_avg_accs_1 = avg_per_task_acc(dict_matrix_1, task_nums=task_nums)
    dict_avg_accs_2 = avg_per_task_acc(dict_matrix_2, task_nums=task_nums)

    dict_peak_accs_0 = peak_per_task_acc(dict_matrix_0, task_nums=task_nums)
    dict_peak_accs_1 = peak_per_task_acc(dict_matrix_1, task_nums=task_nums)
    dict_peak_accs_2 = peak_per_task_acc(dict_matrix_2, task_nums=task_nums)

    dict_avg_bts_0 = avg_bt(dict_matrix_0, task_nums=task_nums)
    dict_avg_bts_1 = avg_bt(dict_matrix_1, task_nums=task_nums)
    dict_avg_bts_2 = avg_bt(dict_matrix_2, task_nums=task_nums)

    plot_metric_alpha(dict_avg_accs_0, dict_avg_accs_1, dict_avg_accs_2, task_nums=task_nums, metric_name='avg per task acc')
    plot_metric_alpha(dict_peak_accs_0, dict_peak_accs_1, dict_peak_accs_2, task_nums=task_nums,
                      metric_name='peak per task acc')
    plot_metric_alpha(dict_avg_bts_0, dict_avg_bts_1, dict_avg_bts_2, task_nums=task_nums,
                      metric_name='avg per task bt', bt=True)