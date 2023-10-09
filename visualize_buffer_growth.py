from visualize_d import *


def plot_buffer_growth(buffer_0, buffer_1, task_nums=10):
    curve_colors = ['purple', 'orangered', 'blue']

    plt.figure(figsize=(3, 2.5), dpi=150)

    plt.plot(np.arange(0, task_nums + 1), buffer_0, '-*', label=f'd=8e-3 Pareto', color=curve_colors[0],
             markersize=8)
    plt.plot(np.arange(0, task_nums + 1), buffer_1, '-*', label=f'd=8e-3 Pareto', color=curve_colors[1],
             markersize=8)
    plt.plot(np.arange(0, task_nums + 1), np.arange(0, 3 * task_nums + 3, 3), '-*', label=f'd=8e-3 Pareto',
             color=curve_colors[2],
             markersize=8)
    plt.xticks(np.arange(0, task_nums+1))
    plt.xlabel('Task num')
    plt.ylabel('Buffered model num')
    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    data_dir = 'cifar100_proc_data'
    task_nums = 10
    buffer_0 = torch.load(os.path.join(data_dir, 'sublinear_buffer_growth_0.005.pt'))
    buffer_1 = torch.load(os.path.join(data_dir, 'sublinear_buffer_growth_0.008.pt'))
    plot_buffer_growth(buffer_0, buffer_1, task_nums=task_nums)