from zero_shot_model_locate import *
import matplotlib.pyplot as plt


# Estimate pareto front by test acc on first two tasks by sample 100 models per HDR
def compute_pareto_front_two_tasks(prefs: list, data_dir, model_type='small', task_name='cifar10', alpha=0.5, n_samples=200):
    # Compute one HDR per preference
    hdrs = []
    for pref in prefs:
        dists_combined = pref_convex_combination(pref)
        hdr = compute_hdr(dists_combined, alpha=alpha)
        hdrs += [hdr]
    # Sample 100 models per HDR, obtain their acc on task0 and task1
    if task_name == 'cifar10':
        data_test_task0, label_test_task0 = get_splitcifar10_test_data(data_dir, task_ind=0)
        data_test_task1, label_test_task1 = get_splitcifar10_test_data(data_dir, task_ind=1)
    elif task_name == 'celeba':
        data_test_task0, label_test_task0 = get_celeba_test_data(data_dir, task_ind=0)
        data_test_task1, label_test_task1 = get_celeba_test_data(data_dir, task_ind=1)
    elif task_name == 'cifar100':
        data_test_task0, label_test_task0 = get_splitcifar100_test_data(data_dir, task_ind=0)
        data_test_task1, label_test_task1 = get_splitcifar100_test_data(data_dir, task_ind=1)
    elif task_name == 'tinyimagenet':
        data_test_task0, label_test_task0 = get_tinyimagenet_test_data(data_dir, task_ind=0)
        data_test_task1, label_test_task1 = get_tinyimagenet_test_data(data_dir, task_ind=1)
    elif task_name == 'rmnist':
        data_test_task0, label_test_task0 = get_rmnist_test_data(data_dir, task_ind=0)
        data_test_task1, label_test_task1 = get_rmnist_test_data(data_dir, task_ind=1)
    elif task_name == '20newsgroup':
        data_test_task0, label_test_task0 = get_20newsgroup_test_data(data_dir, task_ind=0)
        data_test_task1, label_test_task1 = get_20newsgroup_test_data(data_dir, task_ind=1)
    else:
        raise NotImplementedError
    labels_test_task0 = np.array(label_test_task0)
    labels_test_task1 = np.array(label_test_task1)
    testloader_task0 = get_dataloader(data=data_test_task0, labels=labels_test_task0, shuffle=False)
    testloader_task1 = get_dataloader(data=data_test_task1, labels=labels_test_task1, shuffle=False)
    all_acc0 = []
    all_acc1 = []
    for hdr in hdrs:
        acc0 = []
        acc1 = []
        for i in range(n_samples):
            test_bnn = sample_hdr(hdr, model_type=model_type)
            # output_task0 = test_bnn(torch.tensor(data_test_task0))
            # if task_name != 'rmnist':
            #     pred_task0 = (output_task0 >= 0.5).long().numpy()
            # else:
            #     pred_task0 = output_task0.numpy()
            # acc_task0 = accuracy_score(label_test_task0, pred_task0)
            acc_task0 = evaluate(sampled_model=test_bnn, data_loader=testloader_task0, one_hot=(model_type == 'rmnist'))
            acc0 += [acc_task0]
            # output_task1 = test_bnn(torch.tensor(data_test_task1))
            # if task_name != 'rmnist':
            #     pred_task1 = (output_task1 >= 0.5).long().numpy()
            # else:
            #     pred_task1 = output_task1.numpy()
            # acc_task1 = accuracy_score(label_test_task1, pred_task1)
            acc_task1 = evaluate(sampled_model=test_bnn, data_loader=testloader_task1, one_hot=(model_type == 'rmnist'))
            acc1 += [acc_task1]
        all_acc0 += [acc0]
        all_acc1 += [acc1]
    return all_acc0, all_acc1


def max_sum_index(arr_1, arr_2):
    # Pair up elements from the two arrays along with their index
    paired = [(sum_val, idx) for idx, sum_val in enumerate(map(lambda x: x[0]**2 + x[1]**2, zip(arr_1, arr_2)))]
    # Get the index of the maximum sum
    _, max_index = max(paired, key=lambda x: x[0])
    return max_index


if __name__ == '__main__':
    prefs = [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.9, 0.1]]
    data_dir = 'cifar100_proc_data'
    task_name = 'cifar100'
    pyro.get_param_store().load(os.path.join(data_dir, 'fgcs.pth'))
    all_acc0, all_acc1 = compute_pareto_front_two_tasks(prefs, data_dir=data_dir, task_name=task_name, alpha=0.95)
    print(all_acc0)
    print(all_acc1)

    # plt.figure(figsize=(3, 2.5), dpi=150)

    est_pareto_x = []
    est_pareto_y = []
    for k in range(len(prefs)):
        plt.scatter(all_acc0[k], all_acc1[k], label=f'{prefs[k]}', s=12)
        ind_k = max_sum_index(all_acc0[k], all_acc1[k])
        est_pareto_x += [all_acc0[k][ind_k]]
        est_pareto_y += [all_acc1[k][ind_k]]

    plt.plot(est_pareto_x, est_pareto_y, linestyle='-', marker='*', label='est Pareto', markersize=12, c='blue')

    plt.xlabel('accuracy task 1')
    plt.ylabel('accuracy task 2')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
    plt.tight_layout()

    plt.show()
