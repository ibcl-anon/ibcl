"""Zero-shot HDR location based on user preferences, given fgcs.pth"""
import os
import argparse
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from scipy.special import erfinv
from sklearn.metrics import accuracy_score
from models.models import BayesianClassifierSmall
from utils.dataloader_utils import get_splitcifar10_test_data, get_celeba_test_data


# Convex combination of param posteriors given a preference, assuming there are a fixed number of extremes per task
def pref_convex_combination(pref: list, num_extremes=3):
    assert np.abs(np.sum(pref) - 1) <= 1e-3  # need to sum to 1, but tolerate computational errors

    num_tasks = len(pref)
    net = BayesianClassifierSmall()  # modify this if you chose normal or large classifiers

    # Prepare weight vector
    weights = []
    for i in range(num_tasks):
        weights += [pref[i] / num_extremes] * num_extremes

    # Convex combination of responsible extreme points
    dists_combined = {}
    for name, _ in net.named_parameters():
        param_mus = []
        param_sigmas = []
        for i in range(num_tasks):
            for j in range(num_extremes):
                model_name = f'{i}_{j}_'
                mu = pyro.param(model_name + name + '_mu')
                sigma = pyro.param(model_name + name + '_sigma')
                param_mus += [mu]
                param_sigmas += [sigma]
        mu_combined = torch.zeros_like(param_mus[0])
        sigma_combined = torch.zeros_like(param_sigmas[0])
        for i in range(len(param_mus)):
            mu_combined += weights[i] * param_mus[i]
            sigma_combined += weights[i] * param_sigmas[i]
        if 'weight' in name:
            dist_combined = dist.Normal(mu_combined, sigma_combined).to_event(2)
        else:
            dist_combined = dist.Normal(mu_combined, sigma_combined).to_event(1)
        dists_combined[name] = dist_combined
    return dists_combined


# Compute icdf of a normal at q-quantile
def normal_icdf(mu, sigma, q):
    standard_normal_quantile = np.sqrt(2) * erfinv(2 * q - 1)
    return mu + sigma * standard_normal_quantile


# Compute HDR of combined dists, assuming all dists are normal
def compute_hdr(dists_combined: dict, alpha=0.1):
    hdr = {}
    for name in dists_combined.keys():
        mu = dists_combined[name].base_dist.loc
        sigma = dists_combined[name].base_dist.scale
        lower_quantile = normal_icdf(mu, sigma, alpha/2)
        upper_quantile = normal_icdf(mu, sigma, 1-alpha/2)
        hdr[name] = [lower_quantile, upper_quantile]
    return hdr


# Uniform sample a parameter in HDR, and test it on tasks
def sample_hdr(hdr: dist):
    net = BayesianClassifierSmall()
    for name, param in net.named_parameters():
        lower_quantile = hdr[name][0]
        upper_quantile = hdr[name][1]
        sampled_param = lower_quantile + torch.rand(1) * (upper_quantile - lower_quantile)
        param.data.copy_(sampled_param)
    return net


# Estimate pareto front by test acc on first two tasks by sample 100 models per HDR
def compute_pareto_front_two_tasks(prefs: list, data_dir, task_name='cifar10', alpha=0.1, n_samples=100):
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
    else:
        raise NotImplementedError
    all_acc0 = []
    all_acc1 = []
    for hdr in hdrs:
        acc0 = []
        acc1 = []
        for i in range(n_samples):
            test_bnn = sample_hdr(hdr)
            output_task0 = test_bnn(torch.tensor(data_test_task0))
            pred_task0 = (output_task0 >= 0.5).long().numpy()
            acc_task0 = accuracy_score(label_test_task0, pred_task0)
            acc0 += [acc_task0]
            output_task1 = test_bnn(torch.tensor(data_test_task1))
            pred_task1 = (output_task1 >= 0.5).long().numpy()
            acc_task1 = accuracy_score(label_test_task1, pred_task1)
            acc1 += [acc_task1]
        all_acc0 += [acc0]
        all_acc1 += [acc1]
    return all_acc0, all_acc1


# For each task, sample prefs, for each pref, sample models
# For each model, compute accuracy on all tasks so far, save in a dict
def gen_acc_dict(all_data_test, all_label_test, data_dir, num_tasks=5, num_prefs_per_task=100, num_models_per_pref=1000, alpha=0.5):
    dict_all_accs = {}
    dict_all_prefs = {}

    for i in range(num_tasks):
        print(f'Evaluating task {i} ...')

        # Randomly sample preferences
        prefs = []
        if i == 0:
            prefs += [[1]]
        else:
            for j in range(num_prefs_per_task):
                pref = np.random.rand(i + 1)
                pref = pref / np.sum(pref)
                prefs += [pref]
        dict_all_prefs[i] = prefs
        torch.save(dict_all_prefs, os.path.join(data_dir, f'dict_all_prefs_{alpha}.pt'))

        # Compute HDRs of these preferences
        hdrs = []
        for pref in prefs:
            dists_combined = pref_convex_combination(pref)
            hdr = compute_hdr(dists_combined, alpha=alpha)
            hdrs += [hdr]

        # Sample models from HDRs and evalaute on all tasks so far
        task_model_accs = []
        for j in range(len(hdrs)):  # each pref
            pref_accs = []
            for k in range(num_models_per_pref):  # each model
                test_bnn = sample_hdr(hdrs[j])
                model_accs = []
                for l in range(i + 1):
                    output = test_bnn(torch.tensor(all_data_test[l]))
                    pred = (output >= 0.5).long().numpy()
                    acc = accuracy_score(all_label_test[l], pred)
                    model_accs += [acc]
                pref_accs += [model_accs]
            task_model_accs += [pref_accs]
        dict_all_accs[i] = task_model_accs
        torch.save(dict_all_accs, os.path.join(data_dir, f'dict_all_accs_{alpha}.pt'))

    return dict_all_prefs, dict_all_accs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", help="celeba or cifar10", default='cifar10')
    parser.add_argument("--data_dir", help="directory to the preprocessed data", default=os.path.join('data', 'cifar-10-features'))
    parser.add_argument("--alpha", help="alpha value of IBCL in (0, 1)", default=0.5)
    parser.add_argument("--num_prefs_per_task", help="numer of preferences per task", default=100)
    parser.add_argument("--num_models_per_pref", help="number of sampled models per preference", default=1000)
    args = parser.parse_args()

    # Load FGCS knowledge base
    pyro.get_param_store().load(os.path.join(args.data_dir, 'fgcs.pth'))

    # Get testing data and labels
    all_data_test, all_label_test = [], []
    if args.task_name == 'cifar10':
        num_tasks = 5
        for i in range(num_tasks):
            data_test, label_test = get_splitcifar10_test_data(args.data_dir, i)
            all_data_test += [data_test]
            all_label_test += [label_test]
    elif args.task_name == 'celeba':
        num_tasks = 40
        for i in range(num_tasks):
            data_test, label_test = get_celeba_test_data(args.data_dir, i)
            all_data_test += [data_test]
            all_label_test += [label_test]
    else:
        raise NotImplementedError

    # Evaluate testing accuracy on randomly sampled preferences
    _, _ = gen_acc_dict(all_data_test, all_label_test, args.data_dir, num_tasks=num_tasks, num_prefs_per_task=args.num_prefs_per_task, num_models_per_pref=args.num_models_per_pref, alpha=args.alpha)
