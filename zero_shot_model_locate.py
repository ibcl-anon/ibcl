"""Zero-shot HDR location based on user preferences, given fgcs.pth"""
import argparse
import pyro
import pyro.distributions as dist
from scipy.special import erfinv
from models.models import BayesianClassifierSmall, BayesianClassifier, BayesianClassifierLarge
from utils.dataloader_utils import *


# Convex combination of param posteriors given a preference, assuming there are a fixed number of extremes per task
def pref_convex_combination(pref: list, model_type='small', num_extremes=3, dict_reuse_map={}):
    assert np.abs(np.sum(pref) - 1) <= 1e-3  # need to sum to 1, but tolerate computational errors

    num_tasks = len(pref)

    if model_type == 'small':
        net = BayesianClassifierSmall()
    elif model_type == 'normal':
        net = BayesianClassifier()
    elif model_type == 'large':
        net = BayesianClassifierLarge()
    else:
        raise NotImplementedError

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

                # If param exists in Pyro buffer, it means it is not discarded, otherwise we need to find the subsitute model by the reuse map
                if model_name + name + '_mu' not in pyro.get_param_store():
                    model_name = dict_reuse_map[model_name]

                mu = pyro.param(model_name + name + '_mu')
                sigma = pyro.param(model_name + name + '_sigma')
                param_mus += [mu]
                param_sigmas += [sigma]
        mu_combined = torch.zeros_like(param_mus[0])
        sigma_combined = torch.zeros_like(param_sigmas[0])
        for i in range(len(param_mus)):
            mu_combined += weights[i] * param_mus[i]
            sigma_combined += weights[i] * param_sigmas[i]
        if 'weight' in name and 'conv' in name:
            dist_combined = dist.Normal(mu_combined, sigma_combined).to_event(4)
        elif 'weight' in name:
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
    beta = 1-alpha
    for name in dists_combined.keys():
        mu = dists_combined[name].base_dist.loc
        sigma = dists_combined[name].base_dist.scale
        lower_quantile = normal_icdf(mu, sigma, beta/2)
        upper_quantile = normal_icdf(mu, sigma, 1-beta/2)
        hdr[name] = [lower_quantile, upper_quantile]
    return hdr


# Uniform sample a parameter in HDR, and test it on tasks
def sample_hdr(hdr: dist, model_type='small'):
    if model_type == 'small':
        net = BayesianClassifierSmall()
    elif model_type == 'normal':
        net = BayesianClassifier()
    elif model_type == 'large':
        net = BayesianClassifierLarge()
    else:
        raise NotImplementedError

    for name, param in net.named_parameters():
        lower_quantile = hdr[name][0]
        upper_quantile = hdr[name][1]
        sampled_param = lower_quantile + torch.rand(1) * (upper_quantile - lower_quantile)
        param.data.copy_(sampled_param)
    return net


def evaluate(sampled_model, data_loader, one_hot=False):
    correct = 0
    total = 0
    for x, y in data_loader:
        y_pred = sampled_model(x).round()
        if one_hot:
            _, y_pred = torch.max(y_pred, 1)
            _, y_true = torch.max(y, 1)
            print(y_pred)
            print(y_true)
            print()
        else:
            y_true = y
        correct += (y_pred == y_true).sum().item()
        total += y.size(0)
    accuracy = correct / total
    return accuracy


# For each pref, sample deterministic models
# For each model, compute pref-weighted accuracy on all tasks so far, save in a dict
def gen_acc_dict(all_data_test, all_label_test, data_dir, model_type='small', num_tasks=5, num_models_per_pref=10, alpha=0.5, discard_threshold=0.0, dict_reuse_map={}):

    dict_prefs = torch.load(os.path.join(data_dir, 'dict_prefs.pt'))
    dict_all_accs = {}

    for i in range(num_tasks):
        print(f'Evaluating task {i} ...')
        prefs = dict_prefs[i]

        # Compute HDRs of these preferences
        hdrs = []
        for pref in prefs:
            dists_combined = pref_convex_combination(pref, model_type=model_type, dict_reuse_map=dict_reuse_map)
            hdr = compute_hdr(dists_combined, alpha=alpha)
            hdrs += [hdr]

        # Sample models from HDRs and evalaute on all tasks so far
        task_model_accs = []
        for j in range(len(hdrs)):  # each pref
            pref_accs = []  # acc of all sampled models to address this pref
            for k in range(num_models_per_pref):  # each model
                test_bnn = sample_hdr(hdrs[j], model_type=model_type)
                model_accs = []  # acc of a sampled model
                for l in range(i + 1):
                    data_test = np.array(all_data_test[l])
                    labels_test = np.array(all_label_test[l])
                    testloader = get_dataloader(data=data_test, labels=labels_test, shuffle=False)
                    acc = evaluate(sampled_model=test_bnn, data_loader=testloader, one_hot=(model_type=='rmnist'))
                    print(f'Acc on {i}, {j}, {k}, {l} = {acc}')
                    # if model_type != 'rmnist':  # binary classification
                    #     pred = (output >= 0.5).long().numpy()
                    #     label = all_label_test[l]
                    # else:  # multiclass classification
                    #     pred = np.argmax(output.detach().numpy(), axis=1)
                    #     label = np.argmax(all_label_test[l], axis=1)
                    # acc = accuracy_score(label, pred)
                    model_accs += [acc]
                pref_accs += [model_accs]
            task_model_accs += [pref_accs]
        dict_all_accs[i] = task_model_accs
        if discard_threshold == 0.0:
            torch.save(dict_all_accs, os.path.join(data_dir, f'dict_all_accs_{alpha}.pt'))
        else:
            torch.save(dict_all_accs, os.path.join(data_dir, f'dict_all_accs_{alpha}_{discard_threshold}.pt'))

    return dict_all_accs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", help="celeba or cifar10", default='cifar10')
    parser.add_argument("--data_dir", help="directory to the preprocessed data", default=os.path.join('data', 'cifar100_proc_data'))
    parser.add_argument("--model_size", help="small, normal or large", default='small')
    parser.add_argument("--alpha", help="alpha value of IBCL in (0, 1)", default=0.01)
    parser.add_argument("--num_models_per_pref", help="number of sampled models per preference", default=10)
    parser.add_argument("--discard_threshold", help="d threshold of sublinear fgcs", default=0.01)
    args = parser.parse_args()

    # Load FGCS knowledge base
    if args.discard_threshold == 0.0:
        pyro.get_param_store().load(os.path.join(args.data_dir, 'fgcs.pth'))
        dict_reuse_map = {}
    else:
        dict_reuse_map = torch.load(os.path.join(args.data_dir, f'dict_reuse_map_{args.discard_threshold}.pt'))
        pyro.get_param_store().load(os.path.join(args.data_dir, f'fgcs_sublinear_{args.discard_threshold}.pth'))

    # Get testing data and labels
    all_data_test, all_label_test = [], []
    if args.task_name == 'cifar10':
        num_tasks = 5
        for i in range(num_tasks):
            data_test, label_test = get_splitcifar10_test_data(args.data_dir, i)
            all_data_test += [data_test]
            all_label_test += [label_test]
    elif args.task_name == 'celeba':
        num_tasks = 15
        for i in range(num_tasks):
            data_test, label_test = get_celeba_test_data(args.data_dir, i)
            all_data_test += [data_test]
            all_label_test += [label_test]
    elif args.task_name == 'cifar100':
        num_tasks = 10
        for i in range(num_tasks):
            data_test, label_test = get_splitcifar100_test_data(args.data_dir, i)
            all_data_test += [data_test]
            all_label_test += [label_test]
    elif args.task_name == 'tinyimagenet':
        num_tasks = 10
        for i in range(num_tasks):
            data_test, label_test = get_tinyimagenet_test_data(args.data_dir, i)
            all_data_test += [data_test]
            all_label_test += [label_test]
    elif args.task_name == '20newsgroup':
        num_tasks = 5
        for i in range(num_tasks):
            data_test, label_test = get_20newsgroup_test_data(args.data_dir, i)
            all_data_test += [data_test]
            all_label_test += [label_test]
    else:
        raise NotImplementedError

    # Evaluate testing accuracy on randomly sampled preferences
    _ = gen_acc_dict(all_data_test, all_label_test, args.data_dir, model_type=args.model_size, num_tasks=int(num_tasks), num_models_per_pref=int(args.num_models_per_pref), alpha=float(args.alpha), discard_threshold=float(args.discard_threshold), dict_reuse_map=dict_reuse_map)
