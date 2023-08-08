import numpy as np
import torch
from torch.optim import Adam
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.optim import Adam
import os
import argparse
from sklearn.metrics import accuracy_score
from models.models import BayesianClassifier, BayesianClassifierLarge, BayesianClassifierSmall
from models.loss_funcs import CustomTrace_ELBO, CustomSVI
from utils.dataloader_utils import get_celeba_loaders, get_splitcifar10_loaders
from utils.dataloader_utils import get_celeba_test_data, get_splitcifar10_test_data


# Variational continual learning by Nguyen et al. 2018
class VCLGEM():

    def __init__(self, net, prior, num_tasks, memory_size_per_task=500, lr=1e-3, epochs=5):
        self.net = net
        self.prior = prior
        self.num_tasks = num_tasks
        self.memory = {task: {'data': None, 'targets': None} for task in range(num_tasks)}
        self.memory_size_per_task = memory_size_per_task
        self.current_task = 0
        self.lr = lr
        self.epochs = epochs

    def remember(self, dataloader):
        all_data = []
        all_targets = []
        for data, targets in dataloader:
            all_data += [data]
            all_targets += [targets]
        all_data = torch.cat(all_data)
        all_targets = torch.cat(all_targets)
        data = all_data[:self.memory_size_per_task]
        targets = all_targets[:self.memory_size_per_task]
        self.memory[self.current_task] = {'data': data, 'targets': targets}

    def general_model(self):

        def model(x, y):
            lifted_module = pyro.random_module("module", self.net, self.prior)
            lifted_reg_model = lifted_module()
            with pyro.plate("map", len(x)):
                prediction = lifted_reg_model(x)
                pyro.sample("obs", dist.Bernoulli(prediction), obs=y)
            return

        return model

    def general_guide(self):

        def guide(x, y):
            dists = {}
            for name, param in self.net.named_parameters():
                # param of param
                param_mu = pyro.param(name + '_mu', torch.zeros_like(param))
                param_sigma = pyro.param(name + '_sigma', torch.ones_like(param), constraint=constraints.positive)
                if 'weight' in name:  # weight param
                    param_prior_vi = dist.Normal(param_mu, param_sigma).to_event(2)
                else:  # bias param
                    param_prior_vi = dist.Normal(param_mu, param_sigma).to_event(1)
                dists[name] = param_prior_vi
            lifted_module = pyro.random_module("module", self.net, dists)
            return lifted_module()

        return guide

    def compute_loss(self, svi, data, targets, pref):
        total_loss = 0.0
        for task in range(self.current_task + 1):
            if self.memory[task]['data'] is not None:
                old_data, old_targets = self.memory[task]['data'], self.memory[task]['targets']
                old_loss = svi.step(pref[task], old_data, old_targets)
                #                 old_loss = svi.step(old_data, old_targets)
                total_loss += pref[task] * old_loss  # Loss regularized by preferences
        new_loss = svi.step(pref[self.current_task], data, targets)
        #         new_loss = svi.step(data, targets)
        total_loss += pref[self.current_task] * new_loss  # Loss regularized by preferences
        return total_loss

    def learn(self, dataloader, pref):
        optimizer = Adam({"lr": self.lr})
        svi = CustomSVI(self.general_model(), self.general_guide(), optimizer, loss=CustomTrace_ELBO())
        for i in range(self.epochs):
            print(f'Training on task {self.current_task}, pref {pref}, epoch {i} ...')
            for data, targets in dataloader:
                _ = self.compute_loss(svi, data, targets.float(), pref)
        return


# Priors of first task
def first_prior(net: torch.nn.Module, std=1.0):
    prior = {}
    for name, param in net.named_parameters():
        if 'weight' in name: # weight param
            param_prior = dist.Normal(torch.zeros_like(param), std * torch.ones_like(param)).to_event(2)
        else: # bias param
            param_prior = dist.Normal(torch.zeros_like(param), std * torch.ones_like(param)).to_event(1)
        prior[name] = param_prior
    return prior


# Main VCL
def vcl_main(data_dir, model_type='small', task_name='cifar10', num_prefs_per_task=10, num_models_per_pref=10):
    torch.manual_seed(42)

    if model_type == 'small':
        net = BayesianClassifierSmall()
    elif model_type == 'normal':
        net = BayesianClassifier()
    elif model_type == 'large':
        net = BayesianClassifierLarge()
    else:
        raise NotImplementedError

    if task_name == 'cifar10':
        num_tasks = 5
    elif task_name == 'celeba':
        num_tasks = 40
    else:
        raise NotImplementedError

    prior = first_prior(net, std=0.3)
    vclgem = VCLGEM(net, prior, num_tasks, memory_size_per_task=500, lr=1e-3, epochs=10)
    dict_all_accs = {}
    dict_all_prefs = {}

    for i in range(num_tasks):

        if task_name == 'cifar10':
            task_train_loader, task_val_loader = get_splitcifar10_loaders(data_dir, i)
            task_test_data, task_test_labels = get_splitcifar10_test_data(data_dir, i)
        elif task_name == 'celeba':
            task_train_loader, task_val_loader = get_celeba_loaders(data_dir, i)
            task_test_data, task_test_labels = get_celeba_test_data(data_dir, i)
        else:
            raise NotImplementedError

        # Randomly generate prefs
        prefs = []
        if i == 0:
            prefs += [[1]]
        else:
            for j in range(num_prefs_per_task):
                pref = np.random.rand(i + 1)
                pref = pref / np.sum(pref)
                prefs += [pref]
        dict_all_prefs[i] = prefs
        torch.save(dict_all_prefs, os.path.join(data_dir, 'dict_all_prefs_vcl.pt'))

        task_accs = []
        for pref in prefs:

            # Training on a pref
            vclgem.learn(task_train_loader, pref)
            guide = vclgem.general_guide()

            # Testing
            pref_models_accs = []
            for j in range(num_models_per_pref):  # sample models
                sampled_model = guide(None, None)
                sampled_model_accs = []
                for k in range(i + 1):  # for each sampled model, test on all tasks
                    task_test_data, task_test_labels = get_celeba_test_data(data_dir, k)
                    task_test_data = torch.Tensor(task_test_data)
                    task_test_labels = torch.Tensor(task_test_labels)
                    outputs = sampled_model(task_test_data)
                    pred = (outputs >= 0.5).long().numpy()
                    acc = accuracy_score(task_test_labels, pred)
                    sampled_model_accs += [acc]
                pref_models_accs += [sampled_model_accs]

            task_accs += [pref_models_accs]

        print(f'task accs: {task_accs}')
        dict_all_accs[i] = task_accs
        torch.save(dict_all_accs, os.path.join(data_dir, 'dict_all_accs_vcl.pt'))

        # Memorize some training data
        vclgem.remember(task_train_loader)

        # Obtain posteriors and assign it to next prior
        posterior = {}
        for name, param in net.named_parameters():
            if 'weight' in name:
                param_posterior = dist.Normal(pyro.param(name + '_mu'), pyro.param(name + '_sigma')).to_event(2)
            else:
                param_posterior = dist.Normal(pyro.param(name + '_mu'), pyro.param(name + '_sigma')).to_event(1)
            posterior[name] = param_posterior
        vclgem.prior = posterior

        # Update task number
        vclgem.current_task += 1

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", help="celeba or cifar10", default='cifar10')
    parser.add_argument("--data_dir", help="directory to the preprocessed data",
                        default=os.path.join('data', 'cifar-10-features'))
    parser.add_argument("--model_size", help="small, normal or large", default='small')
    parser.add_argument("--num_prefs_per_task", help="numer of preferences per task", default=10)
    parser.add_argument("--num_models_per_pref", help="numer of sampled models per preference", default=10)
    args = parser.parse_args()

    vcl_main(args.data_dir, model_type=args.model_size, task_name=args.task_name,
             num_prefs_per_task=int(args.num_prefs_per_task), num_models_per_pref=int(args.num_models_per_pref))
