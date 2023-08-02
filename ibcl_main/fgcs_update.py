""" IBCL training to update FGCS checkpoints across a task sequence, results stored in fgcs.pth """

import os
import argparse
import numpy as np
import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.infer import SVI
from pyro.optim import Adam
from torch.utils.data import DataLoader
from models.models import BayesianClassifier, BayesianClassifierLarge, BayesianClassifierSmall
from models.loss_funcs import CustomTrace_ELBO
from utils.dataloader_utils import get_celeba_loaders, get_splitcifar10_loaders


# Generate Gaussian priors of first task
def first_prior(std=0.3, model_type='small'):
    if model_type == 'normal':
        net = BayesianClassifier()  # just for param shapes
    elif model_type == 'small':
        net = BayesianClassifierSmall()
    elif model_type == 'large':
        net = BayesianClassifierLarge()
    else:
        raise NotImplementedError
    prior = {}
    for name, param in net.named_parameters():
        if 'weight' in name:  # weight param
            param_prior = dist.Normal(torch.zeros_like(param), std * torch.ones_like(param)).to_event(2)
        else:  # bias param
            param_prior = dist.Normal(torch.zeros_like(param), std * torch.ones_like(param)).to_event(1)
        prior[name] = param_prior
    return prior


# General model function, to assign arbitrary prior distributions
def general_model(net: torch.nn.Module, task_ind: int, prior: dict):

    def model(x, y):
        lifted_module = pyro.random_module("module", net, prior)
        lifted_reg_model = lifted_module()
        with pyro.plate("map", len(x)):
            prediction = lifted_reg_model(x)
            pyro.sample(str(task_ind) + "_obs", dist.Bernoulli(prediction), obs=y)

    return model


# General guide function to register VI posteriors of parameters
def general_guide(net: torch.nn.Module, task_ind: int, extremum_ind: int):
    model_name = str(task_ind) + '_' + str(extremum_ind) + '_'

    def guide(x, y):
        dists = {}
        for name, param in net.named_parameters():
            # param of param
            param_mu = pyro.param(model_name + name + '_mu', torch.zeros_like(param))
            param_sigma = pyro.param(model_name + name + '_sigma', torch.ones_like(param),
                                     constraint=constraints.positive)
            if 'weight' in name:  # weight param
                param_prior_vi = dist.Normal(param_mu, param_sigma).to_event(2)
            else:  # bias param
                param_prior_vi = dist.Normal(param_mu, param_sigma).to_event(1)
            dists[name] = param_prior_vi
        lifted_module = pyro.random_module("module", net, dists)
        return lifted_module()

    return guide


# Define the training loop, return running losses
def train(svi, train_data_loader, val_data_loader, guide, epochs=10, verbose=False):
    running_loss = []
    val_acc = []
    for epoch in range(epochs):
        # Training
        loss = 0.
        for x, y in train_data_loader:
            loss += svi.step(x, y.float())
        avg_loss = loss / len(train_data_loader)
        running_loss += [avg_loss]
        if verbose:
            print(f"Epoch {epoch+1}, Loss: {avg_loss}")
        # Validation
        train_accuracy = evaluate(guide, train_data_loader)
        val_accuracy = evaluate(guide, val_data_loader)  # validation data acc
        val_acc += [val_accuracy]
        if verbose:
            print(f"Train accuracy: {train_accuracy * 100:.2f}%")
            print(f"Val accuracy: {val_accuracy * 100:.2f}%")
    return running_loss, val_acc


# Evaluate the model
def evaluate(guide, data_loader):
    correct = 0
    total = 0
    for x, y in data_loader:
        sampled_model = guide(None, None)
        y_pred = sampled_model(x).round()
        correct += (y_pred == y).sum().item()
        total += y.size(0)
    accuracy = correct / total
    return accuracy


# Knowledge is saved in pyro.get_param_store()
# priors: posteriors from previous task
def fgcs_update(train_data_loader: DataLoader, val_data_loader: DataLoader, priors: list, task_ind: int, model_type='small', num_extremes=3, lr=1e-3, epochs=10, verbose=False):
    running_losses = []
    val_accs = []

    # Instantiate the networks
    if model_type == 'normal':
        nets = [BayesianClassifier() for _ in range(num_extremes)]
    elif model_type == 'small':
        nets = [BayesianClassifierSmall() for _ in range(num_extremes)]
    elif model_type == 'large':
        nets = [BayesianClassifierLarge() for _ in range(num_extremes)]
    else:
        raise NotImplementedError

    # Initialize models and guides
    models = [general_model(nets[j], task_ind, j, priors[j]) for j in range(num_extremes)]
    guides = [general_guide(nets[j], task_ind, j) for j in range(num_extremes)]


    # Train the models
    for j in range(num_extremes):
        if verbose:
            print(f'Training on task {task_ind}, extreme point {j} ...')
        optimizers = [Adam({"lr": lr}) for _ in range(num_extremes)]
        svis = [SVI(models[j], guides[j], optimizers[j], loss=CustomTrace_ELBO()) for j in range(num_extremes)]
        running_loss, val_acc = train(svis[j], train_data_loader, val_data_loader, guides[j], epochs=epochs, verbose=verbose)
        running_losses += [running_loss]
        val_accs += [val_acc]

    # Obtain posteriors
    posteriors = []
    for j in range(num_extremes):
        model_name = str(task_ind) + '_' + str(j) + '_'
        posterior = {}
        for name, param in nets[j].named_parameters():
            if 'weight' in name:
                param_posterior = dist.Normal(pyro.param(model_name + name + '_mu'),
                                              pyro.param(model_name + name + '_sigma')).to_event(2)
            else:
                param_posterior = dist.Normal(pyro.param(model_name + name + '_mu'),
                                              pyro.param(model_name + name + '_sigma')).to_event(1)
            posterior[name] = param_posterior
        posteriors += [posterior]

    return posteriors, running_losses, val_accs


# Main loop
def fgcs_update_main(data_dir, tasks, model_type='small', lr=1e-3, epochs=10, verbose=True, task_name='cifar10'):
    priors = [first_prior(std, model_type=model_type) for std in [0.25, 0.3, 0.35]]
    num_extremes = len(priors)
    loss_logs = []
    val_acc_logs = []
    for i in tasks:
        if task_name == 'cifar10':
            task_train_loader, task_val_loader = get_splitcifar10_loaders(data_dir, i)
        elif task_name == 'celeba':
            task_train_loader, task_val_loader = get_celeba_loaders(data_dir, i)
        else:
            raise NotImplementedError
        posteriors, running_losses, val_accs = fgcs_update(task_train_loader, task_val_loader, priors, model_type=model_type, task_ind=i, num_extremes=num_extremes, lr=lr, epochs=epochs, verbose=verbose)
        priors = posteriors
        loss_logs += [running_losses]
        val_acc_logs += [val_accs]
        print(running_losses)
        print(val_accs)

        # Save info for every task
        pyro.get_param_store().save(os.path.join(data_dir, 'fgcs.pth'))
        np.save(os.path.join(data_dir, 'loss_logs.npy'), loss_logs)
        np.save(os.path.join(data_dir, 'val_acc_logs.npy'), val_acc_logs)

    return loss_logs, val_acc_logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", help="celeba or cifar10", default='cifar10')
    parser.add_argument("--data_dir", help="directory to the preprocessed data", default=os.path.join('data', 'cifar-10-features'))
    parser.add_argument("--model_size", help="small, normal or large", default='small')

    args = parser.parse_args()
    lr, epochs = 1e-3, 10

    if args.task_name == 'cifar10':
        tasks = range(5)
    elif args.task_name == 'celeba':
        tasks = range(40)
    else:
        raise NotImplementedError

    # An example training call, model type can be 'large', 'normal' or 'small'
    # Result FGCS will be stored in fgcs.pth
    _, _ = fgcs_update_main(args.data_dir, tasks=tasks, model_type=args.model_size, lr=lr, epochs=epochs, task_name=args.task_name)
