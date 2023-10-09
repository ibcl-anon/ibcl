""" IBCL training to update FGCS checkpoints across a task sequence, results stored in fgcs.pth """

import argparse
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.infer import SVI
from pyro.optim import Adam
import matplotlib.pyplot as plt
from models.models import BayesianClassifier, BayesianClassifierLarge, BayesianClassifierSmall
from models.loss_funcs import CustomTrace_ELBO
from utils.dataloader_utils import *


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
        if 'conv' in name and 'weight' in name:
            param_prior = dist.Normal(torch.zeros_like(param), std * torch.ones_like(param)).to_event(4)
        elif 'weight' in name:  # weight param
            param_prior = dist.Normal(torch.zeros_like(param), std * torch.ones_like(param)).to_event(2)
        else:  # bias param
            param_prior = dist.Normal(torch.zeros_like(param), std * torch.ones_like(param)).to_event(1)
        prior[name] = param_prior
    return prior


# General model function, to assign arbitrary prior distributions
def general_model(net: torch.nn.Module, task_ind: int, prior: dict, one_hot=False):

    def model(x, y):
        lifted_module = pyro.random_module("module", net, prior)
        lifted_reg_model = lifted_module()
        with pyro.plate("map", len(x)):
            prediction = lifted_reg_model(x)
            if not one_hot:
                pyro.sample(str(task_ind) + "_obs", dist.Bernoulli(prediction), obs=y)
            else:
                pyro.sample(str(task_ind) + "_obs", dist.OneHotCategorical(prediction), obs=y)
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
            if 'weight' in name and 'conv' in name:
                param_prior_vi = dist.Normal(param_mu, param_sigma).to_event(4)
            elif 'weight' in name:  # weight param
                param_prior_vi = dist.Normal(param_mu, param_sigma).to_event(2)
            else:  # bias param
                param_prior_vi = dist.Normal(param_mu, param_sigma).to_event(1)
            dists[name] = param_prior_vi
        lifted_module = pyro.random_module("module", net, dists)
        return lifted_module()

    return guide


# Define the training loop, return running losses
def train(svi, train_data_loader, val_data_loader, guide, epochs=10, verbose=False, one_hot=False):
    running_loss = []
    val_acc = []
    for epoch in range(epochs):
        # Training
        loss = 0.
        count = 0
        for x, y in train_data_loader:
            count += 1
            loss += svi.step(x, y.float())
        print(f'count: {count}')
        avg_loss = loss / len(train_data_loader)
        running_loss += [avg_loss]
        train_accuracy = evaluate(guide, train_data_loader, one_hot=one_hot)
        if verbose:
            print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
            print(f"Train accuracy: {train_accuracy * 100:.2f}%")
        if val_data_loader is not None:
            val_accuracy = evaluate(guide, val_data_loader, one_hot=one_hot)  # validation data acc
            val_acc += [val_accuracy]
            if verbose:
                print(f"Val accuracy: {val_accuracy * 100:.2f}%")
    return running_loss, val_acc


# Evaluate the model
def evaluate(guide, data_loader, one_hot=False):
    correct = 0
    total = 0
    for x, y in data_loader:
        sampled_model = guide(None, None)
        y_pred = sampled_model(x).round()
        if one_hot:
            _, y_pred = torch.max(y_pred, 1)
            _, y_true = torch.max(y, 1)
        else:
            y_true = y
        correct += (y_pred == y_true).sum().item()
        total += y.size(0)
    accuracy = correct / total
    return accuracy


# Get posterior dict by name
def get_posterior_by_name(model_name, model_type='small'):
    if model_type == 'small':
        net = BayesianClassifierSmall()
    elif model_type == 'normal':
        net = BayesianClassifier()
    elif model_type == 'large':
        net = BayesianClassifierLarge()
    else:
        raise NotImplementedError
    posterior = {}
    for name, param in net.named_parameters():
        if 'weight' in name and 'conv' in name:
            param_posterior = dist.Normal(pyro.param(model_name + name + '_mu'),
                                          pyro.param(model_name + name + '_sigma')).to_event(4)
        elif 'weight' in name:
            param_posterior = dist.Normal(pyro.param(model_name + name + '_mu'),
                                          pyro.param(model_name + name + '_sigma')).to_event(2)
        else:
            param_posterior = dist.Normal(pyro.param(model_name + name + '_mu'),
                                          pyro.param(model_name + name + '_sigma')).to_event(1)
        posterior[name] = param_posterior
    return posterior


# Extract parameter values from a saved posterior
def get_param_values(model_name, model_type='small'):
    if model_type == 'small':
        net = BayesianClassifierSmall()
    elif model_type == 'normal':
        net = BayesianClassifier()
    elif model_type == 'large':
        net = BayesianClassifierLarge()
    else:
        raise NotImplementedError
    param_mus = []
    param_sigmas = []
    for name, _ in net.named_parameters():
        mu = pyro.param(model_name + name + '_mu').detach().numpy()
        sigma = pyro.param(model_name + name + '_sigma').detach().numpy()
        mu = np.squeeze(mu)
        sigma = np.squeeze(sigma)
        if len(mu.shape) == 0:  # bias param, add 1 dimension
            mu = np.array([mu])
            sigma = np.array([sigma])
        param_mus += [mu]
        param_sigmas += [sigma]
        param_sigmas += [sigma]
    param_mus = np.concatenate(param_mus)
    param_sigmas = np.concatenate(param_sigmas)
    param_mus = np.array(param_mus).flatten()
    param_sigmas = np.array(param_sigmas).flatten()
    return param_mus, param_sigmas


# Delete a posterior's parameters in Pyro buffer
def delete_posterior(model_name, model_type='small'):
    if model_type == 'small':
        net = BayesianClassifierSmall()
    elif model_type == 'normal':
        net = BayesianClassifier()
    elif model_type == 'large':
        net = BayesianClassifierLarge()
    else:
        raise NotImplementedError
    for name, _ in net.named_parameters():
        del pyro.get_param_store()[model_name + name + '_mu']
        del pyro.get_param_store()[model_name + name + '_sigma']
    return


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
    models = [general_model(nets[j], task_ind, priors[j]) for j in range(num_extremes)]
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
        posterior = get_posterior_by_name(model_name, model_type=model_type)
        posteriors += [posterior]

    return posteriors, running_losses, val_accs


def plot_buffer_growth(buffer_growth: list):
    xticks = np.arange(len(buffer_growth))
    plt.plot(xticks, buffer_growth, color='orange', label='sublinear')
    plt.plot(xticks, xticks * 3, color='blue', label='linear')
    plt.legend()
    plt.title('Linear vs sublinear buffer growth')
    plt.xlabel('Task num')
    plt.ylabel('Model num')
    plt.show()
    return


# Main loop, sublinear version
def fgcs_update_main_sublinear(data_dir, tasks, prior_stds, model_type='small', discard_threshold=1e-2, lr=1e-3, epochs=10, verbose=True, task_name='cifar100'):
    priors = [first_prior(std, model_type=model_type) for std in prior_stds]
    num_extremes = len(priors)
    loss_logs = []
    val_acc_logs = []

    buffered_models = []  # store the indices of buffered posterior models, i.e., (task_ind, extremum_ind)
    dict_reuse_map = {}  # discarded posterior : reused posterior
    buffer_growth = [0]  # track buffer growth

    for i in tasks:
        if task_name == 'cifar10':
            task_train_loader, task_val_loader = get_splitcifar10_loaders(data_dir, i)
        elif task_name == 'celeba':
            task_train_loader, task_val_loader = get_celeba_loaders(data_dir, i)
        elif task_name == 'cifar100':
            task_train_loader, task_val_loader = get_splitcifar100_loaders(data_dir, i)
        elif task_name == 'tinyimagenet':
            task_train_loader, task_val_loader = get_tinyimagenet_loaders(data_dir, i)
        elif task_name == '20newsgroup':
            task_train_loader, task_val_loader = get_20newsgroup_loaders(data_dir, i)
        else:
            raise NotImplementedError
        posteriors, running_losses, val_accs = fgcs_update(task_train_loader, task_val_loader, priors, model_type=model_type, task_ind=i, num_extremes=num_extremes, lr=lr, epochs=epochs, verbose=verbose)

        # Check similarity of current posteriors with the ones before, discard if exists similar ones
        for j in range(len(posteriors)):
            param_mus_j, param_sigmas_j = get_param_values(model_name=f'{i}_{j}_', model_type=model_type)

            distances = {}
            for k in range(len(buffered_models)):
                buffered_model_name = buffered_models[k]
                print(f'Comparing {i}, {j} to {buffered_model_name} ...')
                param_mus_k, param_sigmas_k = get_param_values(model_name=buffered_model_name, model_type=model_type)
                dist = np.linalg.norm(param_mus_j - param_mus_k) / len(param_mus_j) + np.linalg.norm(param_sigmas_j - param_sigmas_k) / len(param_sigmas_j)
                distances[buffered_model_name] = dist

            # No model is buffered yet, continue
            if len(distances.items()) == 0:
                print(f'No posterior buffered yet, buffering posterior {i}, {j} ...')
                buffered_models += [f'{i}_{j}_']
                continue

            # Get the previous model that has the min distance to the current model
            min_posterior_name, min_dist = min(distances.items(), key=lambda x: x[1])
            if min_dist <= discard_threshold:
                # We can use min_posterior to represent this posterior, discard this posterior
                print(f'Within threshold, discarding posterior {i}, {j} ...')
                delete_posterior(model_name=f'{i}_{j}_', model_type=model_type)
                dict_reuse_map[f'{i}_{j}_'] = min_posterior_name
            else:
                # Cache the new posterior in buffer
                print(f'Beyond threshold, buffering posterior {i}, {j} ...')
                buffered_models += [f'{i}_{j}_']

        buffer_growth += [len(buffered_models)]

        # If current posterior is replaced by a previous one, use that one as new prior
        priors = []
        for j in range(len(posteriors)):
            if f'{i}_{j}_' in dict_reuse_map:
                reused_model_name = dict_reuse_map[f'{i}_{j}_']
                new_prior = get_posterior_by_name(model_name=reused_model_name, model_type=model_type)
            else:
                new_prior = posteriors[j]
            priors += [new_prior]

        loss_logs += [running_losses]
        val_acc_logs += [val_accs]
        print(f'Running losses: {running_losses}')
        print(f'Running val accs: {val_accs}')

        # Save info for every task
        pyro.get_param_store().save(os.path.join(data_dir, f'fgcs_sublinear_{discard_threshold}.pth'))
        np.save(os.path.join(data_dir, f'loss_logs_sublinear_{discard_threshold}.npy'), loss_logs)
        np.save(os.path.join(data_dir, f'val_acc_logs_sublinear_{discard_threshold}.npy'), val_acc_logs)
        torch.save(dict_reuse_map, os.path.join(data_dir, f'dict_reuse_map_{discard_threshold}.pt'))

    torch.save(buffer_growth, os.path.join(data_dir, f'sublinear_buffer_growth_{discard_threshold}.pt'))
    plot_buffer_growth(buffer_growth)

    return loss_logs, val_acc_logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", help="valid task name", default='cifar100')
    parser.add_argument("--data_dir", help="directory to the preprocessed data", default=os.path.join('data', 'cifar100_proc_data'))
    parser.add_argument("--model_size", help="small, normal or large", default='small')
    parser.add_argument("--discard_threshold", help="threshold on posterior param distances to discard new posteriors for sublinear buffer growth", default=0.01)

    args = parser.parse_args()

    if args.task_name == 'cifar10':
        tasks = range(5)
        lr, epochs = 1e-3, 10
        prior_stds = [0.2, 0.25, 0.3]
    elif args.task_name == 'celeba':
        tasks = range(15)
        lr, epochs = 1e-3, 10
        prior_stds = [0.2, 0.25, 0.3]
    elif args.task_name == 'cifar100':
        tasks = range(10)
        lr, epochs = 5e-4, 50
        prior_stds = [2.0, 2.5, 3.0]
    elif args.task_name == 'tinyimagenet':
        tasks = range(10)
        lr, epochs = 5e-4, 30
        prior_stds = [2.0, 2.5, 3.0]
    elif args.task_name == '20newsgroup':
        tasks = range(5)
        lr, epochs = 5e-4, 100
        prior_stds = [2.0, 2.5, 3.0]
    else:
        raise NotImplementedError

    # An example training call, model type can be 'large', 'normal' or 'small'
    # Result FGCS will be stored in fgcs.pth
    _, _ = fgcs_update_main_sublinear(args.data_dir, prior_stds=prior_stds, tasks=tasks, model_type=args.model_size, discard_threshold=float(args.discard_threshold), lr=lr, epochs=epochs, task_name=args.task_name)