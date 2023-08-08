import numpy as np
import torch
from torch.optim import Adam
import os
import argparse
from sklearn.metrics import accuracy_score
from models.models import BayesianClassifier, BayesianClassifierLarge, BayesianClassifierSmall
from utils.dataloader_utils import get_celeba_loaders, get_splitcifar10_loaders
from utils.dataloader_utils import get_celeba_test_data, get_splitcifar10_test_data


# Gradient episodic memory (GEM) baseline by Lopez-Paz et al. 2017
class GEM():

    def __init__(self, model, num_tasks, memory_size_per_task=5000, lr=1e-3, epochs=5):
        self.model = model
        self.num_tasks = num_tasks
        self.memory = {task: {'data': None, 'targets': None} for task in range(num_tasks)}
        self.memory_size_per_task = memory_size_per_task
        self.current_task = 0
        self.lr = lr
        self.epochs = epochs
        self.loss = torch.nn.BCELoss()

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

    def compute_loss(self, data, targets, pref):
        total_loss = 0.0
        for task in range(self.current_task + 1):
            if self.memory[task]['data'] is not None:
                old_data, old_targets = self.memory[task]['data'], self.memory[task]['targets']
                old_predictions = self.model(old_data)
                old_loss = self.loss(old_predictions, old_targets)
                total_loss += pref[task] * old_loss # Loss regularized by preferences

        new_predictions = self.model(data)
        new_loss = self.loss(new_predictions, targets)
        total_loss += pref[self.current_task] * new_loss # Loss regularized by preferences
        return total_loss

    def learn(self, dataloader, pref):
        # Training
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        optimizer.zero_grad()
        for i in range(self.epochs):
            print(f'Training on task {self.current_task}, pref {pref}, epoch {i} ...')
            for data, targets in dataloader:
                loss = self.compute_loss(data, targets.float(), pref)
                loss.backward()
                optimizer.step()
        return


# Main GEM
def gem_main(data_dir, model_type='small', task_name='cifar10', num_prefs_per_task=10):

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

    gem = GEM(net, num_tasks, memory_size_per_task=500, lr=1e-3, epochs=10)
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
        torch.save(dict_all_prefs, os.path.join(data_dir, 'dict_all_prefs_gem.pt'))

        task_accs = []
        for pref in prefs:

            # Training on a pref
            gem.model.train()
            gem.learn(task_train_loader, pref)

            # Testing
            gem.model.eval()
            accs = []
            for j in range(i + 1):
                task_test_data, task_test_labels = get_splitcifar10_test_data(data_dir, j)
                task_test_data = torch.Tensor(task_test_data)
                task_test_labels = torch.Tensor(task_test_labels)
                with torch.no_grad():
                    outputs = gem.model(task_test_data)
                    pred = (outputs >= 0.5).long().numpy()
                    acc = accuracy_score(task_test_labels, pred)
                    accs += [acc]
            task_accs += [accs]

        print(f'task accs: {task_accs}')
        dict_all_accs[i] = task_accs
        torch.save(dict_all_accs, os.path.join(data_dir, 'dict_all_accs_gem.pt'))

        # Memorize some training data
        gem.remember(task_train_loader)

        # Update task number
        gem.current_task += 1

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", help="celeba or cifar10", default='cifar10')
    parser.add_argument("--data_dir", help="directory to the preprocessed data",
                        default=os.path.join('data', 'cifar-10-features'))
    parser.add_argument("--model_size", help="small, normal or large", default='small')
    parser.add_argument("--num_prefs_per_task", help="numer of preferences per task", default=10)
    args = parser.parse_args()

    gem_main(args.data_dir, model_type=args.model_size, task_name=args.task_name, num_prefs_per_task=int(args.num_prefs_per_task))
