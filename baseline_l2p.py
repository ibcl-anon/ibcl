import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoModel
import os
from utils.dataloader_utils import *
from sklearn.metrics import accuracy_score
import argparse


class L2P(nn.Module):

    def __init__(self, prompt_length=32, lr=1e-3, epochs=5):
        super(L2P, self).__init__()
        self.trainable_vectors = []  # cache trainable prompts for each task
        self.transformer = AutoModel.from_pretrained('bert-base-uncased')
        for param in self.transformer.parameters():  # freeze the transformer
            param.requires_grad = False
        self.classifier = nn.Linear(self.transformer.config.hidden_size, 1)
        self.prompt_length = prompt_length
        self.lr = lr
        self.epcohs = epochs
        self.current_task = 0
        self.loss = torch.nn.BCELoss()

    def forward(self, x, trainable_vector):
        combined_input = torch.cat((trainable_vector, x), dim=1)
        transformer_output = self.transformer(combined_input)
        logits = self.classifier(transformer_output.pooler_output)
        return torch.sigmoid(logits)

    # Train on a task's dataloader
    def learn(self, dataloader):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        optimizer.zero_grad()
        trainable_vector = nn.Parameter(torch.randn(self.prompt_length), requires_grad=True)
        for i in range(self.epochs):
            print(f'Training on task {self.current_task}, epoch {i} ...')
            for data, targets in dataloader:
                predictions = self(data, trainable_vector)
                loss = self.loss(targets.float(), predictions)
                loss.backward()
                optimizer.step()
        trainable_vector.requires_grad = False
        self.trainable_vectors += [trainable_vector]
        return


# Main L2P
def l2p_main(data_dir,  task_name='cifar10'):

    torch.manual_seed(42)

    if task_name == 'cifar10':
        num_tasks = 5
        lr, epochs = 1e-3, 10
        memory_size_per_task = 200
    elif task_name == 'celeba':
        num_tasks = 15
        lr, epochs = 1e-3, 10
        memory_size_per_task = 200
    elif task_name == 'cifar100':
        num_tasks = 10
        lr, epochs = 5e-4, 50
        memory_size_per_task = 20
    elif task_name == 'tinyimagenet':
        num_tasks = 10
        lr, epochs = 5e-4, 30
        memory_size_per_task = 20
    elif task_name == '20newsgroup':
        num_tasks = 5
        lr, epochs = 5e-4, 100
        memory_size_per_task = 20
    else:
        raise NotImplementedError

    l2p = L2P(lr=lr, epochs=epochs)
    dict_all_accs = {}
    dict_prefs = torch.load(os.path.join(data_dir, 'dict_prefs.pt'))

    for i in range(num_tasks):

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

        # Train a single model to address all prefs
        l2p.train()
        l2p.learn(task_train_loader)

        prefs = dict_prefs[i]
        task_accs = []

        # Testing
        l2p.eval()

        for pref in prefs:
            accs = []

            for j in range(i + 1):
                if task_name == 'cifar10':
                    task_test_data, task_test_labels = get_splitcifar10_test_data(data_dir, j)
                elif task_name == 'celeba':
                    task_test_data, task_test_labels = get_celeba_test_data(data_dir, j)
                elif task_name == 'cifar100':
                    task_test_data, task_test_labels = get_splitcifar100_test_data(data_dir, j)
                elif task_name == 'tinyimagenet':
                    task_test_data, task_test_labels = get_tinyimagenet_test_data(data_dir, j)
                elif task_name == '20newsgroup':
                    task_test_data, task_test_labels = get_20newsgroup_test_data(data_dir, j)
                else:
                    raise NotImplementedError

                task_test_data = torch.Tensor(task_test_data)
                task_test_labels = torch.Tensor(task_test_labels)

                with torch.no_grad():
                    # Covex combination of prompts by preference
                    pref_prompt = torch.dot(pref, torch.FloatTensor(l2p.trainable_vectors))
                    outputs = l2p(task_test_data, pref_prompt)

                pred = (outputs >= 0.5).long().numpy()
                label = np.array(task_test_labels)
                acc = accuracy_score(label, pred)
                accs += [acc]
            task_accs += [acc]

        print(f'task accs: {task_accs}')
        dict_all_accs[i] = task_accs
        torch.save(dict_all_accs, os.path.join(data_dir, 'dict_all_accs_l2p.pt'))

        # Update task number
        l2p.current_task += 1

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", help="celeba, cifar10, cifar100", default='cifar10')
    parser.add_argument("--data_dir", help="directory to the preprocessed data",
                        default=os.path.join('data', 'cifar-10-features'))
    args = parser.parse_args()

    l2p_main(args.data_dir, task_name=args.task_name)