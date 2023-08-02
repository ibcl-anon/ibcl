import torch


# Define the neural network model and the guide function
class BayesianClassifier(torch.nn.Module):
    def __init__(self, input_dim=512, h1=64, output_dim=1):
        super(BayesianClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, h1)
        self.fc2 = torch.nn.Linear(h1, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze()


class BayesianClassifierLarge(torch.nn.Module):
    def __init__(self, input_dim=512, h1=128, h2=16, output_dim=1):
        super(BayesianClassifierLarge, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, h1)
        self.fc2 = torch.nn.Linear(h1, h2)
        self.fc3= torch.nn.Linear(h2, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze()


class BayesianClassifierSmall(torch.nn.Module):
    def __init__(self, input_dim=512, output_dim=1):
        super(BayesianClassifierSmall, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x.squeeze()


