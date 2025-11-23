import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Simple Tabular Classifier
# -------------------------------

class SimpleTabularClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, out_dim: int = 1):
        super(SimpleTabularClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


# -------------------------------
# Optional: Regression Model
# -------------------------------

class SimpleRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super(SimpleRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
