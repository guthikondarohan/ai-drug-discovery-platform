import torch
import torch.nn as nn


class SimpleTabularClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)



class SimpleMultimodalClassifier(nn.Module):
    def __init__(self, mol_dim: int, txt_dim: int, hidden: int = 256):
        super().__init__()
        self.mol_fc = nn.Sequential(nn.Linear(mol_dim, hidden), nn.ReLU())
        self.txt_fc = nn.Sequential(nn.Linear(txt_dim, hidden), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, mol_x, txt_x):
        m = self.mol_fc(mol_x)
        t = self.txt_fc(txt_x)
        combined = torch.cat([m, t], dim=1)
        return self.classifier(combined)
import torch
import torch.nn as nn


class SimpleTabularClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimpleMultimodalClassifier(nn.Module):
    def __init__(self, mol_dim: int, txt_dim: int, hidden: int = 256):
        super().__init__()
        self.mol_fc = nn.Sequential(nn.Linear(mol_dim, hidden), nn.ReLU())
        self.txt_fc = nn.Sequential(nn.Linear(txt_dim, hidden), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, mol_x, txt_x):
        m = self.mol_fc(mol_x)
        t = self.txt_fc(txt_x)
        combined = torch.cat([m, t], dim=1)
        return self.classifier(combined)
