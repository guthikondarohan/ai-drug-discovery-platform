import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import os




class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




def train_tabular(model, X_train, y_train, X_val, y_val, epochs=10, lr=1e-3, batch_size=32, device='cpu'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(X_val.astype(np.float32)).to(device))
            val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()
            auc = roc_auc_score(y_val, val_probs)
        print(f"Epoch {epoch+1}/{epochs} - train_loss={np.mean(losses):.4f} val_auc={auc:.4f}")

    # Save
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), 'results/model.pt')