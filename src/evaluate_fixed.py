import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import torch


def plot_roc(y_true, y_prob, out_path='results/model_performance.png'):
    """Plot and save an ROC curve to `out_path`."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(f'ROC AUC = {roc_auc:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(out_path)
    plt.close()


def predict_and_save(model, X, out_csv='results/prediction_results.csv'):
    """Predict probabilities with `model` on numpy array `X` and save to CSV."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype('float32')))
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
    np.savetxt(out_csv, probs, delimiter=',')
    return probs
