import os
import traceback
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src import models, train as trainer
    # Read preprocessed data
    df = pd.read_csv('data/processed/molecules_preprocessed.csv')
    feat_cols = ['MolWt','NumHDonors','NumHAcceptors','TPSA']
    if not all(c in df.columns for c in feat_cols):
        raise ValueError(f"Missing feature columns. Available columns: {df.columns.tolist()}")
    X = df[feat_cols].fillna(0).values
    y = df['label'].values

    # If y has only one class, avoid stratify to prevent errors
    stratify_arg = y if len(set(y)) > 1 else None

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg)

    print(f"Train size: {len(X_tr)}, Val size: {len(X_val)}")
    model = models.SimpleTabularClassifier(input_dim=X_tr.shape[1])
    trainer.train_tabular(model, X_tr, y_tr, X_val, y_val, epochs=10, lr=1e-3, batch_size=16)
    print('Training finished.')

except Exception as e:
    print('ERROR during script:')
    traceback.print_exc()
    raise
