import torch
import pandas as pd
import numpy as np
from src import models

# Load preprocessed data
df = pd.read_csv("data/processed/molecules_preprocessed.csv")

# Feature columns
feat_cols = ["MolWt", "NumHDonors", "NumHAcceptors", "TPSA"]
for c in feat_cols:
    if c not in df.columns:
        raise ValueError(f"Missing feature column: {c}")

X = df[feat_cols].fillna(0).values.astype("float32")

# Load trained model
model = models.SimpleTabularClassifier(input_dim=X.shape[1])
model.load_state_dict(torch.load("results/model.pt", map_location="cpu"))
model.eval()

# Predict probabilities
with torch.no_grad():
    logits = model(torch.tensor(X))
    probs = torch.sigmoid(logits).numpy().ravel()

# Save results
out_path = "results/prediction_results.csv"
pd.DataFrame({"id": df["id"], "probability": probs}).to_csv(out_path, index=False)

print("Wrote:", out_path)
print("First 5 predictions:")
print(pd.DataFrame({"id": df["id"], "probability": probs}).head())
