import argparse
from src import data_preprocessing as dp
from src import embeddings as emb
import argparse
from pathlib import Path
import numpy as np

from src import data_preprocessing as dp
from src import embeddings as emb
from src import models_fixed as models
from src import train as trainer
from src import evaluate_fixed as evaluate
from src.utils import set_seed, ensure_dir


def run_preprocess(molecules_path: str = None, text_path: str = None):
	"""Run preprocessing for molecules and clinical text and save processed files.

	If a path is not provided, fall back to sample CSVs if present.
	"""
	print('Preprocessing molecules...')
	mol_path = molecules_path or ('data/molecules.csv' if Path('data/molecules.csv').exists() else 'data/molecules_sample.csv')
	mol = dp.preprocess_molecules(mol_path)
	Path('data/processed').mkdir(parents=True, exist_ok=True)
	dp.split_and_save(mol, 'data/processed', key='molecules')

	print('Preprocessing texts...')
	txt_path = text_path or ('data/clinical_text.csv' if Path('data/clinical_text.csv').exists() else 'data/clinical_text_sample.csv')
	txt = dp.preprocess_text(txt_path)
	txt.to_csv('data/processed/clinical_text.csv', index=False)


def run_training(epochs: int = 5, lr: float = 1e-3):
	# Simple example using tabular features created earlier
	import pandas as pd
	proc_dir = Path('data/processed')
	train_file = proc_dir / 'molecules_train.csv'
	test_file = proc_dir / 'molecules_test.csv'
	if not train_file.exists() or not test_file.exists():
		raise FileNotFoundError('Processed molecule train/test files not found. Run preprocessing first.')

	train = pd.read_csv(train_file)
	test = pd.read_csv(test_file)

	# Select numeric feature columns automatically (exclude id/label)
	exclude = {'id', 'label'}
	feature_cols = [c for c in train.columns if c not in exclude and pd.api.types.is_numeric_dtype(train[c])]
	if not feature_cols:
		raise ValueError('No numeric feature columns found for training.')

	X_train = train[feature_cols].fillna(0).values
	y_train = train['label'].values
	X_test = test[feature_cols].fillna(0).values
	y_test = test['label'].values

	model = models.SimpleTabularClassifier(input_dim=X_train.shape[1])
	trainer.train_tabular(model, X_train, y_train, X_test, y_test, epochs=epochs, lr=lr)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--stage', type=str, default='all', choices=['preprocess', 'training', 'evaluate', 'all'])
	parser.add_argument('--molecules', type=str, default=None)
	parser.add_argument('--text', type=str, default=None)
	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--lr', type=float, default=1e-3)
	args = parser.parse_args()

	set_seed(42)
	ensure_dir('results')
	if args.stage in ('preprocess', 'all'):
		run_preprocess(molecules_path=args.molecules, text_path=args.text)
	if args.stage in ('training', 'all'):
		run_training(epochs=args.epochs, lr=args.lr)
	if args.stage in ('evaluate', 'all'):
		# simple evaluate on processed test set using trained model
		import pandas as pd
		from pathlib import Path
		import torch
		proc_dir = Path('data/processed')
		test_file = proc_dir / 'molecules_test.csv'
		if not test_file.exists():
			raise FileNotFoundError('Processed test file not found. Run preprocessing and training first.')
		test = pd.read_csv(test_file)
		exclude = {'id', 'label'}
		feature_cols = [c for c in test.columns if c not in exclude and pd.api.types.is_numeric_dtype(test[c])]
		X_test = test[feature_cols].fillna(0).values
		y_test = test['label'].values
		# Reconstruct model and load weights
		model = models.SimpleTabularClassifier(input_dim=X_test.shape[1])
		state_path = Path('results') / 'model.pt'
		if not state_path.exists():
			raise FileNotFoundError('Saved model not found at results/model.pt')
		model.load_state_dict(torch.load(state_path))
		probs = torch.sigmoid(model(torch.from_numpy(X_test.astype('float32')))).detach().cpu().numpy().ravel()
		evaluate.plot_roc(y_test, probs)