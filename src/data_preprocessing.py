import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# RDKit is optional in a pip-only install. Try to import and otherwise provide
# a lightweight fallback featurizer so preprocessing can run without RDKit.
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False
    logging.getLogger(__name__).warning("RDKit not available — molecule featurization will use a fallback")


def featurize_smiles(smiles: str):
    """Convert a SMILES string to molecular features.

    If RDKit is available this returns a comprehensive set of descriptors. Otherwise,
    returns simple fallback features derived from the SMILES string so pipelines
    that expect numeric columns can still run.
    """
    if pd.isna(smiles) or smiles == "":
        return {}
    if RDKit_AVAILABLE:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        return {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
            'RingCount': Descriptors.RingCount(mol),
            'FractionCsp3': Descriptors.FractionCSP3(mol),
            'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
        }
    # Fallback: simple SMILES-derived numeric features
    return {
        'smiles_len': len(smiles),
        'smiles_num_digits': sum(c.isdigit() for c in smiles),
        'smiles_num_letters': sum(c.isalpha() for c in smiles),
    }


def preprocess_molecules(path: str):
    """Load a molecules CSV and append featurized columns.

    Expects at least a `smiles` column. Returns a pandas DataFrame.
    """
    df = pd.read_csv(path)
    # Expect columns: id, smiles, label
    feats = df['smiles'].apply(lambda s: featurize_smiles(s) if pd.notna(s) else {})
    feat_df = pd.DataFrame(feats.tolist())
    out = pd.concat([df, feat_df], axis=1)
    return out


def preprocess_text(path: str):
    """Load clinical text CSV and perform minimal cleaning."""
    df = pd.read_csv(path)
    # Expect columns: id, notes, outcome
    df['notes'] = df['notes'].fillna("")
    return df


def split_and_save(df, out_dir: str, key: str = 'molecules'):
    """Stratified train/test split and save to `out_dir`. Returns (train, test)."""
    stratify_col = df['label'] if 'label' in df.columns else None
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=stratify_col)
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, f'{key}_train.csv'), index=False)
    test.to_csv(os.path.join(out_dir, f'{key}_test.csv'), index=False)
    return train, test