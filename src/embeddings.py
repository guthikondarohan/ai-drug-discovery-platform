import numpy as np
from sentence_transformers import SentenceTransformer




# Simple RDKit Morgan fingerprint to vector


def smiles_to_fp_vector(smiles: str, n_bits: int = 2048):
    import logging
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        RDKit_AVAILABLE = True
    except Exception:
        RDKit_AVAILABLE = False
        logging.getLogger(__name__).warning('RDKit not available — fingerprinting will return zeros')

    # Simple RDKit Morgan fingerprint to vector (fallback returns zeros if RDKit absent)
    if not RDKit_AVAILABLE:
        return np.zeros(n_bits, dtype=np.int8)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr




# Text embeddings using sentence-transformers
_text_model = None


def get_text_model(name: str = 'all-MiniLM-L6-v2'):
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(name)
    return _text_model


def text_to_embedding(text: str, model_name: str = 'all-MiniLM-L6-v2'):
    model = get_text_model(model_name)
    emb = model.encode(text, show_progress_bar=False)
    return emb