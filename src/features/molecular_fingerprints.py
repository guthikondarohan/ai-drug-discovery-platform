"""
Advanced molecular fingerprints for enhanced feature representation.

Provides multiple fingerprinting methods:
- Morgan (circular) fingerprints
- MACCS structural keys
- Topological fingerprints
- Atom pair fingerprints
"""

import numpy as np
from typing import Optional, Dict, List

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys, Descriptors
    from rdkit.Chem.AtomPairs import Pairs
    from rdkit.Chem.Fingerprints import FingerprintMols
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def get_morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    use_features: bool = False
) -> Optional[np.ndarray]:
    """
    Generate Morgan (circular) fingerprint.
    
    Morgan fingerprints capture the local environment around each atom.
    They are equivalent to ECFP (Extended Connectivity Fingerprints).
    
    Args:
        smiles: SMILES notation string
        radius: Radius of circular fingerprint (default: 2 for ECFP4)
        n_bits: Number of bits in fingerprint (default: 2048)
        use_features: Use pharmacophoric features instead of atom types
    
    Returns:
        Binary fingerprint array, or None if invalid SMILES
    """
    if not RDKIT_AVAILABLE:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    if use_features:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits, useFeatures=True
        )
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    
    return np.array(fp)


def get_maccs_keys(smiles: str) -> Optional[np.ndarray]:
    """
    Generate MACCS structural keys.
    
    MACCS keys are 166 predefined structural patterns that indicate
    the presence/absence of specific substructures.
    
    Args:
        smiles: SMILES notation string
    
    Returns:
        167-bit array (note: first bit is unused), or None if invalid SMILES
    """
    if not RDKIT_AVAILABLE:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)


def get_topological_fingerprint(
    smiles: str,
    n_bits: int = 2048
) -> Optional[np.ndarray]:
    """
    Generate topological (Daylight-like) fingerprint.
    
    Based on path-based hashing of molecular structure.
    
    Args:
        smiles: SMILES notation string
        n_bits: Number of bits in fingerprint
    
    Returns:
        Binary fingerprint array, or None if invalid SMILES
    """
    if not RDKIT_AVAILABLE:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fp = FingerprintMols.FingerprintMol(mol, fpSize=n_bits)
    return np.array(fp)


def get_atom_pair_fingerprint(
    smiles: str,
    n_bits: int = 2048
) -> Optional[np.ndarray]:
    """
    Generate atom pair fingerprint.
    
    Encodes pairs of atoms and the shortest path between them.
    
    Args:
        smiles: SMILES notation string
        n_bits: Number of bits in fingerprint
    
    Returns:
        Binary fingerprint array, or None if invalid SMILES
    """
    if not RDKIT_AVAILABLE:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    return np.array(fp)


def get_all_fingerprints(
    smiles: str,
    morgan_radius: int = 2,
    morgan_bits: int = 2048,
    use_maccs: bool = True,
    use_topological: bool = False,
    use_atom_pair: bool = False
) -> Optional[Dict[str, np.ndarray]]:
    """
    Get multiple fingerprint types for a molecule.
    
    Args:
        smiles: SMILES notation string
        morgan_radius: Radius for Morgan fingerprint
        morgan_bits: Bits for Morgan fingerprint
        use_maccs: Include MACCS keys
        use_topological: Include topological fingerprint
        use_atom_pair: Include atom pair fingerprint
    
    Returns:
        Dictionary with fingerprint arrays, or None if invalid
    """
    if not RDKIT_AVAILABLE:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fingerprints = {}
    
    # Morgan fingerprint
    fingerprints['morgan'] = get_morgan_fingerprint(
        smiles, radius=morgan_radius, n_bits=morgan_bits
    )
    
    # MACCS keys
    if use_maccs:
        fingerprints['maccs'] = get_maccs_keys(smiles)
    
    # Topological fingerprint
    if use_topological:
        fingerprints['topological'] = get_topological_fingerprint(smiles)
    
    # Atom pair fingerprint
    if use_atom_pair:
        fingerprints['atom_pair'] = get_atom_pair_fingerprint(smiles)
    
    return fingerprints


def concatenate_fingerprints(fingerprints: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Concatenate multiple fingerprints into a single feature vector.
    
    Args:
        fingerprints: Dictionary of fingerprint arrays
    
    Returns:
        Concatenated feature vector
    """
    fp_list = [fp for fp in fingerprints.values() if fp is not None]
    if not fp_list:
        return np.array([])
    
    return np.concatenate(fp_list)


def compute_molecular_similarity(
    smiles1: str,
    smiles2: str,
    method: str = 'tanimoto',
    fingerprint_type: str = 'morgan'
) -> Optional[float]:
    """
    Compute molecular similarity between two molecules.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        method: Similarity metric - 'tanimoto', 'dice', or 'cosine'
        fingerprint_type: Type of fingerprint to use
    
    Returns:
        Similarity score between 0 and 1, or None if error
    """
    if not RDKIT_AVAILABLE:
        return None
    
    # Get fingerprints
    if fingerprint_type == 'morgan':
        fp1 = get_morgan_fingerprint(smiles1)
        fp2 = get_morgan_fingerprint(smiles2)
    elif fingerprint_type == 'maccs':
        fp1 = get_maccs_keys(smiles1)
        fp2 = get_maccs_keys(smiles2)
    else:
        return None
    
    if fp1 is None or fp2 is None:
        return None
    
    # Compute similarity
    if method == 'tanimoto':
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        return intersection / union if union > 0 else 0.0
    
    elif method == 'dice':
        intersection = np.sum(fp1 & fp2)
        total = np.sum(fp1) + np.sum(fp2)
        return 2 * intersection / total if total > 0 else 0.0
    
    elif method == 'cosine':
        dot_product = np.sum(fp1 * fp2)
        norm1 = np.sqrt(np.sum(fp1 ** 2))
        norm2 = np.sqrt(np.sum(fp2 ** 2))
        return dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0
    
    return None


def find_similar_molecules(
    query_smiles: str,
    smiles_list: List[str],
    topk: int = 10,
    threshold: float = 0.5
) -> List[tuple]:
    """
    Find molecules similar to a query molecule.
    
    Args:
        query_smiles: Query SMILES string
        smiles_list: List of candidate SMILES strings
        topk: Number of top results to return
        threshold: Minimum similarity threshold
    
    Returns:
        List of (smiles, similarity_score) tuples, sorted by similarity
    """
    similarities = []
    
    for smiles in smiles_list:
        sim = compute_molecular_similarity(query_smiles, smiles)
        if sim is not None and sim >= threshold:
            similarities.append((smiles, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:topk]
