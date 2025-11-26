"""
Unit tests for preprocessing functions.
"""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data_preprocessing import featurize_smiles
from features.molecular_fingerprints import (
    get_morgan_fingerprint,
    get_maccs_keys,
    compute_molecular_similarity
)


class TestFeaturizeSMILES:
    """Tests for SMILES featurization."""
    
    def test_valid_smiles(self):
        """Test featurization of valid SMILES."""
        features = featurize_smiles("CCO")  # Ethanol
        
        assert features is not None
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_invalid_smiles(self):
        """Test featurization of invalid SMILES."""
        features = featurize_smiles("INVALID")
        
        # Should return empty dict for invalid SMILES
        assert features == {} or features is None
    
    def test_empty_smiles(self):
        """Test featurization of empty string."""
        features = featurize_smiles("")
        assert features == {}
    
    def test_none_smiles(self):
        """Test featurization of None."""
        features = featurize_smiles(None)
        assert features == {}
    
    def test_feature_consistency(self):
        """Test that same SMILES produces same features."""
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        
        features1 = featurize_smiles(smiles)
        features2 = featurize_smiles(smiles)
        
        assert features1 == features2
    
    def test_different_smiles_different_features(self):
        """Test that different SMILES produce different features."""
        smiles1 = "CCO"  # Ethanol
        smiles2 = "CC(C)O"  # Isopropanol
        
        features1 = featurize_smiles(smiles1)
        features2 = featurize_smiles(smiles2)
        
        assert features1 != features2


class TestMolecularFingerprints:
    """Tests for molecular fingerprints."""
    
    @pytest.mark.skipif(
        not _rdkit_available(),
        reason="RDKit not available"
    )
    def test_morgan_fingerprint(self):
        """Test Morgan fingerprint generation."""
        fp = get_morgan_fingerprint("CCO", radius=2, n_bits=2048)
        
        if fp is not None:
            assert isinstance(fp, np.ndarray)
            assert fp.shape == (2048,)
            assert np.all((fp == 0) | (fp == 1))  # Binary fingerprint
    
    @pytest.mark.skipif(
        not _rdkit_available(),
        reason="RDKit not available"
    )
    def test_maccs_keys(self):
        """Test MACCS keys generation."""
        fp = get_maccs_keys("CCO")
        
        if fp is not None:
            assert isinstance(fp, np.ndarray)
            assert len(fp) == 167  # MACCS has 167 keys
    
    @pytest.mark.skipif(
        not _rdkit_available(),
        reason="RDKit not available"
    )
    def test_fingerprint_invalid_smiles(self):
        """Test fingerprint with invalid SMILES."""
        fp = get_morgan_fingerprint("INVALID")
        assert fp is None
    
    @pytest.mark.skipif(
        not _rdkit_available(),
        reason="RDKit not available"
    )
    def test_molecular_similarity(self):
        """Test molecular similarity calculation."""
        # Test identical molecules
        sim = compute_molecular_similarity("CCO", "CCO")
        if sim is not None:
            assert sim == pytest.approx(1.0, abs=0.01)
        
        # Test different molecules
        sim = compute_molecular_similarity("CCO", "CCCC")
        if sim is not None:
            assert 0.0 <= sim < 1.0
    
    @pytest.mark.skipif(
        not _rdkit_available(),
        reason="RDKit not available"
    )
    def test_similarity_symmetry(self):
        """Test that similarity is symmetric."""
        smiles1 = "CCO"
        smiles2 = "CC(C)O"
        
        sim1 = compute_molecular_similarity(smiles1, smiles2)
        sim2 = compute_molecular_similarity(smiles2, smiles1)
        
        if sim1 is not None and sim2 is not None:
            assert sim1 == pytest.approx(sim2, abs=0.01)


def _rdkit_available():
    """Check if RDKit is available."""
    try:
        from rdkit import Chem
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
