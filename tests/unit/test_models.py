"""
Unit tests for model architectures.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from models_fixed import SimpleTabularClassifier, SimpleMultimodalClassifier


class TestSimpleTabularClassifier:
    """Tests for SimpleTabularClassifier."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = SimpleTabularClassifier(input_dim=13, hidden_dim=64, output_dim=1)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with valid input."""
        model = SimpleTabularClassifier(input_dim=13, hidden_dim=64, output_dim=1)
        model.eval()
        
        x = torch.randn(8, 13)  # Batch of 8 samples
        output = model(x)
        
        assert output.shape == (8, 1)
        assert not torch.isnan(output).any()
    
    def test_forward_pass_single_sample(self):
        """Test forward pass with single sample."""
        model = SimpleTabularClassifier(input_dim=13, hidden_dim=64)
        model.eval()
        
        x = torch.randn(1, 13)
        output = model(x)
        
        assert output.shape == (1, 1)
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        model = SimpleTabularClassifier(input_dim=13, hidden_dim=64)
        model.train()
        
        x = torch.randn(4, 13, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_batch_normalization(self):
        """Test model with batch normalization."""
        model = SimpleTabularClassifier(
            input_dim=13, 
            hidden_dim=64,
            use_batch_norm=True
        )
        model.eval()
        
        x = torch.randn(16, 13)
        output = model(x)
        
        assert output.shape == (16, 1)
    
    def test_different_hidden_dims(self):
        """Test model with different hidden dimensions."""
        for hidden_dim in [32, 64, 128, 256]:
            model = SimpleTabularClassifier(input_dim=13, hidden_dim=hidden_dim)
            x = torch.randn(4, 13)
            output = model(x)
            assert output.shape == (4, 1)


class TestSimpleMultimodalClassifier:
    """Tests for SimpleMultimodalClassifier."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = SimpleMultimodalClassifier(mol_dim=13, txt_dim=384, hidden_dim=128)
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass with both inputs."""
        model = SimpleMultimodalClassifier(mol_dim=13, txt_dim=384, hidden_dim=128)
        model.eval()
        
        mol_x = torch.randn(8, 13)
        txt_x = torch.randn(8, 384)
        output = model(mol_x, txt_x)
        
        assert output.shape == (8, 1)
        assert not torch.isnan(output).any()
    
    def test_multimodal_fusion(self):
        """Test that both modalities contribute to output."""
        model = SimpleMultimodalClassifier(mol_dim=13, txt_dim=384)
        model.eval()
        
        mol_x1 = torch.randn(1, 13)
        txt_x1 = torch.randn(1, 384)
        output1 = model(mol_x1, txt_x1)
        
        # Change molecular input
        mol_x2 = torch.randn(1, 13)
        output2 = model(mol_x2, txt_x1)
        
        # Outputs should be different
        assert not torch.allclose(output1, output2)


class TestModelSerialization:
    """Tests for model saving and loading."""
    
    def test_save_load_model(self, tmp_path):
        """Test saving and loading model state."""
        model1 = SimpleTabularClassifier(input_dim=13, hidden_dim=64)
        
        # Save model
        save_path = tmp_path / "model.pt"
        torch.save(model1.state_dict(), save_path)
        
        # Load model
        model2 = SimpleTabularClassifier(input_dim=13, hidden_dim=64)
        model2.load_state_dict(torch.load(save_path))
        
        # Test that loaded model produces same output
        model1.eval()
        model2.eval()
        
        x = torch.randn(4, 13)
        output1 = model1(x)
        output2 = model2(x)
        
        assert torch.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
