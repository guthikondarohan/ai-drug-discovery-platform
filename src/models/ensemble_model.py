"""
Ensemble model combining multiple architectures.

Combines predictions from MLP, GNN, and Transformer models
with learnable weights for robust predictions.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models_fixed import SimpleTabularClassifier


class EnsembleModel(nn.Module):
    """
    Ensemble combining multiple model architectures.
    
    Supports:
    - MLP (SimpleTabularClassifier)
    - GNN (MolecularGNN) - if available
    - Transformer (MolecularTransformer) - if available
    
    Args:
        models: Dictionary of models to ensemble
        combination_method: 'average', 'weighted', or 'learned' (default: 'learned')
        output_dim: Output dimension (default: 1)
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        combination_method: str = 'learned',
        output_dim: int = 1
    ):
        super().__init__()
        
        self.models = nn.ModuleDict(models)
        self.combination_method = combination_method
        self.num_models = len(models)
        
        if combination_method == 'learned':
            # Learnable weights for each model
            self.combination_weights = nn.Parameter(
                torch.ones(self.num_models) / self.num_models
            )
        elif combination_method == 'weighted':
            # Initialize with uniform weights
            self.register_buffer(
                'combination_weights',
                torch.ones(self.num_models) / self.num_models
            )
        # For 'average', no weights needed

    
    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            inputs: Dictionary with inputs for each model type:
                - 'tabular': Features for MLP [batch_size, feature_dim]
                - 'graph': Dict with 'node_features', 'edge_index' for GNN
                - 'sequence': Token indices for Transformer [batch_size, seq_len]
        
        Returns:
            Ensemble predictions [batch_size, output_dim]
        """
        predictions = []
        model_names = list(self.models.keys())
        
        # Get predictions from each model
        for name, model in self.models.items():
            if name == 'mlp' and 'tabular' in inputs:
                pred = model(inputs['tabular'])
            elif name == 'gnn' and 'graph' in inputs:
                graph_data = inputs['graph']
                pred = model(
                    graph_data['node_features'],
                    graph_data['edge_index'],
                    graph_data.get('batch_idx')
                )
            elif name == 'transformer' and 'sequence' in inputs:
                pred = model(inputs['sequence'])
            else:
                # Model input not provided, skip
                continue
            
            predictions.append(pred)
        
        if not predictions:
            raise ValueError("No valid model inputs provided to ensemble")
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [num_models, batch_size, output_dim]
        
        # Combine predictions
        if self.combination_method == 'average':
            output = torch.mean(predictions, dim=0)
        elif self.combination_method in ['weighted', 'learned']:
            # Apply softmax to ensure weights sum to 1
            weights = torch.softmax(self.combination_weights[:len(predictions)], dim=0)
            weights = weights.view(-1, 1, 1)  # [num_models, 1, 1]
            output = torch.sum(predictions * weights, dim=0)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        return output
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get the current ensemble weights for each model."""
        if self.combination_method in ['weighted', 'learned']:
            weights = torch.softmax(self.combination_weights, dim=0)
            return {
                name: weight.item() 
                for name, weight in zip(self.models.keys(), weights)
            }
        else:
            return {name: 1.0 / self.num_models for name in self.models.keys()}
    
    def predict_with_uncertainty(
        self, 
        inputs: Dict[str, Any],
        num_samples: int = 10
    ) -> tuple:
        """
        Predict with uncertainty quantification using Monte Carlo dropout.
        
        Args:
            inputs: Input dictionary for models
            num_samples: Number of forward passes with dropout
        
        Returns:
            mean_prediction: Mean prediction across samples
            std_prediction: Standard deviation (uncertainty measure)
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(inputs)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        self.eval()  # Disable dropout
        
        return mean_pred, std_pred


def create_ensemble(
    input_dim: int = 13,
    hidden_dim: int = 128,
    atom_feature_dim: int = 75,
    vocab_size: int = 100,
    use_gnn: bool = False,
    use_transformer: bool = False
) -> EnsembleModel:
    """
    Factory function to create an ensemble model.
    
    Args:
        input_dim: Dimension for MLP input features
        hidden_dim: Hidden dimension for all models
        atom_feature_dim: Atom feature dimension for GNN
        vocab_size: Vocabulary size for Transformer
        use_gnn: Whether to include GNN in ensemble
        use_transformer: Whether to include Transformer in ensemble
    
    Returns:
        EnsembleModel instance
    """
    models = {}
    
    # Always include MLP
    models['mlp'] = SimpleTabularClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1
    )
    
    # Optionally include GNN
    if use_gnn:
        try:
            from .gnn_model import MolecularGNN
            models['gnn'] = MolecularGNN(
                atom_feature_dim=atom_feature_dim,
                hidden_dim=hidden_dim,
                output_dim=1
            )
        except ImportError:
            print("Warning: Could not import MolecularGNN")
    
    # Optionally include Transformer
    if use_transformer:
        try:
            from .transformer_model import MolecularTransformer
            models['transformer'] = MolecularTransformer(
                vocab_size=vocab_size,
                d_model=hidden_dim,
                output_dim=1
            )
        except ImportError:
            print("Warning: Could not import MolecularTransformer")
    
    return EnsembleModel(models=models, combination_method='learned')
