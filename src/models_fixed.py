"""
Neural network models for molecular activity prediction.

This module provides multiple architectures:
- SimpleTabularClassifier: Feed-forward network for tabular molecular features
- SimpleMultimodalClassifier: Combined molecular and text feature classifier
"""

import torch
import torch.nn as nn


class SimpleTabularClassifier(nn.Module):
    """
    Feed-forward neural network for molecular feature classification.
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden layer dimension (default: 128)
        output_dim: Output dimension (default: 1 for binary classification)
        dropout: Dropout probability (default: 0.3)
        use_batch_norm: Whether to use batch normalization (default: True)
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 128, 
        output_dim: int = 1,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Second layer
        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim // 2, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.net(x)


class SimpleMultimodalClassifier(nn.Module):
    """
    Multimodal classifier combining molecular and text features.
    
    Args:
        mol_dim: Dimension of molecular features
        txt_dim: Dimension of text embeddings
        hidden_dim: Hidden layer dimension (default: 256)
        dropout: Dropout probability (default: 0.3)
    """
    def __init__(
        self, 
        mol_dim: int, 
        txt_dim: int, 
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Molecular feature encoder
        self.mol_encoder = nn.Sequential(
            nn.Linear(mol_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Text feature encoder
        self.txt_encoder = nn.Sequential(
            nn.Linear(txt_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mol_x, txt_x):
        """
        Forward pass with both molecular and text features.
        
        Args:
            mol_x: Molecular features tensor
            txt_x: Text features tensor
            
        Returns:
            Classification logits
        """
        mol_features = self.mol_encoder(mol_x)
        txt_features = self.txt_encoder(txt_x)
        combined = torch.cat([mol_features, txt_features], dim=1)
        return self.classifier(combined)

