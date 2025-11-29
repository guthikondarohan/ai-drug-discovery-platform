"""
Molecular Generation using Variational Autoencoder (VAE)

Generate novel molecules with desired properties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from rdkit import Chem


class MoleculeVAE(nn.Module):
    """
    Variational Autoencoder for SMILES generation.
    
    Architecture:
    - Encoder: Character-level GRU → Latent space (μ, σ)
    - Decoder: Latent vector → GRU → SMILES characters
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        max_length: int = 100
    ):
        """
        Args:
            vocab_size: Size of character vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Hidden dimension of GRU
            latent_dim: Dimension of latent space
            max_length: Maximum SMILES length
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_length = max_length
        
        # Encoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, vocab_size)
    
    def encode(self, x):
        """
        Encode SMILES to latent representation.
        
        Args:
            x: Input tensor (batch_size, seq_length)
        
        Returns:
            mu, logvar: Latent parameters
        """
        # Embed
        embedded = self.embedding(x)
        
        # Encode
        _, hidden = self.encoder_gru(embedded)
        hidden = hidden.squeeze(0)
        
        # Latent parameters
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, target_seq=None):
        """
        Decode latent vector to SMILES.
        
        Args:
            z: Latent vector
            target_seq: Target sequence for teacher forcing (optional)
        
        Returns:
            Output logits
        """
        batch_size = z.size(0)
        
        # Initial hidden state from latent
        hidden = self.decoder_fc(z).unsqueeze(0)
        
        # Start token (0)
        input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)
        
        outputs = []
        
        for t in range(self.max_length):
            # Embed current token
            embedded = self.embedding(input_token)
            
            # Decode
            output, hidden = self.decoder_gru(embedded, hidden)
            
            # Predict next character
            logits = self.output_fc(output.squeeze(1))
            outputs.append(logits)
            
            # Teacher forcing or use prediction
            if target_seq is not None and t < target_seq.size(1) - 1:
                input_token = target_seq[:, t+1:t+2]
            else:
                input_token = logits.argmax(dim=1, keepdim=True)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input SMILES tensor
        
        Returns:
            reconstructed, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, x)
        return reconstructed, mu, logvar
    
    def sample(self, num_samples: int = 1, device='cpu'):
        """
        Sample from latent space to generate molecules.
        
        Args:
            num_samples: Number of molecules to generate
            device: Device to use
        
        Returns:
            Generated SMILES logits
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def interpolate(self, smiles1: str, smiles2: str, steps: int = 10):
        """
        Interpolate between two molecules in latent space.
        
        Args:
            smiles1: First SMILES
            smiles2: Second SMILES
            steps: Number of interpolation steps
        
        Returns:
            List of interpolated SMILES
        """
        # TODO: Implement after adding tokenizer
        pass


class SMILESTokenizer:
    """Tokenizer for SMILES strings."""
    
    def __init__(self):
        # Common SMILES characters
        self.chars = [
            ' ', 'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I',
            'c', 'n', 'o', 's',
            '(', ')', '[', ']', '=', '#', '@', '+', '-',
            '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '/', '\\', '%', '.'
        ]
        
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    
    def encode(self, smiles: str, max_length: int = 100) -> torch.Tensor:
        """
        Encode SMILES to tensor.
        
        Args:
            smiles: SMILES string
            max_length: Maximum length
        
        Returns:
            Encoded tensor
        """
        indices = [self.char_to_idx.get(c, 0) for c in smiles]
        
        # Pad or truncate
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, tensor: torch.Tensor) -> str:
        """
        Decode tensor to SMILES.
        
        Args:
            tensor: Encoded tensor
        
        Returns:
            SMILES string
        """
        if tensor.dim() > 1:
            tensor = tensor.argmax(dim=-1)
        
        chars = [self.idx_to_char.get(idx.item(), ' ') for idx in tensor]
        smiles = ''.join(chars).strip()
        
        # Remove padding
        if ' ' in smiles:
            smiles = smiles[:smiles.index(' ')]
        
        return smiles


class MoleculeGenerator:
    """High-level interface for molecular generation."""
    
    def __init__(self, model: MoleculeVAE, tokenizer: SMILESTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def generate(self, num_molecules: int = 10, device='cpu') -> List[str]:
        """
        Generate random molecules.
        
        Args:
            num_molecules: Number to generate
            device: Device to use
        
        Returns:
            List of SMILES strings
        """
        with torch.no_grad():
            logits = self.model.sample(num_molecules, device)
            
            smiles_list = []
            for i in range(num_molecules):
                smiles = self.tokenizer.decode(logits[i])
                smiles_list.append(smiles)
        
        return smiles_list
    
    def validate_smiles(self, smiles: str) -> bool:
        """Check if SMILES is valid."""
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    
    def generate_valid(self, target_count: int = 10, max_attempts: int = 100) -> List[str]:
        """
        Generate valid molecules.
        
        Args:
            target_count: Number of valid molecules wanted
            max_attempts: Maximum generation attempts
        
        Returns:
            List of valid SMILES
        """
        valid_molecules = []
        attempts = 0
        
        while len(valid_molecules) < target_count and attempts < max_attempts:
            candidates = self.generate(num_molecules=10)
            
            for smiles in candidates:
                if self.validate_smiles(smiles):
                    valid_molecules.append(smiles)
                    if len(valid_molecules) >= target_count:
                        break
            
            attempts += 10
        
        return valid_molecules[:target_count]


def create_vae_model(vocab_size: int = 40) -> Tuple[MoleculeVAE, SMILESTokenizer]:
    """
    Create VAE model and tokenizer.
    
    Args:
        vocab_size: Vocabulary size
    
    Returns:
        model, tokenizer
    """
    tokenizer = SMILESTokenizer()
    model = MoleculeVAE(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        latent_dim=128,
        max_length=100
    )
    
    return model, tokenizer
