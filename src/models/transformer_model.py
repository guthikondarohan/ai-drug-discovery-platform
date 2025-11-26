"""
Transformer-based molecular encoder for SMILES sequences.

Implements a character-level or token-level transformer that processes
SMILES notation as a sequence.
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional


class MolecularTransformer(nn.Module):
    """
    Transformer encoder for molecular SMILES sequences.
    
    Args:
        vocab_size: Size of vocabulary (default: 100 for common SMILES chars)
        d_model: Model dimension (default: 256)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer blocks (default: 6)
        dim_feedforward: Feedforward network  dimension (default: 1024)
        max_seq_length: Maximum sequence length (default: 512)
        output_dim: Output dimension (default: 1)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        vocab_size: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        max_seq_length: int = 512,
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_mask=None):
        """
        Forward pass through the transformer.
        
        Args:
            src: Input token indices [batch_size, seq_length]
            src_mask: Attention mask [seq_length, seq_length]
        
        Returns:
            Predictions [batch_size, output_dim]
        """
        # Embedding with scaling
        src = self.embedding(src) * math.sqrt(self.d_model)
        
        # Positional encoding
        src = self.pos_encoder(src)
        
        # Transformer encoding
        output = self.transformer_encoder(src, src_mask)
        
        # Global average pooling
        pooled = torch.mean(output, dim=1)
        
        # Final prediction
        return self.output_head(pooled)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    
    Injects information about the position of tokens in the sequence.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SMILESTokenizer:
    """
    Tokenizer for SMILES notation.
    
    Converts SMILES strings to token indices and back.
    """
    
    def __init__(self):
        # Common SMILES characters
        chars = [
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
            '(', ')', '[', ']', '=', '#', '@', '+', '-',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '%',
            'c', 'n', 'o', 's', 'p',  # Aromatic
            '/', '\\', '.'  # Stereochemistry and disconnected
        ]
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        
        # Build vocabulary
        special_tokens = [self.pad_token, self.unk_token, self.start_token, self.end_token]
        self.vocab = special_tokens + chars
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        self.vocab_size = len(self.vocab)
        self.pad_idx = self.char_to_idx[self.pad_token]
        self.unk_idx = self.char_to_idx[self.unk_token]
    
    def encode(self, smiles: str, max_length: Optional[int] = None) -> List[int]:
        """
        Convert SMILES string to token indices.
        
        Args:
            smiles: SMILES notation string
            max_length: Maximum sequence length (will pad/truncate)
        
        Returns:
            List of token indices
        """
        # Tokenize character by character
        # Handle two-character tokens like 'Cl', 'Br'
        tokens = []
        i = 0
        while i < len(smiles):
            # Check for two-character tokens
            if i < len(smiles) - 1:
                two_char = smiles[i:i+2]
                if two_char in self.char_to_idx:
                    tokens.append(self.char_to_idx[two_char])
                    i += 2
                    continue
            
            # Single character
            char = smiles[i]
            if char in self.char_to_idx:
                tokens.append(self.char_to_idx[char])
            else:
                tokens.append(self.unk_idx)
            i += 1
        
        # Add start and end tokens
        tokens = [self.char_to_idx[self.start_token]] + tokens + [self.char_to_idx[self.end_token]]
        
        # Pad or truncate
        if max_length is not None:
            if len(tokens) < max_length:
                tokens = tokens + [self.pad_idx] * (max_length - len(tokens))
            else:
                tokens = tokens[:max_length]
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token indices back to SMILES string.
        
        Args:
            token_ids: List of token indices
        
        Returns:
            SMILES string
        """
        chars = []
        for idx in token_ids:
            if idx == self.pad_idx:
                break
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                if char not in [self.start_token, self.end_token]:
                    chars.append(char)
        
        return ''.join(chars)
    
    def batch_encode(
        self, 
        smiles_list: List[str], 
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode a batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            max_length: Maximum sequence length
        
        Returns:
            Tensor of token indices [batch_size, max_length]
        """
        # Determine max length if not provided
        if max_length is None:
            max_length = max(len(s) for s in smiles_list) + 2  # +2 for start/end
        
        encoded = [self.encode(smiles, max_length) for smiles in smiles_list]
        return torch.tensor(encoded, dtype=torch.long)
