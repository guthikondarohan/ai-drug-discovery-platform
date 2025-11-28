"""
Protein sequence encoder for protein-ligand interactions.

Uses transformer-based protein language models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


class ProteinEncoder(nn.Module):
    """
    Encode protein sequences to fixed-size embeddings.
    Simple implementation using learned embeddings.
    
    For production, integrate ESM-2 or ProtBERT.
    """
    
    def __init__(
        self,
        vocab_size: int = 25,  # 20 amino acids + special tokens
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()
        
        # Amino acid vocabulary
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_idx = {aa: i+1 for i, aa in enumerate(self.amino_acids)}
        self.aa_to_idx['<PAD>'] = 0
        self.aa_to_idx['<UNK>'] = len(self.aa_to_idx)
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bi-LSTM encoder
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # Projection to fixed size
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
    
    def tokenize(self, sequence: str, max_length: int = 512) -> torch.Tensor:
        """Convert amino acid sequence to indices."""
        indices = []
        for aa in sequence.upper()[:max_length]:
            indices.append(self.aa_to_idx.get(aa, self.aa_to_idx['<UNK>']))
        
        # Pad to max_length
        indices += [0] * (max_length - len(indices))
        
        return torch.tensor(indices, dtype=torch.long)
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequences: [B, seq_len] - tokenized sequences
        
        Returns:
            embeddings: [B, embedding_dim]
        """
        # Embed
        embedded = self.embedding(sequences)  # [B, seq_len, emb_dim]
        
        # LSTM encoding
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use last hidden state (from both directions)
        # hidden: [num_layers*2, B, hidden_dim]
        forward_hidden = hidden[-2, :, :]  # Last forward layer
        backward_hidden = hidden[-1, :, :]  # Last backward layer
        
        # Concatenate
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Project to embedding space
        embeddings = self.projection(combined)
        
        return embeddings
    
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode a single protein sequence."""
        tokens = self.tokenize(sequence).unsqueeze(0)
        
        with torch.no_grad():
            embedding = self.forward(tokens)
        
        return embedding.squeeze(0).numpy()


class BindingAffinityPredictor(nn.Module):
    """
    Predict protein-ligand binding affinity.
    
    Takes:
    - Protein embedding
    - Ligand (molecule) embedding
    
    Outputs:
    - Binding score
    - Predicted IC50/Ki
    """
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        
        # Interaction layer
        self.interaction = nn.Sequential(
            nn.Linear(embedding_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Prediction heads
        self.binding_score = nn.Linear(256, 1)  # Binary binding
        self.affinity_score = nn.Linear(256, 1)  # IC50/Ki prediction
    
    def forward(
        self,
        protein_emb: torch.Tensor,
        ligand_emb: torch.Tensor
    ) -> tuple:
        """
        Args:
            protein_emb: [B, embedding_dim]
            ligand_emb: [B, embedding_dim]
        
        Returns:
            binding_score: [B, 1] - probability of binding
            affinity: [B, 1] - predicted IC50 (in log scale)
        """
        # Concatenate embeddings
        combined = torch.cat([protein_emb, ligand_emb], dim=1)
        
        # Process interaction
        features = self.interaction(combined)
        
        # Predictions
        binding = torch.sigmoid(self.binding_score(features))
        affinity = self.affinity_score(features)
        
        return binding, affinity


def get_common_protein_targets() -> dict:
    """
    Common drug targets with their sequences (shortened for demo).
    In production, load from database or UniProt.
    """
    targets = {
        'EGFR': 'MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFM',
        
        'ACE2': 'MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYP',
        
        'Spike_RBD': 'RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF'
    }
    
    return targets


class ProteinLigandInteraction:
    """
    Analyze protein-ligand interactions.
    """
    
    def __init__(self):
        self.protein_encoder = ProteinEncoder()
        self.binding_predictor = BindingAffinityPredictor()
    
    def predict_binding(
        self,
        protein_sequence: str,
        ligand_embedding: np.ndarray
    ) -> dict:
        """
        Predict if ligand binds to protein.
        
        Returns:
            {
                'binding_probability': float,
                'predicted_ic50': float,
                'confidence': float
            }
        """
        # Encode protein
        protein_emb = self.protein_encoder.encode_sequence(protein_sequence)
        protein_emb_tensor = torch.from_numpy(protein_emb).unsqueeze(0).float()
        
        # Convert ligand embedding
        ligand_emb_tensor = torch.from_numpy(ligand_embedding).unsqueeze(0).float()
        
        # Predict
        with torch.no_grad():
            binding_prob, ic50_log = self.binding_predictor(
                protein_emb_tensor,
                ligand_emb_tensor
            )
        
        # Convert IC50 from log scale
        ic50 = torch.exp(ic50_log).item()
        
        return {
            'binding_probability': binding_prob.item(),
            'predicted_ic50_nM': ic50,
            'confidence': abs(binding_prob.item() - 0.5) * 2  # Simple confidence
        }
    
    def find_best_target(
        self,
        ligand_embedding: np.ndarray,
        target_database: dict
    ) -> List[dict]:
        """
        Find best protein targets for a ligand.
        
        Args:
            ligand_embedding: Molecular embedding
            target_database: {target_name: sequence}
        
        Returns:
            List of predictions ranked by binding probability
        """
        results = []
        
        for target_name, sequence in target_database.items():
            prediction = self.predict_binding(sequence, ligand_embedding)
            prediction['target_name'] = target_name
            results.append(prediction)
        
        # Sort by binding probability
        results.sort(key=lambda x: x['binding_probability'], reverse=True)
        
        return results
