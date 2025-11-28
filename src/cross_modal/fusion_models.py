"""
Cross-modal fusion models.

Combines multiple modalities (SMILES, images, graphs, proteins)
for enhanced predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class MultiModalFusion(nn.Module):
    """
    Fuse embeddings from multiple modalities.
    
    Supports:
    - Early fusion (concatenation)
    - Late fusion (ensemble)
    - Attention fusion (learned weights)
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_modalities: int = 4,  # SMILES, image, graph, protein
        fusion_type: str = 'attention',
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.embedding_dim = embedding_dim
        
        if fusion_type == 'attention':
            # Attention-based fusion
            self.attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            
            self.fusion_layer = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, embedding_dim)
            )
        
        elif fusion_type == 'concat':
            # Concatenation fusion
            self.fusion_layer = nn.Sequential(
                nn.Linear(embedding_dim * num_modalities, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim)
            )
        
        elif fusion_type == 'gated':
            # Gated fusion
            self.gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.Sigmoid()
                )
                for _ in range(num_modalities)
            ])
            
            self.fusion_layer = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: Dict of modality_name -> tensor [B, embedding_dim]
        
        Returns:
            fused_embedding: [B, embedding_dim]
        """
        if self.fusion_type == 'attention':
            # Stack embeddings
            emb_list = list(embeddings.values())
            stacked = torch.stack(emb_list, dim=1)  # [B, num_modalities, emb_dim]
            
            # Self-attention
            attended, _ = self.attention(stacked, stacked, stacked)
            
            # Pool
            pooled = attended.mean(dim=1)  # [B, emb_dim]
            
            # Final projection
            fused = self.fusion_layer(pooled)
        
        elif self.fusion_type == 'concat':
            # Concatenate all embeddings
            emb_list = list(embeddings.values())
            concatenated = torch.cat(emb_list, dim=1)
            
            fused = self.fusion_layer(concatenated)
        
        elif self.fusion_type == 'gated':
            # Gated fusion
            emb_list = list(embeddings.values())
            
            # Apply gates
            gated_embs = []
            for i, emb in enumerate(emb_list):
                gate = self.gates[i](emb)
                gated_embs.append(emb * gate)
            
            # Sum gated embeddings
            summed = torch.stack(gated_embs, dim=0).sum(dim=0)
            
            fused = self.fusion_layer(summed)
        
        return fused


class CrossModalPredictor(nn.Module):
    """
    Unified predictor using cross-modal fusion.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 1,
        fusion_type: str = 'attention'
    ):
        super().__init__()
        
        # Fusion module
        self.fusion = MultiModalFusion(
            embedding_dim=embedding_dim,
            fusion_type=fusion_type
        )
        
        # Prediction head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()  # Ensure positive
        )
    
    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        return_uncertainty: bool = False
    ):
        """
        Args:
            embeddings: Dict of available modalities
            return_uncertainty: Whether to return uncertainty estimate
        
        Returns:
            prediction: [B, num_classes]
            uncertainty: [B, 1] (optional)
        """
        # Fuse modalities
        fused = self.fusion(embeddings)
        
        # Predict
        prediction = self.classifier(fused)
        
        if return_uncertainty:
            uncertainty = self.uncertainty_head(fused)
            return prediction, uncertainty
        
        return prediction


class CrossModalRetrieval:
    """
    Retrieve molecules across different modalities.
    """
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.database = {
            'smiles': [],
            'images': [],
            'embeddings': []
        }
    
    def add_to_database(
        self,
        smiles: str,
        embedding: np.ndarray,
        image_path: Optional[str] = None
    ):
        """Add molecule to database."""
        self.database['smiles'].append(smiles)
        self.database['embeddings'].append(embedding)
        self.database['images'].append(image_path)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        modality_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search database using query embedding.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            modality_filter: Filter by modality
        
        Returns:
            List of similar molecules
        """
        if not self.database['embeddings']:
            return []
        
        # Compute similarities
        db_embeddings = np.array(self.database['embeddings'])
        
        # Cosine similarity
        similarities = np.dot(db_embeddings, query_embedding) / (
            np.linalg.norm(db_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Format results
        results = []
        for idx in top_indices:
            results.append({
                'smiles': self.database['smiles'][idx],
                'similarity': float(similarities[idx]),
                'image_path': self.database['images'][idx]
            })
        
        return results


def create_unified_embedding_space(
    smiles_encoder,
    image_encoder,
    protein_encoder,
    contrastive_loss_weight: float = 0.1
) -> nn.Module:
    """
    Create unified embedding space using contrastive learning.
    
    This aligns embeddings from different modalities.
    Similar to CLIP but for molecules.
    """
    
    class UnifiedEmbeddingSpace(nn.Module):
        def __init__(self):
            super().__init__()
            self.smiles_encoder = smiles_encoder
            self.image_encoder = image_encoder
            self.protein_encoder = protein_encoder
            self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        def contrastive_loss(self, emb1, emb2):
            """InfoNCE loss."""
            # Normalize embeddings
            emb1 = F.normalize(emb1, dim=1)
            emb2 = F.normalize(emb2, dim=1)
            
            # Compute similarity
            logits = torch.matmul(emb1, emb2.T) / self.temperature
            
            # Labels (diagonal)
            labels = torch.arange(len(emb1), device=emb1.device)
            
            # Symmetric loss
            loss_1 = F.cross_entropy(logits, labels)
            loss_2 = F.cross_entropy(logits.T, labels)
            
            return (loss_1 + loss_2) / 2
        
        def forward(self, smiles=None, images=None, proteins=None):
            embeddings = {}
            
            if smiles is not None:
                embeddings['smiles'] = self.smiles_encoder(smiles)
            if images is not None:
                embeddings['images'] = self.image_encoder(images)
            if proteins is not None:
                embeddings['proteins'] = self.protein_encoder(proteins)
            
            return embeddings
    
    return UnifiedEmbeddingSpace()
