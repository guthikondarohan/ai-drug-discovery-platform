"""
Graph Neural Network for molecular property prediction.

This module implements a message-passing neural network (MPNN) that operates
on molecular graphs where atoms are nodes and bonds are edges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class MolecularGNN(nn.Module):
    """
    Graph Neural Network for molecular property prediction.
    
    Uses message passing to aggregate information from neighboring atoms.
    Implements a simplified version of Graph Attention Networks (GAT).
    
    Args:
        atom_feature_dim: Dimension of atom features (default: 75)
        hidden_dim: Hidden layer dimension (default: 128)
        num_layers: Number of message passing layers (default: 4)
        output_dim: Output dimension (default: 1)
        dropout: Dropout probability (default: 0.2)
        aggregation: Aggregation method - 'sum', 'mean', or 'max' (default: 'sum')
    """
    
    def __init__(
        self,
        atom_feature_dim: int = 75,
        hidden_dim: int = 128,
        num_layers: int = 4,
        output_dim: int = 1,
        dropout: float = 0.2,
        aggregation: str = 'sum'
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.aggregation = aggregation
        
        # Initial atom embedding
        self.atom_embedding = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Graph-level readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, node_features, edge_index, batch_idx=None):
        """
        Forward pass through the GNN.
        
        Args:
            node_features: Atom features [num_atoms, atom_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch_idx: Batch assignment for each node (for batching multiple graphs)
        
        Returns:
            Graph-level predictions [batch_size, output_dim]
        """
        # Embed atoms
        x = self.atom_embedding(node_features)
        
        # Message passing
        for layer in self.message_layers:
            x = layer(x, edge_index)
        
        # Aggregate to graph-level representation
        if batch_idx is None:
            # Single graph: aggregate all nodes
            if self.aggregation == 'sum':
                graph_repr = torch.sum(x, dim=0, keepdim=True)
            elif self.aggregation == 'mean':
                graph_repr = torch.mean(x, dim=0, keepdim=True)
            elif self.aggregation == 'max':
                graph_repr = torch.max(x, dim=0, keepdim=True)[0]
        else:
            # Multiple graphs: aggregate per batch
            num_graphs = batch_idx.max().item() + 1
            graph_repr_list = []
            
            for i in range(num_graphs):
                mask = (batch_idx == i)
                graph_nodes = x[mask]
                
                if self.aggregation == 'sum':
                    graph_repr_list.append(torch.sum(graph_nodes, dim=0))
                elif self.aggregation == 'mean':
                    graph_repr_list.append(torch.mean(graph_nodes, dim=0))
                elif self.aggregation == 'max':
                    graph_repr_list.append(torch.max(graph_nodes, dim=0)[0])
            
            graph_repr = torch.stack(graph_repr_list)
        
        # Final prediction
        return self.readout(graph_repr)


class MessagePassingLayer(nn.Module):
    """
    Single message passing layer with attention mechanism.
    """
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        
        self.message_fn = nn.Linear(in_dim, out_dim)
        self.update_fn = nn.GRUCell(out_dim, out_dim)
        self.attention = nn.Linear(in_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(out_dim)
        
    def forward(self, x, edge_index):
        """
        Perform message passing step.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge connectivity [2, num_edges]
        
        Returns:
            Updated node features [num_nodes, out_dim]
        """
        num_nodes = x.size(0)
        
        # Compute messages
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        # Attention weights
        edge_features = torch.cat([x[src_nodes], x[dst_nodes]], dim=1)
        attention_weights = torch.sigmoid(self.attention(edge_features))
        
        # Aggregate messages
        messages = self.message_fn(x[src_nodes]) * attention_weights
        aggregated = torch.zeros(num_nodes, x.size(1), device=x.device)
        
        # Sum messages for each destination node
        for i in range(edge_index.size(1)):
            dst = dst_nodes[i]
            aggregated[dst] += messages[i]
        
        # Update node features
        h_new = self.update_fn(aggregated, x)
        h_new = self.dropout(h_new)
        h_new = self.norm(h_new)
        
        return h_new


def smiles_to_graph(smiles: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Convert SMILES string to graph representation.
    
    Args:
        smiles: SMILES notation string
    
    Returns:
        Dictionary with 'node_features' and 'edge_index' tensors,
        or None if RDKit is not available or SMILES is invalid
    """
    if not RDKIT_AVAILABLE:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Extract atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = get_atom_features(atom)
        atom_features.append(features)
    
    node_features = torch.tensor(atom_features, dtype=torch.float32)
    
    # Extract bond connectivity
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Add both directions for undirected graph
        edge_index.append([i, j])
        edge_index.append([j, i])
    
    if len(edge_index) == 0:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    return {
        'node_features': node_features,
        'edge_index': edge_index
    }


def get_atom_features(atom) -> list:
    """
    Extract feature vector for an atom.
    
    Returns a 75-dimensional feature vector encoding:
    - Atom type (one-hot)
    - Degree (one-hot)
    - Formal charge
    - Hybridization (one-hot)
    - Aromaticity
    - Hydrogen count
    """
    # Atom type (common atoms: C, N, O, S, F, Cl, Br, Other)
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br']
    atom_type_encoding = [int(atom.GetSymbol() == t) for t in atom_types]
    atom_type_encoding.append(int(atom.GetSymbol() not in atom_types))  # Other
    
    # Degree (0-6)
    degree = min(atom.GetDegree(), 6)
    degree_encoding = [int(degree == i) for i in range(7)]
    
    # Formal charge
    formal_charge = atom.GetFormalCharge()
    
    # Hybridization (SP, SP2, SP3, SP3D, SP3D2, Other)
    from rdkit.Chem import HybridizationType
    hybridizations = [
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2
    ]
    hybrid_encoding = [int(atom.GetHybridization() == h) for h in hybridizations]
    hybrid_encoding.append(int(atom.GetHybridization() not in hybridizations))
    
    # Aromaticity
    aromatic = int(atom.GetIsAromatic())
    
    # Hydrogen count (0-4)
    h_count = min(atom.GetTotalNumHs(), 4)
    h_count_encoding = [int(h_count == i) for i in range(5)]
    
    # Implicit valence
    implicit_val = atom.GetImplicitValence()
    
    # In ring
    in_ring = int(atom.IsInRing())
    
    # Concatenate all features
    features = (
        atom_type_encoding +           # 8 features
        degree_encoding +               # 7 features
        [formal_charge] +               # 1 feature
        hybrid_encoding +               # 6 features
        [aromatic] +                    # 1 feature
        h_count_encoding +              # 5 features
        [implicit_val] +                # 1 feature
        [in_ring]                       # 1 feature
    )
    
    # Total: 8 + 7 + 1 + 6 + 1 + 5 + 1 + 1 = 30 features
    # Pad to 75 to match default dimension
    features = features + [0] * (75 - len(features))
    
    return features
