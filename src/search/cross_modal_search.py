"""
Cross-Modal Search System

Enable searching across different molecular representations:
- Image → Molecule
- Text → Molecule
- Molecule → Protein
"""

import numpy as np
from typing import List, Dict, Tuple
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss


class CrossModalSearch:
    """
    Search engine for cross-modal molecular retrieval.
    """
    
    def __init__(self, embedding_dim: int = 512):
        """
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.molecule_index = None
        self.protein_index = None
        self.molecule_data = []
        self.protein_data = []
        
        # Text encoder (for text queries)
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def index_molecules(
        self,
        molecules: List[Dict[str, any]],
        embeddings: np.ndarray
    ):
        """
        Index molecules with their embeddings.
        
        Args:
            molecules: List of molecule dictionaries (with name, smiles, etc.)
            embeddings: Numpy array of shape (n_molecules, embedding_dim)
        """
        self.molecule_data = molecules
        
        # Create FAISS index
        self.molecule_index = faiss.IndexFlatL2(self.embedding_dim)
        self.molecule_index.add(embeddings.astype('float32'))
    
    def index_proteins(
        self,
        proteins: List[Dict[str, any]],
        embeddings: np.ndarray
    ):
        """
        Index proteins with their embeddings.
        
        Args:
            proteins: List of protein dictionaries
            embeddings: Numpy array of shape (n_proteins, embedding_dim)
        """
        self.protein_data = proteins
        
        # Create FAISS index
        self.protein_index = faiss.IndexFlatL2(self.embedding_dim)
        self.protein_index.add(embeddings.astype('float32'))
    
    def search_molecules_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[Dict, float]]:
        """
        Search molecules using embedding vector.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            List of (molecule_dict, similarity_score) tuples
        """
        if self.molecule_index is None:
            return []
        
        # Ensure correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.molecule_index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.molecule_data):
                similarity = 1.0 / (1.0 + dist)  # Convert distance to similarity
                results.append((self.molecule_data[idx], similarity))
        
        return results
    
    def search_proteins_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[Dict, float]]:
        """
        Search proteins using embedding vector.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            List of (protein_dict, similarity_score) tuples
        """
        if self.protein_index is None:
            return []
        
        # Ensure correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.protein_index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.protein_data):
                similarity = 1.0 / (1.0 + dist)
                results.append((self.protein_data[idx], similarity))
        
        return results
    
    def search_by_text(
        self,
        text_query: str,
        top_k: int = 10
    ) -> List[Tuple[Dict, float]]:
        """
        Search molecules using text description.
        
        Args:
            text_query: Text query (e.g., "anti-inflammatory drug")
            top_k: Number of results to return
        
        Returns:
            List of (molecule_dict, similarity_score) tuples
        """
        # Encode text to embedding
        text_embedding = self.text_encoder.encode([text_query])[0]
        
        # Resize to embedding_dim if needed
        if text_embedding.shape[0] != self.embedding_dim:
            # Simple projection (can be improved with trained projection layer)
            text_embedding = text_embedding[:self.embedding_dim]
            if text_embedding.shape[0] < self.embedding_dim:
                padding = np.zeros(self.embedding_dim - text_embedding.shape[0])
                text_embedding = np.concatenate([text_embedding, padding])
        
        return self.search_molecules_by_embedding(text_embedding, top_k)
    
    def find_similar_molecules(
        self,
        smiles: str,
        molecule_encoder,
        top_k: int = 10
    ) -> List[Tuple[Dict, float]]:
        """
        Find similar molecules given a SMILES string.
        
        Args:
            smiles: Query SMILES
            molecule_encoder: Encoder to convert SMILES to embedding
            top_k: Number of results
        
        Returns:
            List of similar molecules
        """
        # Encode SMILES
        query_embedding = molecule_encoder.encode(smiles)
        
        return self.search_molecules_by_embedding(query_embedding, top_k)
    
    def find_binding_proteins(
        self,
        smiles: str,
        molecule_encoder,
        top_k: int = 10
    ) -> List[Tuple[Dict, float]]:
        """
        Find proteins that likely bind to a molecule.
        
        Args:
            smiles: Molecule SMILES
            molecule_encoder: Encoder for molecule
            top_k: Number of results
        
        Returns:
            List of potential binding proteins
        """
        # Encode molecule
        mol_embedding = molecule_encoder.encode(smiles)
        
        # Search proteins in cross-modal space
        return self.search_proteins_by_embedding(mol_embedding, top_k)


class SimpleImageEncoder:
    """
    Simple image encoder using pre-trained ResNet.
    (Placeholder - use actual trained encoder from cross_modal module)
    """
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        # In practice, use trained cross_modal.ImageEncoder
    
    def encode(self, image: Image.Image) -> np.ndarray:
        """
        Encode image to embedding vector.
        
        Args:
            image: PIL Image
        
        Returns:
            Embedding vector
        """
        # Placeholder - return random embedding
        # In production, use actual trained encoder
        return np.random.randn(self.embedding_dim).astype('float32')


class SimpleMoleculeEncoder:
    """
    Simple molecule encoder using molecular fingerprints.
    (Placeholder - use actual trained encoder)
    """
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
    
    def encode(self, smiles: str) -> np.ndarray:
        """
        Encode SMILES to embedding vector.
        
        Args:
            smiles: SMILES string
        
        Returns:
            Embedding vector
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return np.zeros(self.embedding_dim).astype('float32')
        
        # Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_array = np.zeros(2048)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        
        # Reduce to embedding_dim
        if self.embedding_dim < 2048:
            fp_array = fp_array[:self.embedding_dim]
        elif self.embedding_dim > 2048:
            padding = np.zeros(self.embedding_dim - 2048)
            fp_array = np.concatenate([fp_array, padding])
        
        return fp_array.astype('float32')


def create_search_index(
    molecules: List[Dict],
    proteins: List[Dict] = None
) -> CrossModalSearch:
    """
    Convenience function to create search index.
    
    Args:
        molecules: List of molecule dictionaries
        proteins: Optional list of protein dictionaries
    
    Returns:
        Initialized CrossModalSearch instance
    """
    search_engine = CrossModalSearch(embedding_dim=512)
    
    # Encode molecules
    encoder = SimpleMoleculeEncoder(512)
    mol_embeddings = np.array([
        encoder.encode(m['smiles'])
        for m in molecules
    ])
    
    search_engine.index_molecules(molecules, mol_embeddings)
    
    if proteins:
        # TODO: Encode proteins with ProteinEncoder
        prot_embeddings = np.random.randn(len(proteins), 512).astype('float32')
        search_engine.index_proteins(proteins, prot_embeddings)
    
    return search_engine
