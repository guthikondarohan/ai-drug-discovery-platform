"""
Molecule Comparison and Analysis Tools

Compare multiple molecules side-by-side with similarity scoring,
property comparison, and visualizations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit import DataStructs
import plotly.graph_objects as go
import plotly.express as px


class MoleculeComparison:
    """
    Compare multiple molecules and analyze their properties.
    """
    
    def __init__(self):
        self.molecules = {}
    
    def add_molecule(self, name: str, smiles: str):
        """Add a molecule to the comparison."""
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            self.molecules[name] = {
                'smiles': smiles,
                'mol': mol
            }
    
    def calculate_properties(self, smiles: str) -> Dict:
        """
        Calculate molecular properties.
        
        Args:
            smiles: SMILES string
        
        Returns:
            Dictionary of properties
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {}
        
        props = {
            'Molecular Weight': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'H-Bond Donors': Lipinski.NumHDonors(mol),
            'H-Bond Acceptors': Lipinski.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'Rotatable Bonds': Lipinski.NumRotatableBonds(mol),
            'Aromatic Rings': Lipinski.NumAromaticRings(mol),
            'Heavy Atoms': Lipinski.HeavyAtomCount(mol),
            'Fraction Csp3': Lipinski.FractionCsp3(mol)
        }
        
        return props
    
    def calculate_similarity(self, smiles1: str, smiles2: str, method: str = 'tanimoto') -> float:
        """
        Calculate similarity between two molecules.
        
        Args:
            smiles1: First SMILES
            smiles2: Second SMILES
            method: Similarity method ('tanimoto', 'dice')
        
        Returns:
            Similarity score (0-1)
        """
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if not mol1 or not mol2:
            return 0.0
        
        # Morgan fingerprints
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        if method == 'tanimoto':
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        elif method == 'dice':
            return DataStructs.DiceSimilarity(fp1, fp2)
        else:
            return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    def compare_multiple(self, molecules: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Compare multiple molecules and create comparison table.
        
        Args:
            molecules: List of (name, smiles) tuples
        
        Returns:
            DataFrame with properties for all molecules
        """
        data = []
        
        for name, smiles in molecules:
            props = self.calculate_properties(smiles)
            props['Name'] = name
            props['SMILES'] = smiles
            data.append(props)
        
        df = pd.DataFrame(data)
        
        # Reorder columns to put Name and SMILES first
        cols = ['Name', 'SMILES'] + [c for c in df.columns if c not in ['Name', 'SMILES']]
        df = df[cols]
        
        return df
    
    def create_similarity_matrix(self, molecules: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create similarity matrix for multiple molecules.
        
        Args:
            molecules: List of (name, smiles) tuples
        
        Returns:
            DataFrame similarity matrix
        """
        names = [m[0] for m in molecules]
        smiles = [m[1] for m in molecules]
        
        n = len(molecules)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    sim = self.calculate_similarity(smiles[i], smiles[j])
                    similarity_matrix[i][j] = sim
        
        df = pd.DataFrame(similarity_matrix, index=names, columns=names)
        return df
    
    def create_radar_chart(self, molecules: List[Tuple[str, str]]) -> go.Figure:
        """
        Create radar chart comparing molecular properties.
        
        Args:
            molecules: List of (name, smiles) tuples
        
        Returns:
            Plotly figure
        """
        # Properties to compare (normalized)
        comparison_props = [
            'Molecular Weight',
            'LogP',
            'H-Bond Donors',
            'H-Bond Acceptors',
            'TPSA',
            'Rotatable Bonds'
        ]
        
        fig = go.Figure()
        
        for name, smiles in molecules:
            props = self.calculate_properties(smiles)
            
            # Normalize values (0-1 scale)
            values = []
            for prop in comparison_props:
                if prop in props:
                    # Simple normalization (adjust ranges as needed)
                    if prop == 'Molecular Weight':
                        normalized = min(props[prop] / 500, 1.0)
                    elif prop == 'LogP':
                        normalized = (props[prop] + 5) / 10  # LogP range ~-5 to 5
                    elif prop == 'TPSA':
                        normalized = min(props[prop] / 200, 1.0)
                    else:
                        normalized = min(props[prop] / 10, 1.0)
                    
                    values.append(normalized)
                else:
                    values.append(0)
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=comparison_props,
                fill='toself',
                name=name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Molecular Property Comparison"
        )
        
        return fig
    
    def create_heatmap(self, molecules: List[Tuple[str, str]]) -> go.Figure:
        """
        Create heatmap of molecular properties.
        
        Args:
            molecules: List of (name, smiles) tuples
        
        Returns:
            Plotly figure
        """
        df = self.compare_multiple(molecules)
        
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Normalize data
        df_norm = df[numeric_cols].copy()
        for col in numeric_cols:
            max_val = df_norm[col].max()
            if max_val > 0:
                df_norm[col] = df_norm[col] / max_val
        
        fig = px.imshow(
            df_norm.T,
            labels=dict(x="Molecule", y="Property", color="Normalized Value"),
            x=df['Name'].values,
            y=numeric_cols,
            color_continuous_scale='Viridis',
            aspect="auto"
        )
        
        fig.update_layout(
            title="Molecular Properties Heatmap",
            xaxis_title="Molecules",
            yaxis_title="Properties"
        )
        
        return fig
    
    def create_similarity_heatmap(self, molecules: List[Tuple[str, str]]) -> go.Figure:
        """
        Create heatmap of similarity scores.
        
        Args:
            molecules: List of (name, smiles) tuples
        
        Returns:
            Plotly figure
        """
        sim_matrix = self.create_similarity_matrix(molecules)
        
        fig = px.imshow(
            sim_matrix,
            labels=dict(x="Molecule", y="Molecule", color="Similarity"),
            x=sim_matrix.columns,
            y=sim_matrix.index,
            color_continuous_scale='RdYlGn',
            zmin=0,
            zmax=1,
            aspect="auto"
        )
        
        fig.update_layout(
            title="Molecular Similarity Matrix (Tanimoto)"
        )
        
        return fig


def compare_molecules(molecules: List[Tuple[str, str]]) -> Dict:
    """
    Convenience function to compare molecules.
    
    Args:
        molecules: List of (name, smiles) tuples
    
    Returns:
        Dictionary with comparison results and figures
    """
    comp = MoleculeComparison()
    
    results = {
        'properties_table': comp.compare_multiple(molecules),
        'similarity_matrix': comp.create_similarity_matrix(molecules),
        'radar_chart': comp.create_radar_chart(molecules),
        'properties_heatmap': comp.create_heatmap(molecules),
        'similarity_heatmap': comp.create_similarity_heatmap(molecules)
    }
    
    return results
