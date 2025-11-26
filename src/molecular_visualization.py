"""
Molecular structure visualization utilities.

Provides functions to generate 2D molecular structure images using RDKit.
"""

from pathlib import Path
from typing import Optional
import io
import base64

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def smiles_to_image(
    smiles: str,
    width: int = 400,
    height: int = 300,
    highlight_atoms: Optional[list] = None
) -> Optional[str]:
    """
    Generate 2D molecular structure image from SMILES.
    
    Args:
        smiles: SMILES notation
        width: Image width in pixels
        height: Image height in pixels
        highlight_atoms: List of atom indices to highlight
    
    Returns:
        Base64 encoded PNG image string, or None if failed
    """
    if not RDKIT_AVAILABLE:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        
        # Draw molecule
        drawer = Draw.MolDraw2DCairo(width, height)
        
        if highlight_atoms:
            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms,
                highlightAtomColors={i: (1, 0.8, 0.8) for i in highlight_atoms}
            )
        else:
            drawer.DrawMolecule(mol)
        
        drawer.FinishDrawing()
        
        # Convert to base64
        img_bytes = drawer.GetDrawingText()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
    
    except Exception as e:
        print(f"Error generating image: {e}")
        return None


def smiles_to_svg(smiles: str, width: int = 400, height: int = 300) -> Optional[str]:
    """
    Generate SVG molecular structure from SMILES.
    
    Args:
        smiles: SMILES notation
        width: SVG width
        height: SVG height
    
    Returns:
        SVG string, or None if failed
    """
    if not RDKIT_AVAILABLE:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        AllChem.Compute2DCoords(mol)
        drawer = Draw.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg
    except Exception as e:
        print(f"Error generating SVG: {e}")
        return None


def save_molecule_image(smiles: str, output_path: str, width: int = 800, height: int = 600):
    """
    Save molecular structure as PNG file.
    
    Args:
        smiles: SMILES notation
        output_path: Path to save image
        width: Image width
        height: Image height
    """
    if not RDKIT_AVAILABLE:
        return False
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    
    try:
        AllChem.Compute2DCoords(mol)
        Draw.MolToFile(mol, output_path, size=(width, height))
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def generate_3d_conformer(smiles: str) -> Optional[str]:
    """
    Generate 3D conformer and return as PDB string for visualization.
    
    Args:
        smiles: SMILES notation
    
    Returns:
        PDB format string, or None if failed
    """
    if not RDKIT_AVAILABLE:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Convert to PDB format
        pdb_string = Chem.MolToPDBBlock(mol)
        return pdb_string
    
    except Exception as e:
        print(f"Error generating 3D conformer: {e}")
        return None
