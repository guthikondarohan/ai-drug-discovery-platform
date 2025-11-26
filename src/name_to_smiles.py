"""
Common chemical name to SMILES converter.
"""

COMMON_MOLECULES = {
    # Basic elements and simple molecules
    'carbon': 'C',
    'methane': 'C',
    'hydrogen': '[H][H]',
    'oxygen': 'O=O',
    'nitrogen': 'N#N',
    'water': 'O',
    
    # Alcohols
    'ethanol': 'CCO',
    'ethyl alcohol': 'CCO',
    'methanol': 'CO',
    'methyl alcohol': 'CO',
    'wood alcohol': 'CO',
    'isopropanol': 'CC(C)O',
    'isopropyl alcohol': 'CC(C)O',
    'rubbing alcohol': 'CC(C)O',
    'propanol': 'CCCO',
    'butanol': 'CCCCO',
    'pentanol': 'CCCCCO',
    'hexanol': 'CCCCCCO',
    'glycerol': 'OCC(O)CO',
    'glycerin': 'OCC(O)CO',
    
    # Common drugs
    'aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
    'acetylsalicylic acid': 'CC(=O)OC1=CC=CC=C1C(=O)O',
    'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    'ibuprofen': 'CC(C)Cc1ccc(cc1)C(C)C(O)=O',
    'advil': 'CC(C)Cc1ccc(cc1)C(C)C(O)=O',
    'paracetamol': 'CC(=O)Nc1ccc(O)cc1',
    'acetaminophen': 'CC(=O)Nc1ccc(O)cc1',
    'tylenol': 'CC(=O)Nc1ccc(O)cc1',
    'morphine': 'CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O',
    'codeine': 'COc1ccc2c3c1O[C@H]1[C@@H](O)C=C[C@H]4[C@@H](C2)[N@](C)CC[C@]314',
    
    # Hydrocarbons
    'ethane': 'CC',
    'propane': 'CCC',
    'butane': 'CCCC',
    'pentane': 'CCCCC',
    'hexane': 'CCCCCC',
    'heptane': 'CCCCCCC',
    'octane': 'CCCCCCCC',
    'ethylene': 'C=C',
    'propylene': 'CC=C',
    'acetylene': 'C#C',
    
    # Aromatic compounds
    'benzene': 'c1ccccc1',
    'toluene': 'Cc1ccccc1',
    'xylene': 'Cc1ccccc1C',
    'naphthalene': 'c1ccc2ccccc2c1',
    'phenol': 'Oc1ccccc1',
    'aniline': 'Nc1ccccc1',
    'styrene': 'C=Cc1ccccc1',
    
    # Ketones and aldehydes
    'acetone': 'CC(=O)C',
    'formaldehyde': 'C=O',
    'acetaldehyde': 'CC=O',
    'benzaldehyde': 'O=Cc1ccccc1',
    
    # Acids
    'acetic acid': 'CC(=O)O',
    'vinegar': 'CC(=O)O',
    'formic acid': 'C(=O)O',
    'benzoic acid': 'O=C(O)c1ccccc1',
    'citric acid': 'OC(=O)CC(O)(CC(=O)O)C(=O)O',
    'lactic acid': 'CC(O)C(=O)O',
    'oxalic acid': 'C(=O)(C(=O)O)O',
    
    # Esters
    'ethyl acetate': 'CCOC(=O)C',
    'methyl acetate': 'COC(=O)C',
    'vinyl acetate': 'C=COC(=O)C',
    
    # Amines
    'methylamine': 'CN',
    'dimethylamine': 'CNC',
    'trimethylamine': 'CN(C)C',
    'ethylamine': 'CCN',
    
    # Amino acids
    'glycine': 'NCC(=O)O',
    'alanine': 'CC(N)C(=O)O',
    'valine': 'CC(C)C(N)C(=O)O',
    'leucine': 'CC(C)CC(N)C(=O)O',
    'isoleucine': 'CCC(C)C(N)C(=O)O',
    'serine': 'OCC(N)C(=O)O',
    'threonine': 'CC(O)C(N)C(=O)O',
    
    # Sugars
    'glucose': 'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O',
    'fructose': 'OCC1(O)OC(CO)[C@@H](O)[C@@H]1O',
    'sucrose': 'OC[C@H]1O[C@H](OC2(CO)O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O',
    
    # Other common compounds
    'urea': 'NC(=O)N',
    'ammonia': 'N',
    'carbon dioxide': 'O=C=O',
    'carbon monoxide': '[C-]#[O+]',
    'nitric oxide': '[N]=O',
    'sulfuric acid': 'OS(=O)(=O)O',
    'phosphoric acid': 'OP(=O)(O)O',
    'hydrogen peroxide': 'OO',
    'chloroform': 'C(Cl)(Cl)Cl',
    'dichloromethane': 'C(Cl)Cl',
    'tetrahydrofuran': 'C1CCOC1',
    'thf': 'C1CCOC1',
    'dioxane': 'C1COCCO1',
    'pyridine': 'c1ccncc1',
    'furan': 'c1ccoc1',
    'thiophene': 'c1ccsc1',
    'dmso': 'CS(=O)C',
    'dimethyl sulfoxide': 'CS(=O)C',
}

def name_to_smiles(name: str) -> str:
    """
    Convert common chemical name to SMILES notation.
    
    Args:
        name: Chemical name (case-insensitive)
    
    Returns:
        SMILES string if found, otherwise returns the original input
    """
    name_lower = name.lower().strip()
    return COMMON_MOLECULES.get(name_lower, name)


def is_likely_smiles(text: str) -> bool:
    """
    Check if text is likely SMILES notation.
    
    Args:
        text: Input string
    
    Returns:
        True if likely SMILES, False otherwise
    """
    # SMILES characteristics
    smiles_chars = set('CNOPSFClBrI[]()=#@+-/\\123456789cnops')
    text_chars = set(text)
    
    # If contains typical SMILES characters
    if any(c in text for c in ['C', 'N', 'O', '(', ')', '=', '#', 'c']):
        return True
    
    return False
