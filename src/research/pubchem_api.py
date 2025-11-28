"""
PubChem API Client for Multimodal Scientific Dataset Collection

Fetches molecular data from PubChem including:
- SMILES notation
- Molecular properties
- Compound names
- Target proteins
"""

import requests
import json
import time
from typing import Dict, List, Optional
import pandas as pd


class PubChemAPI:
    """
    Client for PubChem REST API.
    Documentation: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
    """
    
    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    def __init__(self, rate_limit: float = 0.2):
        """
        Args:
            rate_limit: Seconds between requests (PubChem allows 5 requests/second)
        """
        self.rate_limit = rate_limit
        self.last_request = 0
    
    def _wait_for_rate_limit(self):
        """Ensure rate limit compliance."""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()
    
    def get_compound_by_name(self, name: str) -> Optional[Dict]:
        """
        Get compound information by name.
        
        Args:
            name: Chemical name (e.g., "aspirin")
        
        Returns:
            Dictionary with compound data or None if not found
        """
        self._wait_for_rate_limit()
        
        url = f"{self.BASE_URL}/compound/name/{name}/JSON"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'PC_Compounds' in data and len(data['PC_Compounds']) > 0:
                compound = data['PC_Compounds'][0]
                return self._parse_compound(compound)
            return None
        
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            return None
    
    def get_compound_by_cid(self, cid: int) -> Optional[Dict]:
        """
        Get compound information by CID.
        
        Args:
            cid: PubChem Compound ID
        
        Returns:
            Dictionary with compound data
        """
        self._wait_for_rate_limit()
        
        url = f"{self.BASE_URL}/compound/cid/{cid}/JSON"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'PC_Compounds' in data and len(data['PC_Compounds']) > 0:
                return self._parse_compound(data['PC_Compounds'][0])
            return None
        
        except Exception as e:
            print(f"Error fetching CID {cid}: {e}")
            return None
    
    def get_smiles(self, identifier: str, id_type: str = "name") -> Optional[str]:
        """
        Get SMILES notation for a compound.
        
        Args:
            identifier: Compound name or CID
            id_type: "name" or "cid"
        
        Returns:
            SMILES string or None
        """
        self._wait_for_rate_limit()
        
        url = f"{self.BASE_URL}/compound/{id_type}/{identifier}/property/CanonicalSMILES/JSON"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                props = data['PropertyTable']['Properties']
                if len(props) > 0 and 'CanonicalSMILES' in props[0]:
                    return props[0]['CanonicalSMILES']
            return None
        
        except Exception as e:
            print(f"Error fetching SMILES for {identifier}: {e}")
            return None
    
    def get_properties(self, cid: int) -> Optional[Dict]:
        """
        Get molecular properties.
        
        Args:
            cid: PubChem Compound ID
        
        Returns:
            Dictionary of properties
        """
        self._wait_for_rate_limit()
        
        properties = [
            'MolecularFormula',
            'MolecularWeight',
            'CanonicalSMILES',
            'IsomericSMILES',
            'InChI',
            'InChIKey',
            'IUPACName',
            'XLogP',
            'HBondDonorCount',
            'HBondAcceptorCount'
        ]
        
        props_str = ','.join(properties)
        url = f"{self.BASE_URL}/compound/cid/{cid}/property/{props_str}/JSON"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                return data['PropertyTable']['Properties'][0]
            return None
        
        except Exception as e:
            print(f"Error fetching properties for CID {cid}: {e}")
            return None
    
    def _parse_compound(self, compound: Dict) -> Dict:
        """Parse PubChem compound data."""
        parsed = {
            'cid': compound.get('id', {}).get('id', {}).get('cid'),
            'atoms': {},
            'bonds': {},
            'properties': {}
        }
        
        # Extract atoms
        if 'atoms' in compound:
            parsed['atoms'] = compound['atoms']
        
        # Extract bonds
        if 'bonds' in compound:
            parsed['bonds'] = compound['bonds']
        
        # Extract properties
        if 'props' in compound:
            for prop in compound['props']:
                if 'urn' in prop and 'value' in prop:
                    label = prop['urn'].get('label', 'unknown')
                    parsed['properties'][label] = prop['value']
        
        return parsed
    
    def search_by_similarity(self, smiles: str, threshold: float = 0.9) -> List[Dict]:
        """
        Find similar compounds by SMILES.
        
        Args:
            smiles: Query SMILES
            threshold: Similarity threshold (0-1)
        
        Returns:
            List of similar compounds
        """
        self._wait_for_rate_limit()
        
        # Note: PubChem similarity search is complex, simplified here
        url = f"{self.BASE_URL}/compound/fastsimilarity_2d/smiles/{smiles}/JSON"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                return data['IdentifierList']['CID'][:10]  # Top 10
            return []
        
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []


class UniProtAPI:
    """
    Client for UniProt REST API.
    Documentation: https://www.uniprot.org/help/api
    """
    
    BASE_URL = "https://rest.uniprot.org/uniprotkb"
    
    def __init__(self, rate_limit: float = 0.2):
        self.rate_limit = rate_limit
        self.last_request = 0
    
    def _wait_for_rate_limit(self):
        """Ensure rate limit compliance."""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()
    
    def get_protein(self, uniprot_id: str) -> Optional[Dict]:
        """
        Get protein information by UniProt ID.
        
        Args:
            uniprot_id: UniProt accession (e.g., "P23219")
        
        Returns:
            Protein data dictionary
        """
        self._wait_for_rate_limit()
        
        url = f"{self.BASE_URL}/{uniprot_id}.json"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'uniprot_id': uniprot_id,
                'protein_name': data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value'),
                'gene_name': data.get('genes', [{}])[0].get('geneName', {}).get('value') if data.get('genes') else None,
                'organism': data.get('organism', {}).get('scientificName'),
                'function': data.get('comments', [{}])[0].get('texts', [{}])[0].get('value') if data.get('comments') else None,
                'sequence': data.get('sequence', {}).get('value'),
                'length': data.get('sequence', {}).get('length')
            }
        
        except Exception as e:
            print(f"Error fetching UniProt {uniprot_id}: {e}")
            return None
    
    def search_protein_by_name(self, name: str) -> List[Dict]:
        """
        Search proteins by name.
        
        Args:
            name: Protein name to search
        
        Returns:
            List of matching proteins
        """
        self._wait_for_rate_limit()
        
        url = f"{self.BASE_URL}/search"
        params = {
            'query': f'protein_name:{name}',
            'format': 'json',
            'size': 10
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for entry in data.get('results', []):
                results.append({
                    'uniprot_id': entry.get('primaryAccession'),
                    'protein_name': entry.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value'),
                    'organism': entry.get('organism', {}).get('scientificName')
                })
            
            return results
        
        except Exception as e:
            print(f"Error searching protein {name}: {e}")
            return []


def create_table_3_1_dataset() -> pd.DataFrame:
    """
    Create the exact dataset from Table 3.1 using real PubChem/UniProt data.
    
    Returns:
        DataFrame with multimodal scientific data
    """
    pubchem = PubChemAPI()
    uniprot = UniProtAPI()
    
    # Define the molecules from Table 3.1
    molecules = [
        {
            'name': 'Aspirin',
            'cid': 2244,
            'protein_name': 'Cyclooxygenase-1',
            'uniprot_id': 'P23219',
            'description': 'Aspirin acetylates COX-1 to irreversibly inhibit prostaglandin synthesis.'
        },
        {
            'name': 'Ibuprofen',
            'cid': 3672,
            'protein_name': 'Cyclooxygenase-2',
            'uniprot_id': 'P35354',
            'description': 'Ibuprofen non-selectively inhibits COX-1 and COX-2, reducing inflammatory mediators.'
        },
        {
            'name': 'Paracetamol',
            'cid': 1983,
            'protein_name': 'Prostaglandin G/H Synthase 1',
            'uniprot_id': 'P23219',
            'description': 'Paracetamol acts as a weak COX inhibitor with strong central analgesic effects.'
        },
        {
            'name': 'Caffeine',
            'cid': 2519,
            'protein_name': 'Adenosine Receptor A2A',
            'uniprot_id': 'P29274',
            'description': 'Caffeine antagonizes adenosine A2A receptors, producing stimulatory CNS effects.'
        },
        {
            'name': 'Metformin',
            'cid': 4091,
            'protein_name': 'AMPK alpha-1',
            'uniprot_id': 'Q13131',
            'description': 'Metformin activates AMPK pathways, improving insulin sensitivity.'
        }
    ]
    
    dataset = []
    
    for mol_data in molecules:
        print(f"Fetching data for {mol_data['name']}...")
        
        # Get SMILES from PubChem
        smiles = pubchem.get_smiles(str(mol_data['cid']), id_type='cid')
        
        # Get protein data from UniProt
        protein = uniprot.get_protein(mol_data['uniprot_id'])
        
        entry = {
            'Molecule': mol_data['name'],
            'PubChem_CID': mol_data['cid'],
            'SMILES': smiles,
            'Target_Protein': mol_data['protein_name'],
            'UniProt_ID': mol_data['uniprot_id'],
            'Gene_Name': protein.get('gene_name') if protein else None,
            'Organism': protein.get('organism') if protein else 'Homo sapiens',
            'Text_Description': mol_data['description'],
            'Protein_Function': protein.get('function') if protein else None,
            'Protein_Length': protein.get('length') if protein else None
        }
        
        dataset.append(entry)
        time.sleep(0.5)  # Be nice to APIs
    
    df = pd.DataFrame(dataset)
    return df
