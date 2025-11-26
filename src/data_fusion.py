"""
Data fusion utility for combining multiple CSV files.

Supports:
- Molecular data (SMILES)
- Biological assay data
- Chemical properties
- Experimental results
- Intelligent column matching
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class DataFusion:
    """
    Intelligent data fusion for multiple CSV files.
    """
    
    def __init__(self):
        self.files = {}
        self.merged_data = None
    
    def add_file(self, name: str, df: pd.DataFrame):
        """Add a DataFrame to the fusion set."""
        self.files[name] = df
    
    def detect_key_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Detect potential key columns for merging.
        
        Returns list of column names that could be keys.
        """
        key_candidates = []
        
        # Common key column names
        common_keys = [
            'id', 'compound_id', 'molecule_id', 'smiles', 'inchi', 'inchikey',
            'cas', 'name', 'compound_name', 'drug_name'
        ]
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check against common keys
            if any(key in col_lower for key in common_keys):
                key_candidates.append(col)
            
            # Check if column has unique values (potential key)
            elif df[col].nunique() == len(df):
                key_candidates.append(col)
        
        return key_candidates
    
    def suggest_merge_strategy(self) -> Dict:
        """
        Suggest how to merge the files based on their content.
        """
        if len(self.files) < 2:
            return {"strategy": "single_file", "description": "Only one file uploaded"}
        
        strategies = []
        
        # Analyze each file
        file_info = {}
        for name, df in self.files.items():
            info = {
                'has_smiles': 'smiles' in [c.lower() for c in df.columns],
                'has_id': any('id' in c.lower() for c in df.columns),
                'key_columns': self.detect_key_columns(df),
                'shape': df.shape
            }
            file_info[name] = info
        
        # Determine merge strategy
        smiles_files = [n for n, i in file_info.items() if i['has_smiles']]
        
        if len(smiles_files) > 0:
            strategies.append({
                'type': 'molecular_merge',
                'description': 'Merge on molecular identifiers (SMILES)',
                'key_column': 'smiles',
                'files_with_key': smiles_files
            })
        
        # Look for common columns
        if len(self.files) >= 2:
            all_columns = [set(df.columns) for df in self.files.values()]
            common_cols = set.intersection(*all_columns)
            
            if common_cols:
                strategies.append({
                    'type': 'column_merge',
                    'description': f'Merge on common columns: {", ".join(common_cols)}',
                    'key_columns': list(common_cols)
                })
        
        return {
            'strategies': strategies,
            'file_info': file_info,
            'recommendation': strategies[0] if strategies else None
        }
    
    def merge_on_column(
        self,
        key_column: str,
        how: str = 'outer',
        fill_missing: bool = True
    ) -> pd.DataFrame:
        """
        Merge all files on a specific column.
        
        Args:
            key_column: Column name to merge on
            how: Merge type ('inner', 'outer', 'left', 'right')
            fill_missing: Whether to fill missing values
        
        Returns:
            Merged DataFrame
        """
        if not self.files:
            return pd.DataFrame()
        
        # Start with first file
        file_names = list(self.files.keys())
        merged = self.files[file_names[0]].copy()
        
        # Add suffix to distinguish source
        merged.columns = [
            f"{col}_{file_names[0]}" if col != key_column else col 
            for col in merged.columns
        ]
        
        # Merge with subsequent files
        for file_name in file_names[1:]:
            df = self.files[file_name].copy()
            
            # Add suffix
            df.columns = [
                f"{col}_{file_name}" if col != key_column else col 
                for col in df.columns
            ]
            
            # Merge
            merged = pd.merge(
                merged,
                df,
                on=key_column,
                how=how,
                suffixes=('', f'_{file_name}')
            )
        
        # Fill missing values if requested
        if fill_missing:
            for col in merged.select_dtypes(include=[np.number]).columns:
                merged[col].fillna(merged[col].median(), inplace=True)
            
            for col in merged.select_dtypes(include=['object']).columns:
                merged[col].fillna('Unknown', inplace=True)
        
        self.merged_data = merged
        return merged
    
    def concatenate_files(self, axis: int = 0) -> pd.DataFrame:
        """
        Concatenate files (stack them).
        
        Args:
            axis: 0 for rows (stack vertically), 1 for columns (stack horizontally)
        """
        if not self.files:
            return pd.DataFrame()
        
        dfs = list(self.files.values())
        
        if axis == 0:
            # Add source column
            for name, df in self.files.items():
                df['data_source'] = name
        
        merged = pd.concat(dfs, axis=axis, ignore_index=True)
        self.merged_data = merged
        return merged
    
    def smart_merge(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Automatically determine best merge strategy and execute.
        
        Returns:
            merged_df: Merged DataFrame
            merge_info: Information about the merge performed
        """
        suggestions = self.suggest_merge_strategy()
        
        if not suggestions['recommendation']:
            # Fallback: concatenate
            merged = self.concatenate_files(axis=0)
            return merged, {
                'strategy': 'concatenate',
                'description': 'Files stacked vertically'
            }
        
        strategy = suggestions['recommendation']
        
        if strategy['type'] == 'molecular_merge':
            merged = self.merge_on_column(strategy['key_column'], how='outer')
            return merged, strategy
        
        elif strategy['type'] == 'column_merge':
            key_col = strategy['key_columns'][0]
            merged = self.merge_on_column(key_col, how='outer')
            return merged, strategy
        
        else:
            merged = self.concatenate_files(axis=0)
            return merged, {'strategy': 'concatenate'}
    
    def get_merge_preview(self, max_rows: int = 10) -> Dict:
        """
        Get a preview of how files would merge.
        """
        suggestions = self.suggest_merge_strategy()
        
        preview = {
            'total_files': len(self.files),
            'suggestions': suggestions,
            'file_shapes': {name: df.shape for name, df in self.files.items()},
            'column_overlap': {}
        }
        
        # Calculate column overlap
        if len(self.files) >= 2:
            file_cols = {name: set(df.columns) for name, df in self.files.items()}
            all_cols = set.union(*file_cols.values())
            
            for col in all_cols:
                files_with_col = [name for name, cols in file_cols.items() if col in cols]
                preview['column_overlap'][col] = files_with_col
        
        return preview


def analyze_csv_type(df: pd.DataFrame) -> Dict:
    """
    Analyze CSV file type and content.
    """
    analysis = {
        'type': 'unknown',
        'has_molecular_data': False,
        'has_biological_data': False,
        'numeric_columns': [],
        'categorical_columns': [],
        'potential_keys': []
    }
    
    # Check for molecular data
    molecular_indicators = ['smiles', 'inchi', 'mol', 'compound']
    for col in df.columns:
        if any(ind in col.lower() for ind in molecular_indicators):
            analysis['has_molecular_data'] = True
            break
    
    # Check for biological data
    bio_indicators = ['ic50', 'ec50', 'ki', 'activity', 'assay', 'inhibition']
    for col in df.columns:
        if any(ind in col.lower() for ind in bio_indicators):
            analysis['has_biological_data'] = True
            break
    
    # Categorize columns
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            analysis['numeric_columns'].append(col)
        else:
            analysis['categorical_columns'].append(col)
    
    # Determine type
    if analysis['has_molecular_data'] and analysis['has_biological_data']:
        analysis['type'] = 'bioactivity'
    elif analysis['has_molecular_data']:
        analysis['type'] = 'molecular'
    elif analysis['has_biological_data']:
        analysis['type'] = 'biological'
    else:
        analysis['type'] = 'general'
    
    return analysis
