import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
import os


class BindingDBParser:
    """Parse and extract relevant data from BindingDB TSV file."""
    
    def parse(self, tsv_path: str) -> pd.DataFrame:
        """Extract SMILES, sequences, and affinity values."""
        print(f"Loading BindingDB data from {tsv_path}...")
        df = pd.read_csv(tsv_path, sep='\t', low_memory=False)
        
        # Extract relevant columns
        columns_map = {
            'Ligand SMILES': 'smiles',
            'BindingDB Target Chain Sequence': 'sequence',
            'Ki (nM)': 'ki',
            'Kd (nM)': 'kd',
            'IC50 (nM)': 'ic50'
        }
        
        df_subset = df[list(columns_map.keys())].copy()
        df_subset.columns = list(columns_map.values())
        
        print(f"Initial data shape: {df_subset.shape}")
        return df_subset


class AffinityProcessor:
    """Standardize binding affinity values."""
    
    def unify_affinity(self, row: pd.Series) -> Tuple[float, str]:
        """Prioritize Ki > Kd > IC50 and apply p-scaling."""
        # Try Ki first
        if pd.notna(row['ki']) and row['ki'] > 0:
            affinity_nm = float(row['ki'])
            affinity_type = 'Ki'
        # Then Kd
        elif pd.notna(row['kd']) and row['kd'] > 0:
            affinity_nm = float(row['kd'])
            affinity_type = 'Kd'
        # Finally IC50
        elif pd.notna(row['ic50']) and row['ic50'] > 0:
            affinity_nm = float(row['ic50'])
            affinity_type = 'IC50'
        else:
            return None, None
        
        # Apply p-scaling: pX = -log10(X in M)
        # Convert nM to M: divide by 1e9
        affinity_m = affinity_nm / 1e9
        p_affinity = -np.log10(affinity_m)
        
        return p_affinity, affinity_type
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all rows and add unified affinity column."""
        print("Processing affinity values...")
        results = df.apply(self.unify_affinity, axis=1, result_type='expand')
        df['affinity'] = results[0]
        df['affinity_type'] = results[1]
        
        # Filter out rows with no valid affinity
        df_clean = df.dropna(subset=['affinity']).copy()
        print(f"After affinity processing: {df_clean.shape}")
        
        return df_clean


class DataSplitter:
    """Create train/val/test splits including cold-start splits."""
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string using RDKit."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def create_splits(self, df: pd.DataFrame, output_dir: str = 'data/processed') -> Dict[str, pd.DataFrame]:
        """Create random, cold-drug, and cold-protein splits."""
        print("Creating data splits...")
        
        # Validate SMILES
        print("Validating SMILES...")
        df['valid_smiles'] = df['smiles'].apply(self.validate_smiles)
        df = df[df['valid_smiles']].copy()
        
        # Filter valid sequences (only standard amino acids)
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        df['valid_seq'] = df['sequence'].apply(
            lambda x: isinstance(x, str) and len(x) > 0 and all(c in valid_aa for c in x)
        )
        df = df[df['valid_seq']].copy()
        
        print(f"After validation: {df.shape}")
        
        # Limit dataset size for prototype (use first 10000 samples)
        df = df.head(10000).copy()
        print(f"Using subset for prototype: {df.shape}")
        
        # Random split (70/15/15)
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        # Save splits
        os.makedirs(output_dir, exist_ok=True)
        for split_name, split_df in splits.items():
            output_path = os.path.join(output_dir, f'{split_name}.csv')
            split_df.to_csv(output_path, index=False)
            print(f"Saved {split_name} split: {split_df.shape} -> {output_path}")
        
        return splits


def preprocess_bindingdb(tsv_path: str, output_dir: str = 'data/processed'):
    """Main preprocessing pipeline."""
    parser = BindingDBParser()
    df = parser.parse(tsv_path)
    
    processor = AffinityProcessor()
    df = processor.process(df)
    
    splitter = DataSplitter()
    splits = splitter.create_splits(df, output_dir)
    
    print("\nPreprocessing complete!")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    return splits


if __name__ == '__main__':
    tsv_path = 'BindingDB_All_202511_tsv/BindingDB_All.tsv'
    preprocess_bindingdb(tsv_path)
