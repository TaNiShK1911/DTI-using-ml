import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# enable DataFrame progress_apply / progress_map
tqdm.pandas()


class BindingDBParser:
    """Parse and extract relevant data from BindingDB TSV file."""

    def parse(self, tsv_path: str, use_chunks: bool = True, chunksize: int = 50_000, 
              max_rows: Optional[int] = 100_000) -> pd.DataFrame:
        """Extract SMILES, sequences, and affinity values.

        If use_chunks=True will stream the TSV in chunks and show a progress bar while
        concatenating relevant columns. This is helpful for very large BindingDB files.
        
        Args:
            tsv_path: Path to BindingDB TSV file
            use_chunks: Whether to read in chunks (recommended for large files)
            chunksize: Number of rows per chunk
            max_rows: Maximum rows to read (None for all rows)
        """
        print(f"Loading BindingDB data from {tsv_path}...")
        columns_to_read = ['Ligand SMILES', 'BindingDB Target Chain Sequence 1', 
                          'Ki (nM)', 'Kd (nM)', 'IC50 (nM)']

        if use_chunks:
            chunks = []
            total_rows = 0
            print(f"Reading in chunks of {chunksize:,} rows...")
            print(f"Will stop after {max_rows:,} rows" if max_rows else "Reading all rows")
            
            try:
                reader = pd.read_csv(tsv_path, sep='\t', usecols=columns_to_read, 
                                    low_memory=True, chunksize=chunksize)
                
                for i, chunk in enumerate(tqdm(reader, desc="Reading TSV chunks")):
                    # Rename columns explicitly
                    chunk = chunk.rename(columns={
                        'Ligand SMILES': 'smiles',
                        'BindingDB Target Chain Sequence 1': 'sequence',
                        'Ki (nM)': 'ki',
                        'Kd (nM)': 'kd',
                        'IC50 (nM)': 'ic50'
                    })
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    
                    # Stop early if we have enough rows
                    if max_rows and total_rows >= max_rows:
                        print(f"Reached {total_rows:,} rows, stopping early...")
                        break
                
                df = pd.concat(chunks, ignore_index=True)
                
                # Trim to exact max_rows if specified
                if max_rows and len(df) > max_rows:
                    df = df.head(max_rows)
                    
            except Exception as e:
                print(f"Error during chunked reading: {e}")
                print("Trying to read with nrows limit...")
                # Fallback: read with nrows limit
                nrows = max_rows if max_rows else 100_000
                df = pd.read_csv(tsv_path, sep='\t', usecols=columns_to_read, 
                               low_memory=True, nrows=nrows)
                df = df.rename(columns={
                    'Ligand SMILES': 'smiles',
                    'BindingDB Target Chain Sequence 1': 'sequence',
                    'Ki (nM)': 'ki',
                    'Kd (nM)': 'kd',
                    'IC50 (nM)': 'ic50'
                })
        else:
            # Direct read with optional row limit
            nrows = max_rows if max_rows else None
            df = pd.read_csv(tsv_path, sep='\t', usecols=columns_to_read, 
                           low_memory=True, nrows=nrows)
            df = df.rename(columns={
                'Ligand SMILES': 'smiles',
                'BindingDB Target Chain Sequence 1': 'sequence',
                'Ki (nM)': 'ki',
                'Kd (nM)': 'kd',
                'IC50 (nM)': 'ic50'
            })

        print(f"Initial data shape: {df.shape}")
        return df


class AffinityProcessor:
    """Standardize binding affinity values."""

    def unify_affinity(self, row: pd.Series) -> Tuple[Optional[float], Optional[str]]:
        """Prioritize Ki > Kd > IC50 and apply p-scaling.

        Returns:
            (p_affinity, affinity_type) or (None, None) when no valid affinity present.
        """
        try:
            # Try Ki first
            if pd.notna(row['ki']) and row['ki'] != '' and float(row['ki']) > 0:
                affinity_nm = float(row['ki'])
                affinity_type = 'Ki'
            # Then Kd
            elif pd.notna(row['kd']) and row['kd'] != '' and float(row['kd']) > 0:
                affinity_nm = float(row['kd'])
                affinity_type = 'Kd'
            # Finally IC50
            elif pd.notna(row['ic50']) and row['ic50'] != '' and float(row['ic50']) > 0:
                affinity_nm = float(row['ic50'])
                affinity_type = 'IC50'
            else:
                return None, None

            # Convert nM to M: divide by 1e9
            affinity_m = affinity_nm / 1e9
            # Prevent math error for non-positive values
            if affinity_m <= 0:
                return None, None

            # Apply p-scaling: pX = -log10(X in M)
            p_affinity = -np.log10(affinity_m)
            return p_affinity, affinity_type
        except Exception:
            return None, None

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all rows and add unified affinity column, with progress bar."""
        print("Processing affinity values...")
        print(f"Input shape: {df.shape}")
        
        # Process affinity values row by row
        affinity_list = []
        affinity_type_list = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing affinity"):
            p_aff, aff_type = self.unify_affinity(row)
            affinity_list.append(p_aff)
            affinity_type_list.append(aff_type)
        
        df['affinity'] = affinity_list
        df['affinity_type'] = affinity_type_list

        # Filter out rows with no valid affinity
        df_clean = df.dropna(subset=['affinity']).copy()
        print(f"After affinity processing: {df_clean.shape}")
        print(f"Sequence non-null after affinity filter: {df_clean['sequence'].notna().sum()}")
        
        if len(df_clean) == 0:
            print("WARNING: No valid affinity values found!")
            print("Checking affinity columns:")
            print(f"Ki non-null: {df['ki'].notna().sum()}")
            print(f"Kd non-null: {df['kd'].notna().sum()}")
            print(f"IC50 non-null: {df['ic50'].notna().sum()}")
        
        # Also filter out rows with no sequence
        df_clean = df_clean.dropna(subset=['sequence']).copy()
        print(f"After removing rows without sequences: {df_clean.shape}")

        return df_clean


class DataSplitter:
    """Create train/val/test splits including cold-start splits (basic)."""

    def validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string using RDKit."""
        if not isinstance(smiles, str) or smiles.strip() == '':
            return False
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            return False

    def create_splits(self, df: pd.DataFrame, output_dir: str = 'data/processed',
                      prototype_limit: Optional[int] = 10_000) -> Dict[str, pd.DataFrame]:
        """Create random splits (train/val/test). Uses progress bars during validation and saving."""
        print("Creating data splits...")

        # Validate SMILES with progress bar
        print("Validating SMILES...")
        df['valid_smiles'] = df['smiles'].progress_apply(self.validate_smiles)
        df = df[df['valid_smiles']].copy().reset_index(drop=True)
        print(f"After SMILES validation: {df.shape}")
        print(f"Sample sequences after SMILES validation: {df['sequence'].head(3).tolist()}")

        # Validate sequences (only standard amino acids)
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        print("Validating protein sequences...")
        print(f"Sample sequences before validation:")
        print(df['sequence'].head(3))
        print(f"Sequence column type: {df['sequence'].dtype}")
        print(f"Non-null sequences: {df['sequence'].notna().sum()}")
        
        df['valid_seq'] = df['sequence'].progress_apply(
            lambda x: isinstance(x, str) and len(x) > 0 and all((c in valid_aa) for c in x)
        )
        print(f"Valid sequences count: {df['valid_seq'].sum()}")
        df = df[df['valid_seq']].copy()

        print(f"After validation: {df.shape}")

        # Limit dataset size for prototype if requested
        if prototype_limit is not None and prototype_limit > 0 and len(df) > prototype_limit:
            df = df.head(prototype_limit).copy()
            print(f"Using subset for prototype: {df.shape}")
        else:
            print(f"Using full dataset: {df.shape}")

        # Random split (70/15/15)
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        splits = {
            'train': train_df.reset_index(drop=True),
            'val': val_df.reset_index(drop=True),
            'test': test_df.reset_index(drop=True)
        }

        # Save splits with progress display
        os.makedirs(output_dir, exist_ok=True)
        for split_name, split_df in tqdm(splits.items(), desc="Saving splits"):
            output_path = os.path.join(output_dir, f'{split_name}.csv')
            split_df.to_csv(output_path, index=False)
            tqdm.write(f"Saved {split_name} split: {split_df.shape} -> {output_path}")

        return splits


def preprocess_bindingdb(tsv_path: str, output_dir: str = 'data/processed',
                         use_chunks: bool = True, max_rows: Optional[int] = 100_000,
                         prototype_limit: Optional[int] = 10_000):
    """Main preprocessing pipeline with progress bars across stages.
    
    Args:
        tsv_path: Path to BindingDB TSV file
        output_dir: Directory to save processed data
        use_chunks: Whether to read in chunks (recommended for large files)
        max_rows: Maximum rows to read from TSV (None for all)
        prototype_limit: Final dataset size after filtering (None for all)
    """
    parser = BindingDBParser()
    df = parser.parse(tsv_path, use_chunks=use_chunks, max_rows=max_rows)

    processor = AffinityProcessor()
    df = processor.process(df)

    splitter = DataSplitter()
    splits = splitter.create_splits(df, output_dir=output_dir, prototype_limit=prototype_limit)

    print("\nPreprocessing complete!")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    return splits


if __name__ == '__main__':
    # Example invocation
    tsv_path = 'BindingDB_All_202511_tsv/BindingDB_All.tsv'
    # Use chunks for large files, read max 100k rows, then filter to 10k for prototype
    preprocess_bindingdb(tsv_path, output_dir='data/processed', 
                        use_chunks=True, max_rows=100_000, prototype_limit=10_000)
