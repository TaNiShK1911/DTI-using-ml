import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict


class DTIDataset(Dataset):
    """PyTorch Dataset for Drug-Target Interaction data."""
    
    def __init__(self, csv_path: str):
        """Initialize dataset from CSV file."""
        self.df = pd.read_csv(csv_path)
        print(f"Loaded dataset with {len(self.df)} samples from {csv_path}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        """Return a single data point."""
        row = self.df.iloc[idx]
        
        return {
            'smiles': row['smiles'],
            'sequence': row['sequence'],
            'affinity': float(row['affinity']),
            'affinity_type': row['affinity_type']
        }
