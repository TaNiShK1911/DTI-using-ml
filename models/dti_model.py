import torch
import torch.nn as nn
from typing import Dict, List
from models.encoders import DrugEncoder, ProteinEncoder
from models.attention import CoAttentionModule
from models.evidential import EvidentialRegressionHead


class DTIModel(nn.Module):
    """Complete Drug-Target Interaction prediction model."""
    
    def __init__(self, config: dict = None):
        super().__init__()
        
        if config is None:
            config = {
                'drug_model': 'seyonec/PubChem10M_SMILES_BPE_450k',
                'protein_model': 'Rostlab/prot_bert',
                'attention_dim': 512,
                'hidden_dims': [512, 256],
                'freeze_encoders': True
            }
        
        self.config = config
        
        # Initialize encoders
        self.drug_encoder = DrugEncoder(
            model_name=config['drug_model'],
            freeze=config['freeze_encoders']
        )
        self.protein_encoder = ProteinEncoder(
            model_name=config['protein_model'],
            freeze=config['freeze_encoders']
        )
        
        # Co-attention module
        self.co_attention = CoAttentionModule(
            drug_dim=self.drug_encoder.embedding_dim,
            protein_dim=self.protein_encoder.embedding_dim,
            attention_dim=config['attention_dim']
        )
        
        # Prediction head
        self.prediction_head = EvidentialRegressionHead(
            input_dim=config['attention_dim'],
            hidden_dims=config['hidden_dims']
        )
    
    def forward(self, smiles: List[str], sequences: List[str]) -> Dict[str, torch.Tensor]:
        """
        Args:
            smiles: List of SMILES strings
            sequences: List of protein sequences
        
        Returns:
            Dictionary with affinity, uncertainty, attention_weights, evidential_params
        """
        # Encode inputs
        drug_emb = self.drug_encoder(smiles)
        protein_emb = self.protein_encoder(sequences)
        
        # Co-attention
        fused_features, attention_weights = self.co_attention(drug_emb, protein_emb)
        
        # Prediction
        affinity, uncertainty, evidential_params = self.prediction_head(fused_features)
        
        return {
            'affinity': affinity,
            'uncertainty': uncertainty,
            'attention_weights': attention_weights,
            'evidential_params': evidential_params
        }
    
    def save(self, path: str):
        """Save model weights and configuration."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(config=checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"Model loaded from {path}")
        return model
