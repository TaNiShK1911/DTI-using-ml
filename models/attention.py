import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CoAttentionModule(nn.Module):
    """Co-attention mechanism for drug-protein interaction."""
    
    def __init__(self, drug_dim: int, protein_dim: int, attention_dim: int = 512):
        super().__init__()
        
        # Store original dimensions
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        
        # Calculate feature group sizes
        self.drug_features = 8
        self.protein_features = 8
        self.drug_feature_dim = drug_dim // self.drug_features
        self.protein_feature_dim = protein_dim // self.protein_features
        
        # Projection layers for drug
        self.drug_query = nn.Linear(self.drug_feature_dim, attention_dim)
        self.drug_key = nn.Linear(self.drug_feature_dim, attention_dim)
        self.drug_value = nn.Linear(self.drug_feature_dim, attention_dim)
        
        # Projection layers for protein
        self.protein_query = nn.Linear(self.protein_feature_dim, attention_dim)
        self.protein_key = nn.Linear(self.protein_feature_dim, attention_dim)
        self.protein_value = nn.Linear(self.protein_feature_dim, attention_dim)
        
        self.attention_dim = attention_dim
        self.scale = attention_dim ** 0.5
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(attention_dim * 2, attention_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(attention_dim)
        )
    
    def forward(self, drug_emb: torch.Tensor, protein_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            drug_emb: [batch_size, drug_dim]
            protein_emb: [batch_size, protein_dim]
        
        Returns:
            fused_representation: [batch_size, attention_dim]
            attention_weights: [batch_size, drug_features, protein_features]
        """
        batch_size = drug_emb.size(0)
        
        # Split into feature groups for richer attention visualization
        drug_split = drug_emb.view(batch_size, self.drug_features, self.drug_feature_dim)
        protein_split = protein_emb.view(batch_size, self.protein_features, self.protein_feature_dim)
        
        # Project to attention space
        drug_q = self.drug_query(drug_split)  # [batch, drug_features, attention_dim]
        protein_k = self.protein_key(protein_split)  # [batch, protein_features, attention_dim]
        protein_v = self.protein_value(protein_split)  # [batch, protein_features, attention_dim]
        
        # Compute cross-attention: drug features attending to protein features
        scores = torch.matmul(drug_q, protein_k.transpose(-2, -1)) / self.scale  # [batch, drug_features, protein_features]
        attention_weights = F.softmax(scores, dim=-1)  # [batch, drug_features, protein_features]
        
        # Apply attention
        attended_protein = torch.matmul(attention_weights, protein_v)  # [batch, drug_features, attention_dim]
        
        # Also compute protein-to-drug attention for fusion
        protein_q = self.protein_query(protein_split)
        drug_k = self.drug_key(drug_split)
        drug_v = self.drug_value(drug_split)
        
        scores_pd = torch.matmul(protein_q, drug_k.transpose(-2, -1)) / self.scale
        attn_weights_pd = F.softmax(scores_pd, dim=-1)
        attended_drug = torch.matmul(attn_weights_pd, drug_v)  # [batch, protein_features, attention_dim]
        
        # Pool attended features
        attended_protein_pooled = attended_protein.mean(dim=1)  # [batch, attention_dim]
        attended_drug_pooled = attended_drug.mean(dim=1)  # [batch, attention_dim]
        
        # Concatenate and fuse
        fused = torch.cat([attended_drug_pooled, attended_protein_pooled], dim=-1)  # [batch, attention_dim * 2]
        fused_representation = self.fusion(fused)  # [batch, attention_dim]
        
        return fused_representation, attention_weights
