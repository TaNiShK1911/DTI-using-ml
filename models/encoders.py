import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List


class DrugEncoder(nn.Module):
    """Encode SMILES strings using pre-trained transformer."""
    
    def __init__(self, model_name: str = "seyonec/PubChem10M_SMILES_BPE_450k", freeze: bool = True):
        super().__init__()
        print(f"Loading drug encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.embedding_dim = self.model.config.hidden_size
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        """Encode SMILES to embeddings [batch_size, hidden_dim]."""
        # Tokenize with proper settings
        encoded = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True
        )
        
        # Move to same device as model (optimized for CUDA)
        device = next(self.model.parameters()).device
        encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}
        
        # Verify token IDs are within vocabulary range
        vocab_size = self.model.config.vocab_size
        if encoded['input_ids'].max() >= vocab_size:
            encoded['input_ids'] = torch.clamp(encoded['input_ids'], 0, vocab_size - 1)
        
        # Get embeddings with gradient checkpointing for memory efficiency
        with torch.set_grad_enabled(self.training):
            outputs = self.model(**encoded)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return self.dropout(embeddings)


class ProteinEncoder(nn.Module):
    """Encode protein sequences using pre-trained language model."""
    
    def __init__(self, model_name: str = "Rostlab/prot_bert", freeze: bool = True):
        super().__init__()
        print(f"Loading protein encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.embedding_dim = self.model.config.hidden_size
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, sequences: List[str]) -> torch.Tensor:
        """Encode sequences to embeddings [batch_size, hidden_dim]."""
        # Add spaces between amino acids for ProtBERT
        sequences_spaced = [' '.join(list(seq[:512])) for seq in sequences]
        
        # Tokenize with proper settings
        encoded = self.tokenizer(
            sequences_spaced,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True
        )
        
        # Move to same device as model (optimized for CUDA)
        device = next(self.model.parameters()).device
        encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}
        
        # Verify token IDs are within vocabulary range
        vocab_size = self.model.config.vocab_size
        if encoded['input_ids'].max() >= vocab_size:
            encoded['input_ids'] = torch.clamp(encoded['input_ids'], 0, vocab_size - 1)
        
        # Get embeddings with gradient checkpointing for memory efficiency
        with torch.set_grad_enabled(self.training):
            outputs = self.model(**encoded)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return self.dropout(embeddings)
