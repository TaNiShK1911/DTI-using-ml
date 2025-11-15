import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr, spearmanr
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import DTIDataset
from models.dti_model import DTIModel
from models.evidential import EvidentialLoss


class Trainer:
    """Training pipeline for DTI model."""
    
    def __init__(self, model: DTIModel, config: dict):
        self.model = model
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = EvidentialLoss(coef=config.get('evidential_coef', 0.1))
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            smiles = batch['smiles']
            sequences = batch['sequence']
            affinity = batch['affinity'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(smiles, sequences)
            
            # Compute loss
            loss = self.criterion(
                outputs['affinity'],
                affinity,
                outputs['evidential_params']
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(outputs['affinity'].detach().cpu().numpy())
            targets.extend(affinity.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        pearson = pearsonr(predictions, targets)[0]
        spearman = spearmanr(predictions, targets)[0]
        
        return {
            'loss': total_loss / len(train_loader),
            'mse': mse,
            'mae': mae,
            'pearson': pearson,
            'spearman': spearman
        }
    
    def validate(self, val_loader: DataLoader) -> dict:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                smiles = batch['smiles']
                sequences = batch['sequence']
                affinity = batch['affinity'].to(self.device)
                
                outputs = self.model(smiles, sequences)
                
                loss = self.criterion(
                    outputs['affinity'],
                    affinity,
                    outputs['evidential_params']
                )
                
                total_loss += loss.item()
                predictions.extend(outputs['affinity'].cpu().numpy())
                targets.extend(affinity.cpu().numpy())
                uncertainties.extend(outputs['uncertainty'].cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        uncertainties = np.array(uncertainties)
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        pearson = pearsonr(predictions, targets)[0]
        spearman = spearmanr(predictions, targets)[0]
        
        return {
            'loss': total_loss / len(val_loader),
            'mse': mse,
            'mae': mae,
            'pearson': pearson,
            'spearman': spearman,
            'mean_uncertainty': np.mean(uncertainties)
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """Full training loop."""
        print(f"Training on device: {self.device}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}, "
                  f"Pearson: {train_metrics['pearson']:.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader)
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"MSE: {val_metrics['mse']:.4f}, MAE: {val_metrics['mae']:.4f}, "
                  f"Pearson: {val_metrics['pearson']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                os.makedirs('checkpoints', exist_ok=True)
                self.model.save('checkpoints/best_model.pt')
                print("Saved best model!")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 5):
                print(f"Early stopping after {epoch + 1} epochs")
                break


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return {
        'smiles': [item['smiles'] for item in batch],
        'sequence': [item['sequence'] for item in batch],
        'affinity': torch.tensor([item['affinity'] for item in batch], dtype=torch.float32),
        'affinity_type': [item['affinity_type'] for item in batch]
    }


def main():
    """Main training script."""
    # Configuration
    config = {
        'batch_size': 8,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'evidential_coef': 0.1,
        'patience': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = DTIDataset('data/processed/train.csv')
    val_dataset = DTIDataset('data/processed/val.csv')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize model
    print("Initializing model...")
    model = DTIModel()
    
    # Train
    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader, config['num_epochs'])
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
