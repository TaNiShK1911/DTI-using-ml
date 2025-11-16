import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import platform
import numpy as np
from scipy.stats import pearsonr, spearmanr
import os
import sys
import json
from copy import deepcopy
from datetime import datetime

# Suppress torch.compile errors on Windows (set early before any operations)
if platform.system() == 'Windows':
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
    except:
        pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import DTIDataset
from models.dti_model import DTIModel
from models.evidential import EvidentialLoss
from models.range_focused_loss import RangeFocusedEvidentialLoss


class ExponentialMovingAverage:
    """Exponential Moving Average (EMA) for model weights to improve stability."""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        """Register all parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class ImprovedTrainer:
    """Enhanced training pipeline with advanced techniques for better accuracy and uncertainty."""
    
    def __init__(self, model: DTIModel, config: dict):
        self.model = model
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Compile model for faster training (PyTorch 2.0+)
        # Disable by default on Windows due to compiler requirements
        is_windows = platform.system() == 'Windows'
        
        # Always disable compilation on Windows, regardless of config
        compile_model = False  # Default to False
        if is_windows:
            print("=" * 80)
            print("WINDOWS DETECTED: torch.compile() disabled (requires C++ compiler)")
            print("=" * 80)
            # Error suppression already set at module level
            # Force compile_model to False on Windows
            config['compile_model'] = False
        elif config.get('compile_model', False):
            # Only try compilation on non-Windows systems if explicitly enabled
            if hasattr(torch, 'compile'):
                try:
                    print("Compiling model with torch.compile() for faster training...")
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                    print("Model compiled successfully!")
                    compile_model = True
                except Exception as e:
                    print(f"Warning: Model compilation failed: {e}. Continuing without compilation.")
                    # Error suppression already set at module level if on Windows
            else:
                print("Note: torch.compile() not available in this PyTorch version.")
        
        # Enable mixed precision training
        self.use_amp = config.get('use_amp', True) and self.device == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Range-focused loss for strong (8-12) and weak (<5) binding affinities
        use_range_focused = config.get('use_range_focused_loss', True)
        if use_range_focused:
            self.criterion = RangeFocusedEvidentialLoss(
                coef=config.get('evidential_coef', 0.1),
                lambda_kl=config.get('lambda_kl', 0.01),
                strong_weight=config.get('strong_weight', 2.0),
                weak_weight=config.get('weak_weight', 2.0),
                annealing_epochs=config.get('annealing_epochs', 5)
            )
            print("Using Range-Focused Loss (emphasizes strong 8-12 and weak <5 binding)")
        else:
            self.criterion = EvidentialLoss(
                coef=config.get('evidential_coef', 0.1),
                lambda_kl=config.get('lambda_kl', 0.01),
                annealing_epochs=config.get('annealing_epochs', 10)
            )
        
        # Separate learning rates for encoders and other layers
        encoder_params = []
        other_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'encoder' in name:
                    encoder_params.append(param)
                else:
                    other_params.append(param)
        
        # Use different learning rates for fine-tuning
        self.optimizer = torch.optim.AdamW(
            [
                {'params': encoder_params, 'lr': config.get('encoder_lr', 1e-5)},
                {'params': other_params, 'lr': config.get('learning_rate', 1e-4)}
            ],
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Better learning rate scheduling with warmup
        warmup_steps = config.get('warmup_steps', 500)
        total_steps = config.get('steps_per_epoch', 100) * config.get('num_epochs', 20)
        
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            else:
                remaining_steps = max(total_steps - warmup_steps, 1)
                progress = (step - warmup_steps) / remaining_steps
                return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
        
        # Use StepLR as base, we'll update manually with step count
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # EMA for model weights
        self.use_ema = config.get('use_ema', True)
        if self.use_ema:
            self.ema = ExponentialMovingAverage(model, decay=config.get('ema_decay', 0.999))
        else:
            self.ema = None
        
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.best_val_loss = float('inf')
        self.best_val_pearson = -float('inf')
        self.best_val_mae = float('inf')
        self.best_strong_mae = float('inf')
        self.best_weak_mae = float('inf')
        self.patience_counter = 0
        self.global_step = 0  # Track global step for scheduler
        
        # Training history with range-specific metrics
        self.history = {
            'train': {
                'loss': [], 'mse': [], 'mae': [], 'pearson': [], 'spearman': [],
                'strong_mae': [], 'weak_mae': [], 'strong_pearson': [], 'weak_pearson': []
            },
            'val': {
                'loss': [], 'mse': [], 'mae': [], 'pearson': [], 'spearman': [], 'uncertainty': [],
                'strong_mae': [], 'weak_mae': [], 'strong_pearson': [], 'weak_pearson': [],
                'strong_precision': [], 'strong_recall': [], 'weak_precision': [], 'weak_recall': []
            }
        }
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif hasattr(obj, 'dtype'):  # Catch any other numpy scalar types
            if np.issubdtype(obj.dtype, np.integer):
                return int(obj)
            elif np.issubdtype(obj.dtype, np.floating):
                return float(obj)
        return obj
    
    def _convert_metric(self, value):
        """Convert a single metric value to Python native type."""
        if isinstance(value, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(value)
        elif isinstance(value, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(value)
        elif hasattr(value, 'dtype'):
            if np.issubdtype(value.dtype, np.integer):
                return int(value)
            elif np.issubdtype(value.dtype, np.floating):
                return float(value)
        return value
    
    def unfreeze_encoders(self, epoch: int, unfreeze_epoch: int = 5):
        """Progressively unfreeze encoders for fine-tuning."""
        if epoch == unfreeze_epoch:
            print(f"\nUnfreezing encoders at epoch {epoch}...")
            for name, param in self.model.named_parameters():
                if 'encoder' in name:
                    param.requires_grad = True
            # Recreate optimizer with unfrozen parameters
            encoder_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if 'encoder' in name:
                        encoder_params.append(param)
                    else:
                        other_params.append(param)
            
            self.optimizer = torch.optim.AdamW(
                [
                    {'params': encoder_params, 'lr': self.config.get('encoder_lr', 1e-5)},
                    {'params': other_params, 'lr': self.config.get('learning_rate', 1e-4)}
                ],
                weight_decay=self.config.get('weight_decay', 0.01),
                betas=(0.9, 0.999),
                eps=1e-8
            )
            print("Encoders unfrozen and optimizer recreated.")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        """Train for one epoch with advanced techniques."""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        uncertainties = []
        
        # Update loss function epoch for annealing
        self.criterion.set_epoch(epoch)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
        for batch_idx, batch in enumerate(pbar):
            smiles = batch['smiles']
            sequences = batch['sequence']
            affinity = batch['affinity'].to(self.device)
            
            # Mixed precision forward pass
            if self.use_amp:
                with autocast(device_type='cuda'):
                    outputs = self.model(smiles, sequences)
                    loss = self.criterion(
                        outputs['affinity'],
                        affinity,
                        outputs['evidential_params']
                    )
                    loss = loss / self.gradient_accumulation_steps
                
                # Scaled backward pass
                self.scaler.scale(loss).backward()
                
                # Update weights with gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.config.get('max_grad_norm', 1.0)
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    # Scheduler step after optimizer step (PyTorch 1.1.0+ requirement)
                    self.scheduler.step()
                    
                    # Update EMA less frequently for speed (every N steps)
                    if self.ema is not None and (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        if (self.global_step + 1) % self.config.get('ema_update_freq', 1) == 0:
                            self.ema.update()
            else:
                # Standard training without AMP
                outputs = self.model(smiles, sequences)
                loss = self.criterion(
                    outputs['affinity'],
                    affinity,
                    outputs['evidential_params']
                )
                loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.get('max_grad_norm', 1.0)
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    # Scheduler step after optimizer step (PyTorch 1.1.0+ requirement)
                    if self.global_step > 0:  # Only step after first update
                        self.scheduler.step()
                    
                    # Update EMA less frequently for speed
                    if self.ema is not None:
                        if (self.global_step + 1) % self.config.get('ema_update_freq', 1) == 0:
                            self.ema.update()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            predictions.extend(outputs['affinity'].detach().cpu().numpy())
            targets.extend(affinity.cpu().numpy())
            uncertainties.extend(outputs['uncertainty'].detach().cpu().numpy())
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'lr': f"{current_lr:.2e}",
                'unc': f"{np.mean(uncertainties[-len(batch['smiles']):]):.3f}"
            })
        
        # Compute metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        uncertainties = np.array(uncertainties)
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        pearson = pearsonr(predictions, targets)[0]
        spearman = spearmanr(predictions, targets)[0]
        mean_uncertainty = np.mean(uncertainties)
        
        # Range-specific metrics for strong (8-12) and weak (<5) binding
        strong_mask = (targets >= 8.0) & (targets <= 12.0)
        weak_mask = targets < 5.0
        
        strong_mae = np.mean(np.abs(predictions[strong_mask] - targets[strong_mask])) if np.any(strong_mask) else 0.0
        weak_mae = np.mean(np.abs(predictions[weak_mask] - targets[weak_mask])) if np.any(weak_mask) else 0.0
        strong_pearson = pearsonr(predictions[strong_mask], targets[strong_mask])[0] if np.sum(strong_mask) > 1 else 0.0
        weak_pearson = pearsonr(predictions[weak_mask], targets[weak_mask])[0] if np.sum(weak_mask) > 1 else 0.0
        
        return {
            'loss': total_loss / len(train_loader),
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'pearson': pearson,
            'spearman': spearman,
            'mean_uncertainty': mean_uncertainty,
            'strong_mae': strong_mae,
            'weak_mae': weak_mae,
            'strong_pearson': strong_pearson,
            'weak_pearson': weak_pearson
        }
    
    def validate(self, val_loader: DataLoader, use_ema: bool = True) -> dict:
        """Evaluate on validation set with optional EMA."""
        # Use EMA weights for validation if available
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        
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
                
                # Use mixed precision for validation
                if self.use_amp:
                    with autocast(device_type='cuda'):
                        outputs = self.model(smiles, sequences)
                        loss = self.criterion(
                            outputs['affinity'],
                            affinity,
                            outputs['evidential_params']
                        )
                else:
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
        
        # Restore original weights if using EMA
        if use_ema and self.ema is not None:
            self.ema.restore()
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        uncertainties = np.array(uncertainties)
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        pearson = pearsonr(predictions, targets)[0]
        spearman = spearmanr(predictions, targets)[0]
        mean_uncertainty = np.mean(uncertainties)
        std_uncertainty = np.std(uncertainties)
        
        # Uncertainty calibration: compute correlation between error and uncertainty
        errors = np.abs(predictions - targets)
        uncertainty_correlation = pearsonr(errors, uncertainties)[0] if len(errors) > 1 else 0.0
        
        # Range-specific metrics for strong (8-12) and weak (<5) binding
        strong_mask = (targets >= 8.0) & (targets <= 12.0)
        weak_mask = targets < 5.0
        
        strong_mae = np.mean(np.abs(predictions[strong_mask] - targets[strong_mask])) if np.any(strong_mask) else 0.0
        weak_mae = np.mean(np.abs(predictions[weak_mask] - targets[weak_mask])) if np.any(weak_mask) else 0.0
        strong_pearson = pearsonr(predictions[strong_mask], targets[strong_mask])[0] if np.sum(strong_mask) > 1 else 0.0
        weak_pearson = pearsonr(predictions[weak_mask], targets[weak_mask])[0] if np.sum(weak_mask) > 1 else 0.0
        
        # Classification accuracy for strong/weak binding
        # Strong binding: predicted >= 8, Weak binding: predicted < 5
        pred_strong = predictions >= 8.0
        pred_weak = predictions < 5.0
        true_strong = strong_mask
        true_weak = weak_mask
        
        strong_precision = np.sum(pred_strong & true_strong) / (np.sum(pred_strong) + 1e-8)
        strong_recall = np.sum(pred_strong & true_strong) / (np.sum(true_strong) + 1e-8)
        weak_precision = np.sum(pred_weak & true_weak) / (np.sum(pred_weak) + 1e-8)
        weak_recall = np.sum(pred_weak & true_weak) / (np.sum(true_weak) + 1e-8)
        
        return {
            'loss': total_loss / len(val_loader),
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'pearson': pearson,
            'spearman': spearman,
            'mean_uncertainty': mean_uncertainty,
            'std_uncertainty': std_uncertainty,
            'uncertainty_correlation': uncertainty_correlation,
            'strong_mae': strong_mae,
            'weak_mae': weak_mae,
            'strong_pearson': strong_pearson,
            'weak_pearson': weak_pearson,
            'strong_precision': strong_precision,
            'strong_recall': strong_recall,
            'weak_precision': weak_precision,
            'weak_recall': weak_recall
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """Full training loop with advanced features."""
        print(f"Training on device: {self.device}")
        if self.use_ema:
            print("Using Exponential Moving Average (EMA) for model weights")
        print(f"Progressive unfreezing: {self.config.get('unfreeze_epoch', 3)}")
        val_freq = self.config.get('val_frequency', 1)  # Validate every N epochs
        print(f"Validation frequency: Every {val_freq} epoch(s)")
        print("-" * 80)
        
        for epoch in range(num_epochs):
            # Progressive unfreezing - earlier for faster convergence
            if self.config.get('progressive_unfreeze', True):
                self.unfreeze_encoders(epoch, self.config.get('unfreeze_epoch', 3))
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate less frequently for speed
            if (epoch + 1) % val_freq == 0 or epoch == 0 or (epoch + 1) == num_epochs:
                val_metrics = self.validate(val_loader, use_ema=True)
            else:
                # Use previous validation metrics or skip
                val_metrics = None
                print(f"\nEpoch {epoch + 1}/{num_epochs} (Skipping validation)")
            
            # Store history (convert numpy types to Python types)
            for key in train_metrics:
                if key in self.history['train']:
                    self.history['train'][key].append(self._convert_metric(train_metrics[key]))
            
            # Store validation history if we validated
            if val_metrics is not None:
                for key in val_metrics:
                    if key in self.history['val']:
                        self.history['val'][key].append(self._convert_metric(val_metrics[key]))
            
            # Print metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}, "
                  f"Pearson: {train_metrics['pearson']:.4f}, "
                  f"Uncertainty: {train_metrics['mean_uncertainty']:.4f}")
            if 'strong_mae' in train_metrics and train_metrics.get('strong_mae', 0) > 0:
                print(f"       Strong(8-12): MAE={train_metrics['strong_mae']:.3f}, "
                      f"Pearson={train_metrics.get('strong_pearson', 0):.3f} | "
                      f"Weak(<5): MAE={train_metrics.get('weak_mae', 0):.3f}, "
                      f"Pearson={train_metrics.get('weak_pearson', 0):.3f}")
            
            if val_metrics is not None:
                print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"MSE: {val_metrics['mse']:.4f}, MAE: {val_metrics['mae']:.4f}, "
                      f"Pearson: {val_metrics['pearson']:.4f}, "
                      f"Uncertainty: {val_metrics['mean_uncertainty']:.4f}, "
                      f"Unc-Corr: {val_metrics['uncertainty_correlation']:.4f}")
                if 'strong_mae' in val_metrics:
                    print(f"       Strong(8-12): MAE={val_metrics['strong_mae']:.3f}, "
                          f"Pearson={val_metrics['strong_pearson']:.3f}, "
                          f"Prec={val_metrics['strong_precision']:.3f}, "
                          f"Rec={val_metrics['strong_recall']:.3f}")
                    print(f"       Weak(<5): MAE={val_metrics['weak_mae']:.3f}, "
                          f"Pearson={val_metrics['weak_pearson']:.3f}, "
                          f"Prec={val_metrics['weak_precision']:.3f}, "
                          f"Rec={val_metrics['weak_recall']:.3f}")
                
                # Validation history already stored above, no need to store again
                
                # Save best model based on multiple criteria (including range-specific)
                improved = False
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    improved = True
                if val_metrics['pearson'] > self.best_val_pearson:
                    self.best_val_pearson = val_metrics['pearson']
                    improved = True
                if val_metrics['mae'] < self.best_val_mae:
                    self.best_val_mae = val_metrics['mae']
                    improved = True
                
                # Also check range-specific improvements
                if 'strong_mae' in val_metrics and val_metrics.get('strong_mae', float('inf')) < getattr(self, 'best_strong_mae', float('inf')):
                    self.best_strong_mae = val_metrics['strong_mae']
                    improved = True
                if 'weak_mae' in val_metrics and val_metrics.get('weak_mae', float('inf')) < getattr(self, 'best_weak_mae', float('inf')):
                    self.best_weak_mae = val_metrics['weak_mae']
                    improved = True
                
                if improved:
                    self.patience_counter = 0
                    os.makedirs('checkpoints', exist_ok=True)
                    
                    # Save with EMA weights if available
                    if self.ema is not None:
                        self.ema.apply_shadow()
                        self.model.save('checkpoints/best_model.pt')
                        self.ema.restore()
                    else:
                        self.model.save('checkpoints/best_model.pt')
                    
                    # Save training history less frequently (every improvement)
                    # Convert numpy types to native Python types for JSON serialization
                    history_path = 'checkpoints/training_history.json'
                    history_serializable = self._convert_to_serializable(self.history)
                    with open(history_path, 'w') as f:
                        json.dump(history_serializable, f, indent=2)
                    
                    print("✓ Saved best model!")
                else:
                    self.patience_counter += 1
            else:
                # Store train metrics only (convert numpy types to Python types)
                for key in train_metrics:
                    if key in self.history['train']:
                        self.history['train'][key].append(self._convert_metric(train_metrics[key]))
            
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping (only check if we validated)
            if val_metrics is not None:
                patience = self.config.get('patience', 10)
                if self.patience_counter >= patience:
                    print(f"\nEarly stopping after {epoch + 1} epochs (patience: {patience})")
                    break
            
            print("-" * 80)
        
        # Final evaluation with EMA
        print("\nFinal evaluation with EMA weights...")
        final_metrics = self.validate(val_loader, use_ema=True)
        print(f"Final Val - Loss: {final_metrics['loss']:.4f}, "
              f"MSE: {final_metrics['mse']:.4f}, MAE: {final_metrics['mae']:.4f}, "
              f"Pearson: {final_metrics['pearson']:.4f}")


def collate_fn(batch):
    """Optimized collate function for DataLoader."""
    # Pre-allocate list for faster appending
    smiles_list = []
    sequence_list = []
    affinity_list = []
    affinity_type_list = []
    
    for item in batch:
        smiles_list.append(item['smiles'])
        sequence_list.append(item['sequence'])
        affinity_list.append(item['affinity'])
        affinity_type_list.append(item['affinity_type'])
    
    # Use tensor constructor for faster creation
    affinity_tensor = torch.tensor(affinity_list, dtype=torch.float32)
    
    return {
        'smiles': smiles_list,
        'sequence': sequence_list,
        'affinity': affinity_tensor,
        'affinity_type': affinity_type_list
    }


def main():
    """Main training script with advanced optimizations."""
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 80)
    print("DTI Model Training - Enhanced Version")
    print("=" * 80)
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Ultra-optimized configuration for <30 minute training with better accuracy
    # Auto-detect optimal batch size based on GPU memory
    if device == 'cuda':
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb >= 16:
            batch_size = 48  # Very large batch for speed
            num_workers = 8
        elif gpu_memory_gb >= 8:
            batch_size = 32  # Large batch
            num_workers = 6
        else:
            batch_size = 24  # Medium batch (increased from 16)
            num_workers = 4
    else:
        batch_size = 16  # CPU
        num_workers = 2
    
    # Force compile_model to False on Windows
    is_windows = platform.system() == 'Windows'
    
    config = {
        # Data settings - ultra-optimized for speed
        'batch_size': batch_size,
        'gradient_accumulation_steps': 1,  # No accumulation needed with larger batches
        'num_workers': num_workers,
        'prefetch_factor': 6,  # More prefetching for speed
        
        # Training settings - optimized for <30 minutes
        'num_epochs': 10,  # Reduced epochs but more effective
        'patience': 6,  # Faster early stopping
        'val_frequency': 1,  # Validate every epoch for better monitoring
        
        # Learning rates - aggressive for fast convergence
        'learning_rate': 5e-4,  # Higher LR for faster learning
        'encoder_lr': 2e-5,  # Higher encoder LR for faster fine-tuning
        'weight_decay': 0.01,
        'warmup_steps': 100,  # Very fast warmup
        
        # Loss settings - range-focused for accuracy
        'use_range_focused_loss': True,  # Focus on strong (8-12) and weak (<5)
        'strong_weight': 3.0,  # High weight for strong binding (8-12)
        'weak_weight': 3.0,  # High weight for weak binding (<5)
        'evidential_coef': 0.15,  # Slightly higher for better uncertainty
        'lambda_kl': 0.01,
        'annealing_epochs': 5,  # Fast annealing
        
        # Optimization
        'use_amp': True,
        'max_grad_norm': 1.0,
        'compile_model': False,  # Disabled on Windows
        'use_ema': True,
        'ema_decay': 0.9995,  # Slightly faster EMA updates
        'ema_update_freq': 1,  # Update every step for better accuracy
        
        # Progressive unfreezing - very early for fast convergence
        'progressive_unfreeze': True,
        'unfreeze_epoch': 2,  # Unfreeze encoders very early (epoch 2)
        
        # Device
        'device': device,
    }
    
    print(f"\n🚀 ULTRA-OPTIMIZED CONFIGURATION (<30 min training):")
    print(f"  Batch size: {config['batch_size']} (effective: {config['batch_size'] * config['gradient_accumulation_steps']})")
    print(f"  Workers: {config['num_workers']}")
    print(f"  Epochs: {config['num_epochs']} (optimized for speed)")
    print(f"  Validation frequency: Every {config['val_frequency']} epoch(s)")
    print(f"  Range-focused loss: {config.get('use_range_focused_loss', False)}")
    if config.get('use_range_focused_loss', False):
        print(f"    - Strong binding (8-12) weight: {config.get('strong_weight', 2.0)}x")
        print(f"    - Weak binding (<5) weight: {config.get('weak_weight', 2.0)}x")
    print(f"  Model compilation: {config['compile_model']}")
    
    # Estimate training time (will be faster with larger batches)
    estimated_time_per_epoch = 6.0  # minutes (optimistic with larger batches)
    total_estimated = estimated_time_per_epoch * config['num_epochs']
    print(f"\n⏱️  Estimated training time: ~{total_estimated:.1f} minutes")
    if total_estimated <= 30:
        print(f"✅ Target: <30 minutes!")
    else:
        print(f"⚠️  May be ~{total_estimated:.0f} minutes, but optimized for best accuracy")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = DTIDataset('data/processed/train.csv')
    val_dataset = DTIDataset('data/processed/val.csv')
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    
    # Calculate steps per epoch for scheduler
    effective_batch_size = config['batch_size'] * config['gradient_accumulation_steps']
    config['steps_per_epoch'] = len(train_dataset) // effective_batch_size
    print(f"Steps per epoch: {config['steps_per_epoch']}")
    print(f"Effective batch size: {effective_batch_size}")
    
    # Optimized data loaders for speed
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if config['num_workers'] > 0 else False,
        prefetch_factor=config.get('prefetch_factor', 4) if config['num_workers'] > 0 else None,
        drop_last=True  # For consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,  # Larger batch for faster validation
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if config['num_workers'] > 0 else False,
        prefetch_factor=config.get('prefetch_factor', 4) if config['num_workers'] > 0 else None
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = DTIModel()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    
    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    trainer = ImprovedTrainer(model, config)
    trainer.train(train_loader, val_loader, config['num_epochs'])
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"Best model saved to: checkpoints/best_model.pt")
    print(f"Training history saved to: checkpoints/training_history.json")
    print(f"\nBest metrics:")
    print(f"  - Loss: {trainer.best_val_loss:.4f}")
    print(f"  - Pearson: {trainer.best_val_pearson:.4f}")
    print(f"  - MAE: {trainer.best_val_mae:.4f}")
    if hasattr(trainer, 'best_strong_mae') and trainer.best_strong_mae < float('inf'):
        print(f"  - Strong binding (8-12) MAE: {trainer.best_strong_mae:.4f}")
    if hasattr(trainer, 'best_weak_mae') and trainer.best_weak_mae < float('inf'):
        print(f"  - Weak binding (<5) MAE: {trainer.best_weak_mae:.4f}")


if __name__ == '__main__':
    main()
