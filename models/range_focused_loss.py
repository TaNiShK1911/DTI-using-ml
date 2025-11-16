import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RangeFocusedEvidentialLoss(nn.Module):
    """
    Evidential loss with focus on important binding affinity ranges:
    - Strong binding: 8-12 (high priority)
    - Weak binding: <5 (high priority)
    """
    
    def __init__(self, coef: float = 0.1, lambda_kl: float = 0.01, 
                 strong_weight: float = 2.0, weak_weight: float = 2.0,
                 annealing_epochs: int = 5):
        super().__init__()
        self.coef = coef
        self.lambda_kl = lambda_kl
        self.strong_weight = strong_weight  # Weight for 8-12 range
        self.weak_weight = weak_weight  # Weight for <5 range
        self.annealing_epochs = annealing_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Update current epoch for annealing."""
        self.current_epoch = epoch
    
    def _get_sample_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """Get sample weights based on affinity ranges."""
        weights = torch.ones_like(targets)
        
        # Strong binding range (8-12): high weight
        strong_mask = (targets >= 8.0) & (targets <= 12.0)
        weights[strong_mask] = self.strong_weight
        
        # Weak binding range (<5): high weight
        weak_mask = targets < 5.0
        weights[weak_mask] = self.weak_weight
        
        return weights
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                evidential_params: torch.Tensor) -> torch.Tensor:
        """
        Range-focused evidential loss.
        
        Args:
            predictions: Predicted affinity values (gamma)
            targets: True affinity values
            evidential_params: [gamma, nu, alpha, beta]
        """
        gamma = evidential_params[:, 0]
        nu = evidential_params[:, 1]
        alpha = evidential_params[:, 2]
        beta = evidential_params[:, 3]
        
        # Ensure parameters are in valid ranges
        nu = torch.clamp(nu, min=1.0)
        alpha = torch.clamp(alpha, min=1.0)
        beta = torch.clamp(beta, min=1e-6)
        
        # Get sample weights for important ranges
        sample_weights = self._get_sample_weights(targets)
        
        # Primary prediction loss - weighted MSE for important ranges
        error = targets - gamma
        mse_loss = (error ** 2) * sample_weights
        weighted_mse = torch.mean(mse_loss)
        
        # Also use Huber loss for robustness
        huber_loss = F.smooth_l1_loss(gamma, targets, reduction='none', beta=1.0)
        weighted_huber = torch.mean(huber_loss * sample_weights)
        
        # Combine MSE and Huber (Huber is more robust to outliers)
        prediction_loss = 0.7 * weighted_huber + 0.3 * weighted_mse
        
        # Negative log-likelihood for evidential learning
        error_sq = error ** 2
        nll = 0.5 * torch.log(torch.pi / nu) \
              - alpha * torch.log(2 * beta) \
              + (alpha + 0.5) * torch.log(nu * error_sq + 2 * beta) \
              + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        
        # Weight NLL by sample importance
        weighted_nll = torch.mean(nll * sample_weights)
        
        # KL divergence to regularize uncertainty
        kl = torch.abs(error) * (2 * nu + alpha) / torch.clamp(alpha - 1, min=1e-6)
        weighted_kl = torch.mean(kl * sample_weights)
        
        # Annealing: gradually increase evidential loss weight
        annealing_factor = min(1.0, self.current_epoch / self.annealing_epochs)
        
        # Combined loss with range focus
        loss = prediction_loss + self.coef * annealing_factor * weighted_nll + \
               self.lambda_kl * annealing_factor * weighted_kl
        
        return loss

