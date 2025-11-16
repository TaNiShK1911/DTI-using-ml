import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EvidentialRegressionHead(nn.Module):
    """Evidential regression for uncertainty quantification."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [512, 256]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output 4 evidential parameters
        self.output = nn.Linear(prev_dim, 4)
    
    def forward(self, fused_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            affinity_pred: Predicted affinity (gamma)
            uncertainty: Total uncertainty
            evidential_params: [gamma, nu, alpha, beta]
        """
        features = self.mlp(fused_features)
        params = self.output(features)
        
        # Split parameters
        gamma = params[:, 0]  # Mean (affinity prediction)
        nu = F.softplus(params[:, 1]) + 1  # Degrees of freedom (>0)
        alpha = F.softplus(params[:, 2]) + 1  # Shape (>0)
        beta = F.softplus(params[:, 3])  # Scale (>0)
        
        # Compute uncertainty
        # Epistemic uncertainty (from alpha, beta)
        epistemic = beta / (alpha - 1 + 1e-8)
        # Aleatoric uncertainty (from nu)
        aleatoric = beta / (nu * (alpha - 1) + 1e-8)
        # Total uncertainty
        uncertainty = epistemic + aleatoric
        
        evidential_params = torch.stack([gamma, nu, alpha, beta], dim=-1)
        
        return gamma, uncertainty, evidential_params


class EvidentialLoss(nn.Module):
    """Improved evidential regression loss function with better uncertainty calibration."""
    
    def __init__(self, coef: float = 0.1, lambda_kl: float = 0.01, annealing_epochs: int = 10):
        super().__init__()
        self.coef = coef
        self.lambda_kl = lambda_kl  # Separate weight for KL term
        self.annealing_epochs = annealing_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Update current epoch for annealing."""
        self.current_epoch = epoch
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                evidential_params: torch.Tensor) -> torch.Tensor:
        """
        Improved evidential loss with proper uncertainty calibration.
        
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
        
        # Primary prediction loss - use Huber loss for robustness
        error = targets - gamma
        huber_loss = F.smooth_l1_loss(gamma, targets, beta=1.0)
        
        # Negative log-likelihood for evidential learning
        # Using Student-t distribution likelihood
        error_sq = error ** 2
        nll = 0.5 * torch.log(torch.pi / nu) \
              - alpha * torch.log(2 * beta) \
              + (alpha + 0.5) * torch.log(nu * error_sq + 2 * beta) \
              + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        
        # KL divergence to regularize uncertainty (prevent overconfidence)
        # Prior: uniform distribution (high uncertainty)
        # Simplified KL term that encourages higher uncertainty when error is large
        kl = torch.abs(error) * (2 * nu + alpha) / torch.clamp(alpha - 1, min=1e-6)
        
        # Annealing: gradually increase evidential loss weight
        annealing_factor = min(1.0, self.current_epoch / self.annealing_epochs)
        
        # Combined loss with better weighting
        # Primary focus on prediction accuracy, then uncertainty calibration
        loss = huber_loss + self.coef * annealing_factor * torch.mean(nll) + \
               self.lambda_kl * annealing_factor * torch.mean(kl)
        
        return loss
