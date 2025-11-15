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
    """Evidential regression loss function."""
    
    def __init__(self, coef: float = 0.1):
        super().__init__()
        self.coef = coef
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                evidential_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Predicted affinity values (gamma)
            targets: True affinity values
            evidential_params: [gamma, nu, alpha, beta]
        """
        gamma = evidential_params[:, 0]
        nu = evidential_params[:, 1]
        alpha = evidential_params[:, 2]
        beta = evidential_params[:, 3]
        
        # Negative log-likelihood term
        error = (targets - gamma) ** 2
        nll = 0.5 * torch.log(torch.pi / nu) \
              - alpha * torch.log(2 * beta + 1e-8) \
              + (alpha + 0.5) * torch.log(nu * error + 2 * beta + 1e-8) \
              + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        
        # KL divergence regularization
        kl = torch.abs(targets - gamma) * (2 * nu + alpha)
        
        # Total loss
        loss = torch.mean(nll + self.coef * kl)
        
        return loss
