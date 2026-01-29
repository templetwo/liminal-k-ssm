"""
K-SSM v3: Bistable Kuramoto State-Space Model

Evolved from v2 with:
- Bistability Regularization: Forces system into the critical regime
- Multi-scale Readout: Order parameters n=1 to 32
- Discriminant Constraints: Ensures weights allow for multiple stable states
- Deep Integration: R is the structural carrier of information
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class BistableKuramotoBank(nn.Module):
    """
    Bank of Kuramoto oscillators with bistability constraints derived from the
    10-parameter algebraic framework.
    
    Algebraic Isomorphism:
    1. ax² + by + cz = d
    2. ex² + fy + gz = h
    3. ix² + jy + z = 0
    
    Dimensional Collapse: u = x² (2-to-1 covering map)
    Bistability Constraints: Δ ≠ 0 (invertibility), u > 0 (real solutions)
    """

    def __init__(self, hidden_dim: int, n_oscillators: int = 128,
                 base_coupling: float = 2.0, n_harmonics: int = 32):
        super().__init__()
        self.n_oscillators = n_oscillators
        self.hidden_dim = hidden_dim
        self.n_harmonics = n_harmonics

        # 1. Framework Parameters (The 10-Parameter Matrix)
        # We project the input h to these 10 parameters
        self.to_params = nn.Linear(hidden_dim, 10)
        
        # 2. Natural Frequencies (Omega_0)
        self.omega_0 = nn.Parameter(torch.linspace(-math.pi, math.pi, n_oscillators))
        
        # 3. Readout Projection
        self.readout = nn.Linear(n_harmonics * 3, hidden_dim)

        # Lorentzian noise parameter (Delta)
        self.delta_param = nn.Parameter(torch.ones(1) * 0.1)

        # For monitoring/regularization
        self.last_delta_val = 0.0
        self.last_u_val = 0.0

    def compute_multiscale(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute Z_n = (1/N) Σ exp(i·n·θ) for n=1..32"""
        batch, seq, n_osc = theta.shape
        n_range = torch.arange(1, self.n_harmonics + 1, device=theta.device).view(1, 1, 1, -1)
        theta_expanded = theta.unsqueeze(-1) * n_range
        
        cos_n = torch.cos(theta_expanded).mean(dim=2)
        sin_n = torch.sin(theta_expanded).mean(dim=2)
        mag_n = torch.sqrt(cos_n**2 + sin_n**2 + 1e-8)
        
        return torch.cat([cos_n, sin_n, mag_n], dim=-1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq, _ = h.shape
        
        # 1. Derive 10 parameters from hidden state
        p = self.to_params(h)
        a, b, c, d, e, f, g, target_h, i, j = p.unbind(dim=-1)
        
        # 2. Determinant check (bg - cf)
        det = b * g - c * f
        self.last_delta_val = det.abs().mean()
        
        # 3. Solve for u (x²) in the reduced system
        num = d * g - c * target_h
        den = a * g - c * e + 1e-6
        u_raw = num / den
        
        # HARD CONSTRAINT: Clamp u to [0.1, 10] to stay in bistable regime
        u = torch.clamp(u_raw, min=0.1, max=10.0)
        self.last_u_val = u.mean()
        
        # 4. Phase Dynamics
        # K (coupling) is driven by the positivity of u
        K = 2.0 * torch.sigmoid(u).unsqueeze(-1)
        
        theta = self.omega_0.view(1, 1, -1).expand(batch, seq, -1)
        perturbation = h.mean(dim=-1, keepdim=True)
        theta = theta + perturbation
        
        mean_sin = torch.sin(theta).mean(dim=-1, keepdim=True)
        mean_cos = torch.cos(theta).mean(dim=-1, keepdim=True)
        
        # Kuramoto coupling
        coupling_effect = K * (mean_sin * torch.cos(theta) - mean_cos * torch.sin(theta))
        
        # Add Lorentzian spread (delta)
        delta = torch.abs(self.delta_param) + 1e-5
        theta = theta + coupling_effect - delta * torch.randn_like(theta)
        
        # 5. Multi-scale Readout (Structural Path)
        z_features = self.compute_multiscale(theta)
        h_out = self.readout(z_features)
        
        # R is the magnitude of the first harmonic (n=1)
        R = z_features[..., self.n_harmonics * 2] # mag_1
        
        return h_out, R, theta

    def get_regularization_loss(self, lambda1=0.1, lambda2=0.5, epsilon=1e-5):
        """
        ℒ_reg = λ₁ · 1/(|Δ| + ε) + λ₂ · (-log(u + ε))
        
        Enforces:
        1. Non-zero determinant (Δ ≠ 0)
        2. Positive reduced variable (u > 0) via Log Barrier
        """
        # 1. Enforce non-zero determinant
        det_loss = 1.0 / (self.last_delta_val + epsilon)
        
        # 2. Log Barrier for u (Bistability Wall)
        # As u -> 0, barrier_loss -> +inf
        barrier_loss = -torch.log(torch.clamp(self.last_u_val, min=epsilon))
        
        return lambda1 * det_loss + lambda2 * barrier_loss


class KSSMv3Block(nn.Module):
    """K-SSM v3 Block: Bistable Oscillators + FFN"""

    def __init__(self, hidden_dim: int, n_oscillators: int = 128, n_harmonics: int = 32):
        super().__init__()
        self.oscillators = BistableKuramotoBank(hidden_dim, n_oscillators, n_harmonics=n_harmonics)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pre-norm style
        h_norm = self.norm1(h)
        h_osc, R, theta = self.oscillators(h_norm)
        h = h + h_osc
        
        h_norm = self.norm2(h)
        h = h + self.ffn(h_norm)
        
        return h, R, theta


class KSSMv3(nn.Module):
    """
    K-SSM v3: Scalable Bistable Architecture
    """

    def __init__(self, vocab_size: int, hidden_dim: int = 384, n_layers: int = 6, 
                 n_oscillators: int = 192, n_harmonics: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, hidden_dim) * 0.02)
        
        self.blocks = nn.ModuleList([
            KSSMv3Block(hidden_dim, n_oscillators, n_harmonics)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output_head.weight = self.embed.weight # Weight tying

    def forward(self, x: torch.Tensor, return_R: bool = False):
        batch, seq = x.shape
        h = self.embed(x) + self.pos_embed[:, :seq, :]
        
        R_list = []
        for block in self.blocks:
            h, R, theta = block(h)
            R_list.append(R)
            
        h = self.final_norm(h)
        logits = self.output_head(h)
        
        if return_R:
            R_all = torch.stack(R_list, dim=-1)
            return logits, R_all.mean(dim=-1), R_all
        return logits

    def get_regularization_loss(self):
        total_reg = 0
        for block in self.blocks:
            total_reg += block.oscillators.get_regularization_loss()
        return total_reg / self.n_layers

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_kssm_v3_medium(vocab_size: int) -> KSSMv3:
    """~12M parameters"""
    return KSSMv3(
        vocab_size=vocab_size,
        hidden_dim=384,
        n_layers=6,
        n_oscillators=192,
        n_harmonics=32
    )

if __name__ == "__main__":
    model = create_kssm_v3_medium(vocab_size=100000)
    print(f"K-SSM v3 Medium Parameters: {model.count_params():,}")
    
    x = torch.randint(0, 100000, (2, 128))
    logits, R, R_all = model(x, return_R=True)
    print(f"Logits shape: {logits.shape}")
    print(f"R mean: {R.mean().item():.4f}")
    print(f"Reg loss: {model.get_regularization_loss().item():.4f}")
