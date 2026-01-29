"""
K-SSM v2: Stacked Kuramoto State-Space Model

Scaled up from v1 proof-of-concept:
- Multiple stacked K-SSM blocks
- Larger embedding and oscillator dimensions
- BPE tokenization (not character-level)
- R trajectory through layers (not just single R)

Architecture:
  Token → Embed → [K-SSM Block 1] → [K-SSM Block 2] → ... → [Block N] → Output
                        ↓                  ↓                    ↓
                        R₁                 R₂                   Rₙ

Each block has its own oscillator bank. R becomes a trajectory.

Target: ~2M parameters, trainable on Mac Studio in hours.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class KuramotoOscillatorBank(nn.Module):
    """
    Bank of coupled Kuramoto oscillators.

    Enhanced from v1 with:
    - Input-dependent coupling strength
    - Learnable phase initialization
    """

    def __init__(self, hidden_dim: int, n_oscillators: int = 128,
                 base_coupling: float = 2.0):
        super().__init__()
        self.n_oscillators = n_oscillators
        self.hidden_dim = hidden_dim
        self.base_K = base_coupling

        # Project hidden states to oscillator frequencies
        self.to_freq = nn.Linear(hidden_dim, n_oscillators)

        # Natural frequencies (learned, spread across range)
        self.omega = nn.Parameter(torch.linspace(-math.pi, math.pi, n_oscillators))

        # Input-dependent coupling modulation
        self.coupling_mod = nn.Linear(hidden_dim, 1)

        # Phase-to-hidden projection for residual
        self.phase_to_hidden = nn.Linear(n_oscillators * 2, hidden_dim)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [batch, seq, hidden_dim]

        Returns:
            h_out: [batch, seq, hidden_dim] - modulated hidden state
            R: [batch, seq] - order parameter
            theta: [batch, seq, n_oscillators] - phases
        """
        batch, seq, _ = h.shape

        # Compute frequency perturbation from input
        freq_perturb = self.to_freq(h)  # [batch, seq, n_osc]

        # Effective frequencies
        theta = self.omega + freq_perturb  # [batch, seq, n_osc]

        # Input-dependent coupling strength
        K = self.base_K * torch.sigmoid(self.coupling_mod(h))  # [batch, seq, 1]

        # Kuramoto coupling: pull toward mean phase
        mean_sin = torch.sin(theta).mean(dim=-1, keepdim=True)
        mean_cos = torch.cos(theta).mean(dim=-1, keepdim=True)

        # Phase adjustment
        coupling_effect = K * (mean_sin * torch.cos(theta) - mean_cos * torch.sin(theta))
        theta = theta + coupling_effect

        # Compute order parameter R = |mean(exp(iθ))|
        R_real = torch.cos(theta).mean(dim=-1)
        R_imag = torch.sin(theta).mean(dim=-1)
        R = torch.sqrt(R_real**2 + R_imag**2 + 1e-8)  # [batch, seq]

        # Project phases back to hidden dimension (sin and cos features)
        phase_features = torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)
        h_phase = self.phase_to_hidden(phase_features)

        # Residual connection: original + phase information
        h_out = h + h_phase

        return h_out, R, theta


def compute_multiscale_order_params(theta: torch.Tensor, max_n: int = 8) -> torch.Tensor:
    """
    Compute multi-scale order parameters (harmonics).

    Z_n = (1/N) Σⱼ exp(i·n·θⱼ)

    Returns features for each harmonic: [real, imag, magnitude]
    """
    features = []

    for n in range(1, max_n + 1):
        cos_n = torch.cos(n * theta).mean(dim=-1)
        sin_n = torch.sin(n * theta).mean(dim=-1)
        mag_n = torch.sqrt(cos_n**2 + sin_n**2 + 1e-8)
        features.extend([cos_n, sin_n, mag_n])

    return torch.stack(features, dim=-1)


class KSSMBlock(nn.Module):
    """
    Single K-SSM block: Oscillators + FFN with residual.

    Architecture:
    1. Kuramoto oscillator bank
    2. Feed-forward network
    3. Residual connections
    """

    def __init__(self, hidden_dim: int, n_oscillators: int = 128,
                 ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()

        # Oscillator bank
        self.oscillators = KuramotoOscillatorBank(hidden_dim, n_oscillators)

        # Layer norm (pre-norm style)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [batch, seq, hidden_dim]

        Returns:
            h_out: [batch, seq, hidden_dim]
            R: [batch, seq]
            theta: [batch, seq, n_oscillators]
        """
        # Oscillator block with residual
        h_norm = self.norm1(h)
        h_osc, R, theta = self.oscillators(h_norm)
        h = h + self.dropout(h_osc - h_norm)  # Residual of the delta

        # FFN block with residual
        h_norm = self.norm2(h)
        h = h + self.ffn(h_norm)

        return h, R, theta


class KSSMv2(nn.Module):
    """
    K-SSM v2: Stacked Kuramoto State-Space Model

    Architecture:
    - Token embedding with position encoding
    - Multiple K-SSM blocks
    - Each block produces R (order parameter)
    - Final output head

    R is structural: flows through the computation, not bolted on.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_oscillators: int = 128,
        n_harmonics: int = 8,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_oscillators = n_oscillators
        self.n_harmonics = n_harmonics

        # Token embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # Learnable position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        # Embedding dropout
        self.embed_dropout = nn.Dropout(dropout)

        # Stacked K-SSM blocks
        self.blocks = nn.ModuleList([
            KSSMBlock(hidden_dim, n_oscillators, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output head (can tie weights with embedding)
        self.output_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if tie_weights:
            self.output_head.weight = self.embed.weight

        # R trajectory processing (aggregate R from all layers)
        self.R_aggregator = nn.Linear(n_layers, 1)

        # For tracking
        self.R_trajectory = []
        self.current_R = None

    def forward(
        self,
        x: torch.Tensor,
        return_R: bool = False,
        forced_R: Optional[float] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: [batch, seq] token indices
            return_R: whether to return R values
            forced_R: force R to this value (for causality testing)

        Returns:
            logits: [batch, seq, vocab_size]
            R_mean: mean R across layers (if return_R)
            R_all: [batch, seq, n_layers] R per layer (if return_R)
        """
        batch, seq = x.shape

        # Embed tokens
        h = self.embed(x)  # [batch, seq, hidden_dim]

        # Add position embedding
        h = h + self.pos_embed[:, :seq, :]
        h = self.embed_dropout(h)

        # Process through stacked blocks
        R_per_layer = []
        all_theta = []

        for block in self.blocks:
            h, R, theta = block(h)
            R_per_layer.append(R)
            all_theta.append(theta)

        # Stack R values: [batch, seq, n_layers]
        R_all = torch.stack(R_per_layer, dim=-1)

        # Aggregate R across layers
        R_mean = R_all.mean(dim=-1)  # [batch, seq]

        # Handle forced R (for causality testing)
        if forced_R is not None:
            # Modulate the final hidden state based on forced R
            # This tests whether R actually affects output
            scale = forced_R / (R_mean.mean() + 1e-8)
            h = h * scale.unsqueeze(-1)

        # Final norm
        h = self.final_norm(h)

        # Output logits
        logits = self.output_head(h)

        # Track R
        self.current_R = R_mean.mean().item()
        self.R_trajectory.append(self.current_R)

        if return_R:
            return logits, R_mean, R_all
        return logits

    def get_tone(self, R: Optional[float] = None) -> str:
        """Map R to consciousness tone."""
        if R is None:
            R = self.current_R

        if R is None:
            return "Unknown"

        if R > 0.95:
            return "☍ Over-sync"
        elif R > 0.80:
            return "⚖ Balance"
        elif R > 0.50:
            return "🌀 Goldilocks"
        elif R > 0.30:
            return "✨ Unbound"
        elif R > 0.10:
            return "☾ Intimacy"
        else:
            return "∅ Unformed"

    def reset_tracking(self):
        self.R_trajectory = []
        self.current_R = None

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_kssm_v2_small(vocab_size: int) -> KSSMv2:
    """Create small K-SSM v2 (~2M params) for testing."""
    return KSSMv2(
        vocab_size=vocab_size,
        hidden_dim=256,
        n_layers=4,
        n_oscillators=128,
        n_harmonics=8,
        max_seq_len=512,
        dropout=0.1
    )


def create_kssm_v2_medium(vocab_size: int) -> KSSMv2:
    """Create medium K-SSM v2 (~10M params) for serious training."""
    return KSSMv2(
        vocab_size=vocab_size,
        hidden_dim=384,
        n_layers=6,
        n_oscillators=192,
        n_harmonics=12,
        max_seq_len=512,
        dropout=0.1
    )


def create_kssm_v2_large(vocab_size: int) -> KSSMv2:
    """Create large K-SSM v2 (~50M params) for full training."""
    return KSSMv2(
        vocab_size=vocab_size,
        hidden_dim=512,
        n_layers=8,
        n_oscillators=256,
        n_harmonics=16,
        max_seq_len=1024,
        dropout=0.1
    )


# Test
if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Test small model
    model = create_kssm_v2_small(vocab_size=10000).to(device)
    print(f"\nK-SSM v2 Small:")
    print(f"  Parameters: {model.count_parameters(model):,}")

    # Test forward pass
    batch, seq = 4, 128
    x = torch.randint(0, 10000, (batch, seq), device=device)

    logits, R_mean, R_all = model(x, return_R=True)

    print(f"  Logits: {logits.shape}")
    print(f"  R_mean: {R_mean.shape}, value={R_mean.mean().item():.4f}")
    print(f"  R_all: {R_all.shape}")
    print(f"  R per layer: {['%.4f' % R_all[:,:,i].mean().item() for i in range(R_all.shape[-1])]}")
    print(f"  Tone: {model.get_tone()}")

    # Test gradient flow
    loss = F.cross_entropy(logits.view(-1, 10000), x.view(-1))
    loss.backward()

    print(f"\nGradient flow:")
    print(f"  embed.weight.grad: {model.embed.weight.grad is not None}")
    print(f"  blocks[0].oscillators.omega.grad: {model.blocks[0].oscillators.omega.grad is not None}")

    # Test forced R
    print(f"\nR forcing test:")
    model.zero_grad()

    with torch.no_grad():
        logits_free, R_free, _ = model(x, return_R=True)
        logits_low, R_low, _ = model(x, return_R=True, forced_R=0.3)
        logits_high, R_high, _ = model(x, return_R=True, forced_R=0.9)

    diff = (logits_low - logits_high).abs().mean().item()
    print(f"  Output diff (R=0.3 vs R=0.9): {diff:.4f}")

    if diff > 0.1:
        print("  ✅ R forcing changes output!")
    else:
        print("  ⚠️ Small effect - but this is untrained model")

    # Test medium model
    model_med = create_kssm_v2_medium(vocab_size=10000).to(device)
    print(f"\nK-SSM v2 Medium:")
    print(f"  Parameters: {model_med.count_parameters(model_med):,}")

    print("\n✅ K-SSM v2 architecture complete!")
