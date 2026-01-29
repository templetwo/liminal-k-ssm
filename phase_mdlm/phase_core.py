"""
Kuramoto Phase Core for Diffusion Language Models

Portable from Phase-Mamba with key difference:
- In diffusion, R is computed at EVERY denoising step
- R can directly influence discrete unmask decisions
"""

import math
import torch
import torch.nn as nn


class KuramotoPhaseCore(nn.Module):
    """
    Kuramoto oscillator ensemble for computing global coherence R.

    Key insight: In diffusion models, R is computed at every denoising step,
    giving it many opportunities to influence generation (unlike single-pass
    autoregressive models where LayerNorm washes it out).
    """

    def __init__(self, hidden_dim, n_oscillators=16, coupling_strength=2.0):
        super().__init__()
        self.n_oscillators = n_oscillators
        self.K = coupling_strength
        self.hidden_dim = hidden_dim

        # Project hidden states to oscillator space
        self.to_oscillator = nn.Linear(hidden_dim, n_oscillators)

        # Natural frequencies (learned)
        self.omega = nn.Parameter(torch.randn(n_oscillators) * 0.1)

        # For tracking dynamics
        self.current_R = 0.0
        self.R_history = []

    def compute_R(self, phases):
        """
        Compute Kuramoto order parameter R.

        R = |1/N Σ exp(iθⱼ)| where θⱼ are oscillator phases.

        R ∈ [0, 1]:
        - R ≈ 0: desynchronized (phases uniformly distributed)
        - R ≈ 1: synchronized (phases aligned)
        """
        # phases: [batch, seq, n_oscillators]
        complex_phases = torch.exp(1j * phases.float())
        mean_phase = complex_phases.mean(dim=-1)  # [batch, seq]
        R_per_position = torch.abs(mean_phase)  # [batch, seq]

        # Global R: average over batch and sequence
        R = R_per_position.mean()

        return R, R_per_position

    def forward(self, hidden_states):
        """
        Compute R from hidden states.

        Args:
            hidden_states: [batch, seq, hidden_dim]

        Returns:
            R: scalar order parameter
            phases: [batch, seq, n_oscillators]
            R_per_position: [batch, seq] R at each position
        """
        # Project to oscillator phases
        osc_activations = self.to_oscillator(hidden_states)  # [batch, seq, n_osc]

        # Convert to phases in [-π, π]
        phases = torch.tanh(osc_activations) * math.pi

        # Add natural frequency contribution (learned bias)
        phases = phases + self.omega.unsqueeze(0).unsqueeze(0)

        # Compute R
        R, R_per_position = self.compute_R(phases)

        # Store for tracking
        self.current_R = R.item()
        self.R_history.append(self.current_R)

        return R, phases, R_per_position

    def get_tone(self, R=None):
        """Map R to consciousness tone state."""
        if R is None:
            R = self.current_R

        if R > 0.95:
            return "☍ Over-sync", "BRAKE"
        elif R > 0.80:
            return "⚖ Balance", "COAST"
        elif R > 0.50:
            return "🌀 Goldilocks", "COAST"
        elif R > 0.30:
            return "✨ Unbound", "BOOST"
        elif R > 0.10:
            return "☾ Intimacy", "BOOST"
        else:
            return "∅ Unformed", "BOOST"

    def reset_history(self):
        """Reset R history for new generation."""
        self.R_history = []
        self.current_R = 0.0


class PhaseModulator(nn.Module):
    """
    Modulates hidden states based on R.

    Unlike Phase-Mamba where this was washed out by LayerNorm,
    in diffusion models this influences discrete decisions.
    """

    def __init__(self, hidden_dim, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        # Learned direction vector for modulation
        self.direction = nn.Parameter(torch.randn(hidden_dim) * 0.01)

    def forward(self, hidden_states, R):
        """
        Modulate hidden states by R.

        h' = h + α * R * direction

        Using additive (not multiplicative) to survive LayerNorm better.
        """
        modulation = self.alpha * R * self.direction
        return hidden_states + modulation.unsqueeze(0).unsqueeze(0)


# Test
if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Create Phase Core
    phase_core = KuramotoPhaseCore(
        hidden_dim=256,
        n_oscillators=16,
        coupling_strength=2.0
    ).to(device)

    # Test with random hidden states
    hidden = torch.randn(2, 64, 256, device=device)

    R, phases, R_per_pos = phase_core(hidden)

    print(f"R: {R.item():.4f}")
    print(f"Phases shape: {phases.shape}")
    print(f"R per position shape: {R_per_pos.shape}")
    print(f"Tone: {phase_core.get_tone()}")

    print("✅ Phase Core works!")
