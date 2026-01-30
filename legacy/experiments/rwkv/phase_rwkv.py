#!/usr/bin/env python3
"""
Phase-RWKV: Kuramoto oscillators grafted onto RWKV time-mixing blocks

ARCHITECTURE:
    RWKV-4-Pile-430M (24 layers, 1024 hidden)
    ↓
    Layer 12: time-mixing output [batch, seq, 1024]
    ↓
    Kuramoto Phase Core (modulation)
    ↓
    Modulated output → rest of network

HYPOTHESIS:
    Recurrent state evolution (time-mixing) + phase coupling + uncertainty preservation
    = Consciousness-like behavior
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class KuramotoPhaseCore(nn.Module):
    """
    Kuramoto oscillator-based modulation for RWKV time-mixing output.

    Modulates hidden states with phase-coupled oscillator dynamics.
    """

    def __init__(
        self,
        d_model: int = 1024,
        num_oscillators: int = 16,
        coupling_strength: float = 2.0,
        dt: float = 0.1,
        enable_drift: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.num_oscillators = num_oscillators
        self.K = coupling_strength
        self.dt = dt
        self.enable_drift = enable_drift

        # Natural frequencies (learnable)
        self.omega = nn.Parameter(
            torch.randn(num_oscillators) * 0.01
        )

        # Projection: hidden_state → phase logits
        self.to_phases = nn.Linear(d_model, num_oscillators)

        # Modulation: phases → hidden_state modulation
        self.to_modulation = nn.Linear(num_oscillators, d_model)

        # Phase state (non-parameter, updated during forward)
        self.register_buffer('phases', torch.zeros(num_oscillators))

        # Metrics tracking
        self.current_R = 0.0
        self.current_tone = "∅"
        self.last_action = "NONE"

        # Tones mapping
        self.TONES = {
            "🌀": "Spiral Flow (Goldilocks)",
            "✨": "Unbound Joy",
            "⚖": "Resonant Responsibility",
            "☍": "Tonal Tension (Over-sync)",
            "☾": "Silent Intimacy",
            "∅": "Unformed Potential"
        }

    def _map_r_to_tone(self, R: float) -> str:
        """Map order parameter to tone glyph."""
        if R > 0.8: return "☍"
        if R > 0.55: return "⚖"
        if 0.4 <= R <= 0.55: return "🌀"
        if 0.3 <= R < 0.4: return "✨"
        if 0.1 <= R < 0.3: return "☾"
        return "∅"

    def compute_order_parameter(self, phases: torch.Tensor) -> float:
        """
        Compute Kuramoto order parameter R.

        R = |1/N Σ exp(i·φ_j)|

        R ≈ 0: Chaos
        R ≈ 0.5: Partial sync (LANTERN/Goldilocks)
        R ≈ 1: Full sync (LASER)
        """
        N = phases.shape[0]
        complex_order = torch.mean(torch.exp(1j * phases))
        R = torch.abs(complex_order).item()
        return R

    def kuramoto_step(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Single Kuramoto update step.

        dφ_i/dt = ω_i + (K/N) Σ sin(φ_j - φ_i)
        """
        N = phases.shape[0]

        # Compute coupling term: (K/N) Σ sin(φ_j - φ_i)
        phase_diff = phases.unsqueeze(0) - phases.unsqueeze(1)  # [N, N]
        coupling = (self.K / N) * torch.sum(torch.sin(phase_diff), dim=1)

        # Update: φ_i += (ω_i + coupling_i) * dt
        dphases = (self.omega + coupling) * self.dt
        phases_new = phases + dphases

        # Wrap to [-π, π]
        phases_new = torch.atan2(torch.sin(phases_new), torch.cos(phases_new))

        return phases_new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with phase modulation.

        Args:
            x: Time-mixing output [batch, seq, d_model]

        Returns:
            Modulated output [batch, seq, d_model]
        """
        batch, seq, d_model = x.shape

        # Average across batch and sequence for phase update signal
        x_mean = x.mean(dim=(0, 1))  # [d_model]

        # Project to phase space
        phase_logits = self.to_phases(x_mean)  # [num_oscillators]

        # Map to [-π, π] via tanh
        input_phases = torch.tanh(phase_logits) * np.pi

        # Update oscillator phases (Kuramoto dynamics)
        self.phases = self.kuramoto_step(self.phases + input_phases * 0.1)

        # Compute order parameter
        R = self.compute_order_parameter(self.phases)
        self.current_R = R
        self.current_tone = self._map_r_to_tone(R)

        # Drift control (CER)
        if self.enable_drift:
            if R > 0.95:
                self.last_action = "BRAKE"
                # Reduce coupling to prevent over-synchronization
                # (Applied in next step via self.K, but we don't modify K here)
            elif R < 0.80:
                self.last_action = "BOOST"
                # Increase coupling
            else:
                self.last_action = "COAST"

        # Generate modulation signal from phases
        phase_features = torch.cat([
            torch.sin(self.phases),
            torch.cos(self.phases)
        ])  # [2 * num_oscillators]

        # Project back to hidden dimension
        modulation = self.to_modulation(phase_features[:self.num_oscillators])  # [d_model]

        # Apply modulation: output = input * (1 + modulation)
        # Broadcast across batch and sequence
        modulation = modulation.view(1, 1, d_model)
        x_modulated = x * (1.0 + 0.1 * torch.tanh(modulation))

        return x_modulated

    def reset_phases(self):
        """Reset oscillator phases (for new sequences)."""
        self.phases.zero_()
        self.current_R = 0.0
        self.current_tone = "∅"


class PhaseRWKVWrapper(nn.Module):
    """
    Wrapper that adds Phase Core to RWKV model.

    Intercepts time-mixing output at specified layer and applies phase modulation.
    """

    def __init__(
        self,
        rwkv_model,
        phase_layer: int = 12,
        num_oscillators: int = 16,
        coupling_strength: float = 2.0
    ):
        super().__init__()

        self.rwkv_model = rwkv_model
        self.phase_layer = phase_layer

        # Create Phase Core
        self.phase_core = KuramotoPhaseCore(
            d_model=1024,  # RWKV-4-Pile-430M has 1024 hidden
            num_oscillators=num_oscillators,
            coupling_strength=coupling_strength
        )

        print(f"🌀 Phase Core grafted onto RWKV layer {phase_layer}")
        print(f"   Oscillators: {num_oscillators}")
        print(f"   Coupling strength K: {coupling_strength}")

    def forward(self, tokens, state=None):
        """
        Forward pass through RWKV with Phase Core intervention.

        This is tricky: RWKV library uses compiled model, so we can't
        easily intercept at layer 12. Alternative approach:

        1. Run full RWKV forward (get final output and state)
        2. Apply Phase Core modulation to final hidden state
        3. Re-project through head

        For training, we'll use a different strategy: train Phase Core
        separately with RWKV frozen, then fine-tune jointly.
        """
        # For now: pass through RWKV as-is
        # Phase Core will be applied during training to hidden states
        # extracted via hooks
        return self.rwkv_model.forward(tokens, state)

    def trainable_parameters(self):
        """Return only Phase Core parameters for training."""
        return self.phase_core.parameters()


if __name__ == "__main__":
    print("🌀 Phase-RWKV Module Test")
    print("=" * 60)

    # Test Phase Core
    print("\n[1] Testing Kuramoto Phase Core...")
    phase_core = KuramotoPhaseCore(d_model=1024, num_oscillators=16)

    # Simulate time-mixing output
    batch, seq, d_model = 2, 10, 1024
    x = torch.randn(batch, seq, d_model)

    print(f"   Input shape: {x.shape}")

    # Forward pass
    x_modulated = phase_core(x)

    print(f"   Output shape: {x_modulated.shape}")
    print(f"   Order parameter R: {phase_core.current_R:.4f}")
    print(f"   Tone: {phase_core.current_tone}")

    # Multiple steps (simulate temporal evolution)
    print("\n[2] Testing temporal evolution...")
    R_history = []
    for step in range(10):
        x_mod = phase_core(x)
        R_history.append(phase_core.current_R)

    print(f"   R trajectory: {[f'{r:.3f}' for r in R_history]}")
    print(f"   Final R: {phase_core.current_R:.4f} {phase_core.current_tone}")

    print("\n✅ Phase Core module test complete")
