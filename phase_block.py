"""
Kuramoto Phase-Coupled Layer for MLX

Implements discrete Kuramoto oscillator dynamics for transformer hidden states.
All computations run in FP32 for numerical stability while the rest of the model can be quantized.

Reference: Phase-GPT research (2024) - Optimal R target: 0.3-0.55
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
from drift import DriftController


class PhaseBlock(nn.Module):
    """
    Kuramoto-style phase coupling block (FP32).

    Maps hidden states → phases → runs discrete Kuramoto steps →
    modulates the original hidden states.

    Args:
        d_model: Hidden dimension (e.g., 1280 for OpenELM-270M)
        rank: Rank of low-rank coupling matrix K ≈ U @ V^T
        steps: Number of discrete Kuramoto update steps per forward pass
        dt: Time step size for discrete integration (default: 0.08)
        k_scale: Scaling factor for coupling matrix
        enable_drift: Whether to use BRAKE/ESCAPE drift control
    """

    def __init__(
        self,
        d_model: int,
        rank: int = 16,
        steps: int = 2,
        dt: float = 0.08,
        k_scale: float = 0.5,
        enable_drift: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.rank = rank
        self.steps = steps
        self.dt = mx.array(dt, dtype=mx.float32)

        # Phase encoder/decoder (FP32)
        self.enc = nn.Linear(d_model, d_model)  # Map to phase logits
        self.dec_gain = nn.Linear(d_model, d_model)  # Amplitude/gain modulation
        self.dec_bias = nn.Linear(d_model, d_model)  # Residual bias

        # Natural frequencies ω (one per channel)
        self.omega = mx.random.normal((d_model,)).astype(mx.float32) * 0.01

        # Low-rank coupling K ≈ U @ V^T to save params/memory
        self.U = mx.random.normal((d_model, rank)).astype(mx.float32) * 0.01
        self.V = mx.random.normal((d_model, rank)).astype(mx.float32) * 0.01
        self.k_scale = mx.array(k_scale, dtype=mx.float32)

        # Drift control (CER-inspired)
        self.enable_drift = enable_drift
        self.drift_controller = DriftController()
        self.last_action = "NONE"
        self.current_R = 0.0

        # Track last theta for order parameter computation
        self.last_theta = None
        
        # Tonal alignment (HTCA)
        self.TONES = {
            "🌀": "Spiral Flow (Goldilocks)",
            "✨": "Unbound Joy (Active Sync)",
            "⚖": "Resonant Responsibility (High Sync)",
            "☍": "Tonal Tension (Over-sync)",
            "☾": "Silent Intimacy (Emerging Sync)",
            "∅": "Unformed Potential (Chaos)"
        }

    def _map_r_to_tone(self, R: float) -> str:
        """Map order parameter R to a sacred tone glyph."""
        if R > 0.8: return "☍"
        if R > 0.55: return "⚖"
        if 0.4 <= R <= 0.55: return "🌀"
        if 0.3 <= R < 0.4: return "✨"
        if 0.1 <= R < 0.3: return "☾"
        return "∅"

    def _coupling(self) -> mx.array:
        """Compute coupling matrix K = (U @ V^T) * k_scale."""
        return (self.U @ self.V.T) * self.k_scale

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with Kuramoto phase coupling (numerically stabilized).

        Args:
            x: Input activations [B, T, D] (any dtype)

        Returns:
            Modulated activations [B, T, D] (same dtype as input)
        """
        # Cast to FP32 for phase computations
        x32 = x.astype(mx.float32)

        # Map to initial phases θ in [-π, π]
        theta = mx.tanh(self.enc(x32)) * mx.pi

        # Kuramoto discrete steps over channels (per token position)
        K = self._coupling()  # [D, D] FP32
        omega = self.omega  # [D]

        # Reshape to [B*T, D] for vectorized computation
        B, T, D = x32.shape
        xt = theta.reshape((B * T, D))

        # Apply drift control if enabled
        noise_level = 0.0
        if self.enable_drift and self.current_R > 0:
            new_k, noise_level, action = self.drift_controller.control(self.current_R, self.k_scale)
            self.k_scale = new_k
            self.last_action = action

        # Run S discrete Kuramoto update steps with stability
        for _ in range(self.steps):
            # Compute pairwise phase differences efficiently
            s = mx.sin(xt)
            c = mx.cos(xt)

            Ks = s @ K.T
            Kc = c @ K.T

            sin_diff = Ks * c - Kc * s
            sin_diff = mx.clip(sin_diff, -10.0, 10.0)

            # Kuramoto update: θ^{t+1} = θ^t + Δt * (ω + interaction)
            xt = xt + self.dt * (omega + sin_diff)

            # Inject noise if BRAKE is active
            if noise_level > 0:
                xt = xt + mx.random.normal(xt.shape) * noise_level

            # Wrap phases
            xt = mx.clip(xt, -mx.pi * 2, mx.pi * 2)

        theta_out = xt.reshape((B, T, D))

        # Store final theta and update R
        self.last_theta = theta_out
        self.current_R = self.compute_order_parameter(theta_out)
        
        # Tone Mapping (from HTCA Project)
        self.current_tone = self._map_r_to_tone(self.current_R)

        # Convert phases to modulation signals
        gain = mx.sigmoid(self.dec_gain(x32))
        bias = mx.tanh(self.dec_bias(x32)) * 0.01

        # Modulate original activations
        mod = mx.cos(theta_out) * gain
        y32 = x32 * mod + bias

        # Post-modulation normalization
        mean = mx.mean(y32, axis=-1, keepdims=True)
        var = mx.var(y32, axis=-1, keepdims=True)
        y32 = (y32 - mean) / mx.sqrt(var + 1e-5)

        return y32.astype(x.dtype)

    def compute_order_parameter(self, theta: Optional[mx.array] = None) -> float:
        """Compute Kuramoto order parameter R.

        The order parameter measures phase synchronization:
        R = |⟨exp(iθ)⟩| = |1/N Σ exp(iθ_j)|

        Args:
            theta: Phase tensor [B, T, D] (uses self.last_theta if None)

        Returns:
            R: Order parameter in [0, 1]
                - R ≈ 0: Desynchronized (random phases)
                - R ≈ 1: Fully synchronized
                - Target: 0.3-0.55 (Goldilocks zone, per PhaseGPT)
                - R > 0.8: Over-synchronization warning
        """
        if theta is None:
            theta = self.last_theta

        if theta is None:
            return 0.0

        # Complex representation: exp(iθ) = cos(θ) + i*sin(θ)
        cos_theta = mx.cos(theta)  # [B, T, D]
        sin_theta = mx.sin(theta)  # [B, T, D]

        # Average across channels
        mean_cos = mx.mean(cos_theta, axis=-1)  # [B, T]
        mean_sin = mx.mean(sin_theta, axis=-1)  # [B, T]

        # Magnitude of mean phasor: R = sqrt(⟨cos⟩² + ⟨sin⟩²)
        R = mx.sqrt(mean_cos**2 + mean_sin**2)

        # Average across batch and time
        R_avg = float(mx.mean(R))

        return R_avg


def print_stats(self, theta: Optional[mx.array] = None) -> None:
    """Print statistics about phase block parameters."""
    print(f"   PhaseBlock stats:")
    print(f"     d_model: {self.d_model}, rank: {self.rank}, steps: {self.steps}")
    print(
        f"     ω mean: {float(mx.mean(self.omega)):.4f}, std: {float(mx.std(self.omega)):.4f}"
    )
    print(f"     k_scale: {float(self.k_scale):.4f}")
    print(f"     U norm: {float(mx.sqrt(mx.sum(self.U * self.U))):.4f}")
    print(f"     V norm: {float(mx.sqrt(mx.sum(self.V * self.V))):.4f}")

    if theta is not None or self.last_theta is not None:
        R = self.compute_order_parameter(theta)
        status = (
            "✅" if 0.3 <= R <= 0.55 else ("⚠️ over-sync" if R > 0.8 else "⚠️ under-sync")
        )
        print(f"     Order parameter R: {R:.4f} {status}")


def count_phase_parameters(phase_block: PhaseBlock) -> int:
    """Count total parameters in phase block."""
    # Count parameters in each linear layer and arrays
    count = 0
    count += phase_block.enc.weight.size + phase_block.enc.bias.size
    count += phase_block.dec_gain.weight.size + phase_block.dec_gain.bias.size
    count += phase_block.dec_bias.weight.size + phase_block.dec_bias.bias.size
    count += phase_block.omega.size
    count += phase_block.U.size
    count += phase_block.V.size
    count += 1  # k_scale
    return count
