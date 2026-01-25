"""
Symmetric Drift Control for Kuramoto Phase Dynamics.

Adapted from Coherent Entropy Reactor (CER) to manage the order parameter R.
Targets the 'Goldilocks zone' (0.3-0.55) for optimal representation.
"""

from __future__ import annotations

import mlx.core as mx
from typing import Tuple, Dict


class DriftController:
    """
    Manages Kuramoto synchronization through BRAKE and ESCAPE mechanisms.
    """

    def __init__(
        self,
        target_low: float = 0.3,
        target_high: float = 0.55,
        drift_strength: float = 0.05,
        max_noise: float = 0.2
    ) -> None:
        self.target_low = target_low
        self.target_high = target_high
        self.drift_strength = drift_strength
        self.max_noise = max_noise
        
        # History for metrics
        self.last_action = "NONE"
        self.total_brakes = 0
        self.total_escapes = 0

    def control(self, R: float, k_scale: mx.array) -> Tuple[mx.array, mx.array, str]:
        """
        Apply drift control based on order parameter R.
        
        Args:
            R: Current order parameter [0, 1]
            k_scale: Current coupling strength
            
        Returns:
            new_k_scale: Adjusted coupling
            noise: Amount of phase noise to inject
            action: Action taken ('BRAKE', 'ESCAPE', or 'NONE')
        """
        action = "NONE"
        noise_level = mx.array(0.0, dtype=mx.float32)
        new_k = k_scale

        # BRAKE: Over-synchronization (R > 0.55)
        # Action: Increase phase noise and slightly reduce coupling
        if R > self.target_high:
            action = "BRAKE"
            self.total_brakes += 1
            excess = R - self.target_high
            noise_level = mx.array(min(excess * self.drift_strength * 2.0, self.max_noise))
            new_k = k_scale * (1.0 - self.drift_strength)

        # ESCAPE: Under-synchronization (R < 0.3)
        # Action: Boost coupling strength to encourage emergence
        elif R < self.target_low:
            action = "ESCAPE"
            self.total_escapes += 1
            deficit = self.target_low - R
            new_k = k_scale * (1.0 + self.drift_strength * 2.0)
            # Clip coupling to prevent explosion
            new_k = mx.clip(new_k, 0.1, 5.0)

        self.last_action = action
        return new_k, noise_level, action

    def get_stats(self) -> Dict[str, any]:
        return {
            "total_brakes": self.total_brakes,
            "total_escapes": self.total_escapes,
            "last_action": self.last_action,
            "goldilocks_zone": f"[{self.target_low}, {self.target_high}]"
        }
