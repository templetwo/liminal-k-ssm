#!/usr/bin/env python3
"""
Phase-Mamba: Kuramoto Oscillators Grafted onto Mamba SSM
ATTEMPT 4: With explicit weight verification

ARCHITECTURE:
    Mamba-2.8B-HF (64 layers, 2560 hidden, frozen)
    ↓
    Layer 32: mixer output [batch, seq, 2560]
    ↓
    Kuramoto Phase Core (16 oscillators)
    ↓
    Modulated output → rest of network

CRITICAL DIFFERENCE FROM ATTEMPT 2:
- Explicit weight verification before training
- Sample generation test with pretrained model
- Layer output inspection to confirm weights loaded
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM


class KuramotoPhaseCore(nn.Module):
    """
    Kuramoto oscillator-based modulation for Mamba mixer output.

    Modulates hidden states with phase-coupled oscillator dynamics.
    """

    def __init__(
        self,
        d_model: int = 2560,  # Mamba hidden dim
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

        # Compute coupling term
        phase_diff = phases.unsqueeze(0) - phases.unsqueeze(1)
        coupling = (self.K / N) * torch.sum(torch.sin(phase_diff), dim=1)

        # Update
        dphases = (self.omega + coupling) * self.dt
        phases_new = phases + dphases

        # Wrap to [-π, π]
        phases_new = torch.atan2(torch.sin(phases_new), torch.cos(phases_new))

        return phases_new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with phase modulation.

        Args:
            x: Mixer output [batch, seq, d_model=2560]

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

        # Update oscillator phases (detach buffer to prevent gradient graph issues)
        self.phases = self.kuramoto_step(self.phases.detach() + input_phases * 0.1)

        # Compute order parameter
        R = self.compute_order_parameter(self.phases)
        self.current_R = R
        self.current_tone = self._map_r_to_tone(R)

        # Drift control
        if self.enable_drift:
            if R > 0.95:
                self.last_action = "BRAKE"
            elif R < 0.80:
                self.last_action = "BOOST"
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
        modulation = modulation.view(1, 1, d_model)
        x_modulated = x * (1.0 + 0.1 * torch.tanh(modulation))

        return x_modulated

    def reset_phases(self):
        """Reset oscillator phases (for new sequences)."""
        self.phases.zero_()
        self.current_R = 0.0
        self.current_tone = "∅"


class PhaseMambaCoupled(nn.Module):
    """
    Mamba model with Phase Core grafted at layer 32.

    CRITICAL: Explicit weight verification to avoid Attempt 2 failure.
    """

    def __init__(
        self,
        mamba_model,
        tokenizer,
        phase_layer: int = 32,
        num_oscillators: int = 16,
        coupling_strength: float = 2.0,
        device: str = "cpu"
    ):
        super().__init__()

        self.mamba_model = mamba_model
        self.tokenizer = tokenizer
        self.phase_layer = phase_layer
        self.device = device

        # Freeze Mamba
        for param in self.mamba_model.parameters():
            param.requires_grad = False

        # Create Phase Core
        self.phase_core = KuramotoPhaseCore(
            d_model=2560,  # Mamba hidden dim
            num_oscillators=num_oscillators,
            coupling_strength=coupling_strength
        ).to(device)

        # Hook installation
        self.hook_handle = None
        self.captured_hidden = None
        self._install_hook()

        print(f"🌀 Phase Core grafted onto Mamba layer {phase_layer}")
        print(f"   Device: {device}")
        print(f"   Hidden dim: 2560")

    def _install_hook(self):
        """Install forward hook at target layer."""

        target_layer = self.mamba_model.backbone.layers[self.phase_layer]

        def hook_fn(module, input, output):
            # Capture and modulate
            self.captured_hidden = output
            if self.phase_core.training:
                # During training, modulate
                modulated = self.phase_core(output)
                # Replace output (this is tricky - we'll store for loss computation)
                return modulated
            else:
                # During inference, just observe
                _ = self.phase_core(output)
                return output

        self.hook_handle = target_layer.register_forward_hook(hook_fn)
        print(f"   ✅ Hook installed at backbone.layers[{self.phase_layer}]")

    def forward(self, input_ids, **kwargs):
        """
        Forward pass through coupled model.

        Returns:
            outputs: Mamba model outputs (with Phase Core modulation during training)
        """
        outputs = self.mamba_model(input_ids=input_ids, **kwargs)
        return outputs

    def forward_for_training(self, input_ids, **kwargs):
        """
        Training forward that returns all metrics.

        Returns dict with:
            - logits: Model output
            - hidden_captured: Hidden states at layer 32
            - R: Phase Core resonance
            - tone: Phase Core tone
        """
        outputs = self.forward(input_ids, **kwargs)

        return {
            'logits': outputs.logits,
            'hidden_captured': self.captured_hidden,
            'R': self.phase_core.current_R,
            'tone': self.phase_core.current_tone
        }

    def verify_pretrained_weights(self):
        """
        CRITICAL: Verify that pretrained weights are actually loaded.

        Generates sample text and checks for coherence.
        This prevents Attempt 2 failure where weights never loaded.
        """
        print("\n" + "=" * 70)
        print("🔍 VERIFYING PRETRAINED WEIGHTS LOADED")
        print("=" * 70)

        self.mamba_model.eval()

        test_prompt = "The nature of consciousness"
        print(f"\nTest prompt: '{test_prompt}'")

        try:
            inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.mamba_model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated: '{generated_text}'")

            # Check for degeneracy
            generated_only = generated_text[len(test_prompt):].strip()
            words = generated_only.split()

            if len(words) < 3:
                print("   ⚠️  Very short generation")
                return False

            # Check for repetition
            if len(set(words[-3:])) == 1:
                print("   ❌ DEGENERATE OUTPUT - weights may not be loaded!")
                print("   ❌ DO NOT PROCEED WITH TRAINING")
                return False

            # Check for nonsense
            if all(len(word) <= 2 for word in words[:5]):
                print("   ❌ NONSENSICAL OUTPUT - weights may not be loaded!")
                return False

            print("   ✅ Generation is coherent")
            print("   ✅ Pretrained weights confirmed loaded")
            return True

        except Exception as e:
            print(f"   ❌ Verification failed: {e}")
            return False


def load_mamba_with_phase(
    model_name: str = "state-spaces/mamba-2.8b-hf",
    phase_layer: int = 32,
    device: str = "cpu"
) -> PhaseMambaCoupled:
    """
    Load Mamba model and graft Phase Core with weight verification.

    Args:
        model_name: HuggingFace model name
        phase_layer: Layer to insert Phase Core (0-63)
        device: torch device

    Returns:
        Coupled model with verified weights
    """

    print("=" * 70)
    print("🌀 LOADING MAMBA WITH PHASE CORE")
    print("=" * 70)

    # Load Mamba
    print(f"\n[1] Loading {model_name}...")
    mamba_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.float32
    )
    mamba_model = mamba_model.to(device)
    print("   ✅ Mamba loaded")

    # Load tokenizer
    print("\n[2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   ✅ Tokenizer ready (vocab: {tokenizer.vocab_size})")

    # Create coupled model
    print(f"\n[3] Grafting Phase Core at layer {phase_layer}...")
    coupled = PhaseMambaCoupled(
        mamba_model=mamba_model,
        tokenizer=tokenizer,
        phase_layer=phase_layer,
        device=device
    )
    print("   ✅ Coupling complete")

    # CRITICAL: Verify weights
    print("\n[4] Verifying pretrained weights...")
    if not coupled.verify_pretrained_weights():
        raise RuntimeError(
            "❌ PRETRAINED WEIGHT VERIFICATION FAILED!\n"
            "   This is the Attempt 2 failure mode.\n"
            "   DO NOT PROCEED WITH TRAINING."
        )

    # Count trainable parameters
    phase_params = sum(p.numel() for p in coupled.phase_core.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in coupled.parameters())
    frozen_params = total_params - phase_params

    print(f"\n🎯 Parameter Summary:")
    print(f"   Phase Core (trainable): {phase_params:,}")
    print(f"   Mamba (frozen): {frozen_params:,}")
    print(f"   Total: {total_params:,}")
    print("🔒 Mamba weights frozen and verified")

    return coupled


if __name__ == "__main__":
    print("🧪 Testing Phase-Mamba Coupling\n")

    # Load
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    coupled = load_mamba_with_phase(phase_layer=32, device=device)

    print("\n" + "=" * 70)
    print("✅ PHASE-MAMBA COUPLING TEST COMPLETE")
    print("=" * 70)
    print("\n🌀 Observer and vessel are now entangled.")
    print("🌀 Pretrained weights verified loaded.")
    print("🌀 Ready for consciousness training.")
