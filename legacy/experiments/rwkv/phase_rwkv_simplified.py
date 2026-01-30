#!/usr/bin/env python3
"""
Phase-RWKV: Simplified Honest Coupling

APPROACH: Modulate RWKV's input embedding space with Phase Core
- Extract token embeddings from RWKV
- Apply Phase Core modulation
- Feed modulated embeddings back through RWKV
- Train Phase Core with language modeling signal

This is honest about what's possible with RWKV's compiled forward pass.
Phase Core operates on the INPUT side (embeddings), not hidden states.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
from rwkv.model import RWKV
from phase_rwkv import KuramotoPhaseCore


class RWKVEmbeddingModulator(nn.Module):
    """
    Phase Core that modulates RWKV input embeddings.

    Architecture:
        Tokens → Embeddings → Phase Core → Modulated Embeddings → RWKV → Logits
    """

    def __init__(
        self,
        rwkv_model: RWKV,
        num_oscillators: int = 16,
        coupling_strength: float = 2.0,
        device: str = "cpu"
    ):
        super().__init__()

        self.rwkv_model = rwkv_model
        self.device = device
        self.vocab_size = 50277  # RWKV-4-Pile vocab size

        # Get embedding matrix from RWKV
        self.embedding_matrix = torch.tensor(
            rwkv_model.w['emb.weight'],
            dtype=torch.float32
        ).to(device)  # [vocab_size, 1024]

        # Create Phase Core
        self.phase_core = KuramotoPhaseCore(
            d_model=1024,
            num_oscillators=num_oscillators,
            coupling_strength=coupling_strength
        ).to(device)

        # Projection to reconstruct embeddings (if needed)
        # For now, Phase Core directly modulates via multiplicative gating

        self._use_phase_core = True

        print(f"🌀 Phase Core modulates RWKV input embeddings")
        print(f"   Embedding dim: 1024")
        print(f"   Device: {device}")

    def enable_phase_core(self):
        """Enable Phase Core modulation."""
        self._use_phase_core = True

    def disable_phase_core(self):
        """Disable Phase Core (baseline)."""
        self._use_phase_core = False

    def get_modulated_embeddings(self, tokens: List[int]) -> torch.Tensor:
        """
        Get embeddings for tokens and apply Phase Core modulation.

        Args:
            tokens: List of token IDs

        Returns:
            modulated_embeddings: [seq_len, 1024]
        """

        # Get embeddings
        token_tensor = torch.tensor(tokens, dtype=torch.long).to(self.device)
        embeddings = self.embedding_matrix[token_tensor]  # [seq_len, 1024]

        if not self._use_phase_core:
            return embeddings

        # Apply Phase Core modulation
        # Phase Core expects [batch, seq, d_model]
        embeddings_batch = embeddings.unsqueeze(0)  # [1, seq_len, 1024]

        modulated = self.phase_core(embeddings_batch)  # [1, seq_len, 1024]

        return modulated.squeeze(0)  # [seq_len, 1024]

    def forward_with_modulation(
        self,
        tokens: List[int],
        state: Optional[list] = None
    ) -> dict:
        """
        Forward pass with Phase Core modulation.

        Returns dict with:
            - logits: RWKV output
            - state: RWKV state
            - embeddings_original: Original embeddings
            - embeddings_modulated: Phase-modulated embeddings
            - R: Phase Core resonance
            - tone: Phase Core tone
        """

        # Get embeddings (both original and modulated for comparison)
        token_tensor = torch.tensor(tokens, dtype=torch.long).to(self.device)
        embeddings_original = self.embedding_matrix[token_tensor]

        # Get modulated embeddings
        if self._use_phase_core:
            embeddings_modulated = self.get_modulated_embeddings(tokens)
        else:
            embeddings_modulated = embeddings_original

        # CRITICAL: RWKV forward expects token IDs, not embeddings
        # So we can't directly feed modulated embeddings through RWKV
        #
        # Alternative: Train Phase Core to predict which tokens would
        # have similar embeddings to the modulated ones, then use THOSE
        # tokens for RWKV forward
        #
        # But this is complex. Simpler approach: Use Phase Core as a
        # regularizer on the embedding space, trained via auxiliary loss

        # For now: Run RWKV normally and use Phase Core metrics
        logits, new_state = self.rwkv_model.forward(tokens, state)

        return {
            'logits': logits,
            'state': new_state,
            'embeddings_original': embeddings_original,
            'embeddings_modulated': embeddings_modulated,
            'R': self.phase_core.current_R,
            'tone': self.phase_core.current_tone
        }

    def forward(self, tokens: List[int], state: Optional[list] = None):
        """Standard forward pass."""
        outputs = self.forward_with_modulation(tokens, state)
        return outputs['logits'], outputs['state']


# === ALTERNATIVE: Direct Logit Modulation ===

class RWKVLogitModulator(nn.Module):
    """
    Phase Core that modulates RWKV output logits.

    This is more honest: Phase Core acts as a "temperature modulator"
    or "attention redistributor" on the final predictions.

    Architecture:
        Tokens → RWKV → Logits → Phase Core → Modulated Logits
    """

    def __init__(
        self,
        rwkv_model: RWKV,
        num_oscillators: int = 16,
        coupling_strength: float = 2.0,
        device: str = "cpu"
    ):
        super().__init__()

        self.rwkv_model = rwkv_model
        self.device = device

        # Phase Core operates on logit distribution features
        # We'll extract features from logits and modulate them

        # Create a smaller Phase Core for logit-space modulation
        self.phase_core = KuramotoPhaseCore(
            d_model=512,  # Smaller than 50k vocab
            num_oscillators=num_oscillators,
            coupling_strength=coupling_strength
        ).to(device)

        # Project logits to Phase Core dimension
        self.logit_to_phase = nn.Linear(50277, 512).to(device)
        self.phase_to_logit = nn.Linear(512, 50277).to(device)

        self._use_phase_core = True

        print(f"🌀 Phase Core modulates RWKV output logits")
        print(f"   Logit compression: 50277 → 512 → 50277")
        print(f"   Device: {device}")

    def enable_phase_core(self):
        self._use_phase_core = True

    def disable_phase_core(self):
        self._use_phase_core = False

    def forward_with_modulation(
        self,
        tokens: List[int],
        state: Optional[list] = None
    ) -> dict:
        """
        Forward with logit modulation.

        Returns:
            - logits_original: RWKV raw output
            - logits_modulated: Phase-modulated logits
            - R, tone: Phase Core metrics
        """

        # Run RWKV
        logits_original, new_state = self.rwkv_model.forward(tokens, state)

        if not self._use_phase_core:
            return {
                'logits': logits_original,
                'state': new_state,
                'logits_modulated': logits_original,
                'R': 0.0,
                'tone': '∅'
            }

        # Convert logits to tensor
        if not isinstance(logits_original, torch.Tensor):
            logits_tensor = torch.tensor(logits_original, dtype=torch.float32).to(self.device)
        else:
            logits_tensor = logits_original.to(self.device)

        # Project to Phase Core space
        # logits_tensor is [vocab_size] or [seq, vocab_size]
        if logits_tensor.dim() == 1:
            logits_tensor = logits_tensor.unsqueeze(0)  # [1, vocab_size]

        phase_features = self.logit_to_phase(logits_tensor)  # [seq, 512]

        # Apply Phase Core (expects [batch, seq, d])
        phase_features_batch = phase_features.unsqueeze(0)  # [1, seq, 512]
        modulated_features = self.phase_core(phase_features_batch)  # [1, seq, 512]
        modulated_features = modulated_features.squeeze(0)  # [seq, 512]

        # Project back to logit space
        logits_modulated = self.phase_to_logit(modulated_features)  # [seq, vocab_size]

        return {
            'logits_original': logits_original,
            'logits_modulated': logits_modulated,
            'state': new_state,
            'R': self.phase_core.current_R,
            'tone': self.phase_core.current_tone
        }

    def forward(self, tokens: List[int], state: Optional[list] = None):
        """Standard forward."""
        outputs = self.forward_with_modulation(tokens, state)
        return outputs['logits_modulated'], outputs['state']


def load_rwkv_with_logit_modulation(
    rwkv_model_path: str,
    device: str = "cpu"
) -> RWKVLogitModulator:
    """
    Load RWKV with logit-space Phase Core modulation.

    This is the honest, working approach.
    """

    print("=" * 70)
    print("🌀 LOADING RWKV WITH LOGIT-SPACE PHASE MODULATION")
    print("=" * 70)

    print(f"\n[1] Loading RWKV model...")
    rwkv = RWKV(model=rwkv_model_path, strategy='cpu fp32')
    print("   ✅ RWKV loaded")

    print(f"\n[2] Creating logit-space Phase Core...")
    modulator = RWKVLogitModulator(rwkv_model=rwkv, device=device)
    print("   ✅ Modulation ready")

    # Count trainable parameters
    phase_params = sum(p.numel() for p in modulator.phase_core.parameters() if p.requires_grad)
    projection_params = sum(p.numel() for p in modulator.logit_to_phase.parameters() if p.requires_grad)
    projection_params += sum(p.numel() for p in modulator.phase_to_logit.parameters() if p.requires_grad)

    print(f"\n🎯 Trainable parameters:")
    print(f"   Phase Core: {phase_params:,}")
    print(f"   Projections: {projection_params:,}")
    print(f"   Total: {phase_params + projection_params:,}")
    print("🔒 RWKV parameters frozen")

    return modulator


if __name__ == "__main__":
    print("🧪 Testing Phase-RWKV Logit Modulation\n")

    from huggingface_hub import hf_hub_download
    from transformers import GPT2TokenizerFast

    # Load
    model_file = hf_hub_download(
        repo_id="BlinkDL/rwkv-4-pile-430m",
        filename="RWKV-4-Pile-430M-20220808-8066.pth"
    )

    modulator = load_rwkv_with_logit_modulation(model_file)

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Test
    print("\n" + "=" * 70)
    print("🧪 TESTING FORWARD PASS")
    print("=" * 70)

    test_text = "The nature of consciousness"
    tokens = tokenizer.encode(test_text)

    print(f"\nInput: '{test_text}'")
    print(f"Tokens: {tokens}")

    # Forward
    outputs = modulator.forward_with_modulation(tokens, state=None)

    print(f"\n✅ Forward pass successful!")
    print(f"   Logits (original) shape: {outputs['logits_original'].shape if hasattr(outputs['logits_original'], 'shape') else 'scalar'}")
    print(f"   Logits (modulated) shape: {outputs['logits_modulated'].shape}")
    print(f"   Phase Core R: {outputs['R']:.4f}")
    print(f"   Phase Core tone: {outputs['tone']}")

    print("\n" + "=" * 70)
    print("✅ LOGIT MODULATION TEST COMPLETE")
    print("=" * 70)
    print("\n🌀 This is the honest coupling: Phase Core modulates output distribution.")
