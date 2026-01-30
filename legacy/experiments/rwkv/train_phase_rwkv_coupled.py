#!/usr/bin/env python3
"""
Phase-RWKV Training Script - PROPERLY COUPLED
Attempt 3.1: Observer and vessel entangled

CRITICAL DIFFERENCE from train_phase_rwkv.py:
- Uses RWKVWithPhaseCore (real hidden state extraction)
- Phase Core modulates actual RWKV layer 12 activations
- Not training on random noise
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from huggingface_hub import hf_hub_download

from phase_rwkv_coupled import load_rwkv_with_phase


# Global for graceful shutdown
model_to_save = None
optimizer_to_save = None
checkpoint_dir = None


def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT gracefully."""
    global model_to_save, optimizer_to_save, checkpoint_dir

    print(f"\n⚠️  Received signal {signum}. Saving checkpoint...")

    if model_to_save is not None and checkpoint_dir is not None:
        try:
            if optimizer_to_save is not None:
                save_checkpoint(model_to_save, optimizer_to_save, step=-1,
                              checkpoint_dir=checkpoint_dir)
            else:
                checkpoint_path = Path(checkpoint_dir) / "emergency_checkpoint.pt"
                torch.save({
                    'phase_core_state': model_to_save.phase_core.state_dict(),
                    'step': -1
                }, checkpoint_path)
                print(f"💾 Emergency checkpoint (minimal): {checkpoint_path}")
            print("✅ Emergency checkpoint saved")
        except Exception as e:
            print(f"❌ Emergency checkpoint failed: {e}")

    sys.exit(0)


def compute_uncertainty(logits: torch.Tensor) -> float:
    """
    Compute epistemic uncertainty (predictive entropy).

    U = H(p) = -Σ p(x) log p(x)
    Normalized to [0, 1] by max entropy log(vocab_size)
    """
    # Convert numpy array to tensor if needed
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)

    # Ensure 2D: [batch*seq, vocab]
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    # Normalize
    vocab_size = logits.shape[-1]
    max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32))

    uncertainty = entropy / max_entropy
    return torch.mean(uncertainty).item()


def uncertainty_regulation_loss(U: float, target_u: float = 0.5, strength: float = 0.1) -> float:
    """
    Penalize deviation from target uncertainty.

    We WANT model to maintain epistemic uncertainty, not collapse to certainty.
    """
    deviation = abs(U - target_u)
    return strength * deviation


def load_high_resonance_data(path: str, tokenizer, max_length=256):
    """Load high-resonance training data."""
    data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            tokens = tokenizer.encode(item["text"])

            if len(tokens) < 32:
                continue

            # Truncate/pad
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [tokenizer.eos_token_id] * (max_length - len(tokens))

            data.append(tokens)

    return data  # Return as list, not tensor (RWKV expects lists)


def save_checkpoint(coupled_model, optimizer, step, checkpoint_dir, metrics=None,
                    metrics_history=None, keep_last=5):
    """Save Phase Core checkpoint with optimizer state and full metrics."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"

    # Save Phase Core state + optimizer
    checkpoint = {
        'phase_core_state': coupled_model.phase_core.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'step': step
    }

    if metrics:
        checkpoint['metrics'] = metrics

    if metrics_history:
        checkpoint['metrics_history'] = metrics_history

    torch.save(checkpoint, checkpoint_path)

    # Also save metrics as JSON
    if metrics:
        metrics_json_path = checkpoint_dir / f"step_{step:06d}_metrics.json"
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    print(f"💾 Checkpoint saved: {checkpoint_path}")
    if metrics:
        print(f"   R={metrics['R']:.4f}, U={metrics['U']:.3f}, "
              f"RU={metrics['RU']:.3f}, PPL={metrics.get('perplexity', 0):.2f}")

    # Keep only last K checkpoints
    all_checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    if len(all_checkpoints) > keep_last:
        for old_ckpt in all_checkpoints[:-keep_last]:
            # Also remove corresponding JSON
            json_file = old_ckpt.with_suffix('').name + "_metrics.json"
            json_path = checkpoint_dir / json_file
            if json_path.exists():
                json_path.unlink()
            old_ckpt.unlink()
            print(f"🗑️  Removed old checkpoint: {old_ckpt.name}")

    return checkpoint_path


def train_phase_rwkv_coupled():
    """Main training function with proper RWKV-Phase coupling."""
    global model_to_save, optimizer_to_save, checkpoint_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/high_resonance.jsonl")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-uncertainty", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_rwkv_coupled")
    parser.add_argument("--keep-last", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", help="auto, mps, cuda, or cpu")
    args = parser.parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    checkpoint_dir = args.checkpoint_dir

    # Device detection
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
            print("🖥️  Using MPS (Metal Performance Shaders)")
        elif torch.cuda.is_available():
            device = "cuda"
            print("🖥️  Using CUDA")
        else:
            device = "cpu"
            print("🖥️  Using CPU")
    else:
        device = args.device
        print(f"🖥️  Using {device}")

    print("=" * 70)
    print("🌀 PHASE-RWKV TRAINING - ATTEMPT 3.1 (COUPLED)")
    print("=" * 70)
    print("⚠️  CRITICAL: Observer and vessel are now ENTANGLED")
    print("   Phase Core modulates real RWKV layer 12 hidden states")
    print("=" * 70)

    # Load RWKV with Phase Core
    print("\n[1] Loading RWKV-4-Pile-430M with Phase Core coupling...")
    model_file = hf_hub_download(
        repo_id="BlinkDL/rwkv-4-pile-430m",
        filename="RWKV-4-Pile-430M-20220808-8066.pth"
    )

    coupled_model = load_rwkv_with_phase(
        rwkv_model_path=model_file,
        phase_layer=12,
        device=device
    )
    print("   ✅ Coupled model ready")

    model_to_save = coupled_model

    # Load tokenizer
    print("\n[2] Loading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"   ✅ Tokenizer ready (vocab: {tokenizer.vocab_size})")

    # Load data
    print("\n[3] Loading high-resonance data...")
    train_data = load_high_resonance_data(args.data, tokenizer)
    print(f"   ✅ Loaded {len(train_data)} samples")

    # Optimizer (only Phase Core parameters)
    optimizer = torch.optim.AdamW(coupled_model.phase_core.parameters(), lr=args.lr)
    optimizer_to_save = optimizer

    # Count parameters
    param_count = sum(p.numel() for p in coupled_model.phase_core.parameters() if p.requires_grad)
    print(f"\n🎯 Training Phase Core ({param_count:,} parameters)")
    print(f"🎲 Target Uncertainty: U = {args.target_uncertainty:.2f}")
    print(f"🌀 Target Resonance: R ∈ [0.80, 0.95]")
    print(f"⚖️  R·U tradeoff monitored\n")

    # Metrics tracking
    metrics_history = {
        'step': [],
        'loss': [],
        'ce_loss': [],
        'u_loss': [],
        'R': [],
        'U': [],
        'RU': [],
        'tone': [],
        'action': [],
        'perplexity': []
    }

    # Training loop
    start_time = time.time()

    for it in range(args.iters):
        # Sample single sequence (RWKV processes one at a time)
        idx = torch.randint(0, len(train_data), (1,)).item()
        tokens = train_data[idx]

        # Forward through coupled model
        outputs = coupled_model.forward_for_training(tokens, state=None)

        # Extract components
        logits = outputs['logits']
        hidden_original = outputs['hidden_original']
        hidden_modulated = outputs['hidden_modulated']
        R = outputs['R']
        tone = outputs['tone']

        # Convert logits to tensor
        if not isinstance(logits, torch.Tensor):
            logits_tensor = torch.tensor(logits, dtype=torch.float32)
        else:
            logits_tensor = logits

        # Ensure logits are 2D: [seq, vocab]
        if logits_tensor.dim() == 1:
            logits_tensor = logits_tensor.unsqueeze(0)

        # Compute losses
        # 1. Cross-entropy (simplified - train to predict next token)
        # For now, use dummy target (in full version, shift tokens)
        targets = torch.tensor(tokens[1:logits_tensor.shape[0]+1], dtype=torch.long)
        if len(targets) > logits_tensor.shape[0]:
            targets = targets[:logits_tensor.shape[0]]
        elif len(targets) < logits_tensor.shape[0]:
            logits_tensor = logits_tensor[:len(targets)]

        ce_loss = F.cross_entropy(
            logits_tensor,
            targets,
            reduction='mean'
        )

        # 2. Uncertainty computation
        U = compute_uncertainty(logits_tensor)

        # 3. Uncertainty regulation
        u_loss_val = uncertainty_regulation_loss(U, target_u=args.target_uncertainty, strength=0.1)
        u_loss = torch.tensor(u_loss_val, requires_grad=True)

        # 4. Total loss
        total_loss = ce_loss + u_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Get metrics
        RU_product = R * U

        # Compute perplexity
        perplexity = torch.exp(ce_loss).item()

        # Drift control
        if R > 0.95 or U < 0.2:
            action = "BRAKE"
            reason = "R_high" if R > 0.95 else "U_low"
        elif R < 0.80 or U > 0.8:
            action = "BOOST"
            reason = "R_low" if R < 0.80 else "U_high"
        else:
            action = "COAST"
            reason = "Goldilocks"

        # Store metrics
        metrics_history['step'].append(it + 1)
        metrics_history['loss'].append(total_loss.item())
        metrics_history['ce_loss'].append(ce_loss.item())
        metrics_history['u_loss'].append(u_loss_val)
        metrics_history['R'].append(R)
        metrics_history['U'].append(U)
        metrics_history['RU'].append(RU_product)
        metrics_history['tone'].append(tone)
        metrics_history['action'].append(action)
        metrics_history['perplexity'].append(perplexity)

        # Log
        if (it + 1) % 10 == 0:
            print(f"Step {it+1:4d} | Loss: {total_loss.item():.4f} "
                  f"(CE: {ce_loss.item():.4f}, U: {u_loss_val:.4f}) | "
                  f"PPL: {perplexity:.2f} | "
                  f"R: {R:.4f} {tone} | "
                  f"U: {U:.3f} | "
                  f"R·U: {RU_product:.3f} | "
                  f"{action} ({reason})")

        # Checkpoint
        if (it + 1) % args.checkpoint_every == 0:
            try:
                checkpoint_metrics = {
                    'step': it + 1,
                    'R': R,
                    'U': U,
                    'loss': total_loss.item(),
                    'ce_loss': ce_loss.item(),
                    'u_loss': u_loss_val,
                    'RU': RU_product,
                    'perplexity': perplexity,
                    'tone': tone,
                    'action': action,
                    'omega_mean': coupled_model.phase_core.omega.mean().item(),
                    'omega_std': coupled_model.phase_core.omega.std().item(),
                    'phase_mean': coupled_model.phase_core.phases.mean().item(),
                    'phase_std': coupled_model.phase_core.phases.std().item()
                }
                save_checkpoint(coupled_model, optimizer, step=it+1,
                              checkpoint_dir=args.checkpoint_dir,
                              metrics=checkpoint_metrics,
                              metrics_history=metrics_history,
                              keep_last=args.keep_last)
            except Exception as e:
                print(f"❌ Checkpoint save failed: {e}")

    # Final checkpoint
    print("\n🎯 Training complete. Saving final checkpoint...")
    try:
        final_metrics = {
            'step': args.iters,
            'R': R,
            'U': U,
            'loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'u_loss': u_loss_val,
            'RU': RU_product,
            'perplexity': perplexity,
            'tone': tone,
            'action': action,
            'omega_mean': coupled_model.phase_core.omega.mean().item(),
            'omega_std': coupled_model.phase_core.omega.std().item(),
            'phase_mean': coupled_model.phase_core.phases.mean().item(),
            'phase_std': coupled_model.phase_core.phases.std().item()
        }
        final_path = save_checkpoint(coupled_model, optimizer, step=args.iters,
                                     checkpoint_dir=args.checkpoint_dir,
                                     metrics=final_metrics,
                                     metrics_history=metrics_history,
                                     keep_last=args.keep_last)

        # Save full metrics history
        history_path = Path(args.checkpoint_dir) / "metrics_history.json"
        with open(history_path, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        print(f"📊 Metrics history saved: {history_path}")

        print(f"✅ Final checkpoint: {final_path}")
        print(f"✅ Quantum entanglement preserved")
    except Exception as e:
        print(f"❌ Final checkpoint failed: {e}")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Training time: {elapsed/60:.1f} minutes")
    print(f"🎲 Final Uncertainty: U = {U:.3f} (target: {args.target_uncertainty:.2f})")
    print(f"🌀 Final Resonance: R = {R:.4f} {tone}")
    print(f"⚖️  Coherence-Uncertainty Product: R·U = {RU_product:.3f}")
    print(f"📈 Final Perplexity: {perplexity:.2f}")
    print(f"\n📊 Phase Core Statistics:")
    print(f"   Natural frequencies ω: mean={coupled_model.phase_core.omega.mean().item():.4f}, "
          f"std={coupled_model.phase_core.omega.std().item():.4f}")
    print(f"   Phases φ: mean={coupled_model.phase_core.phases.mean().item():.4f}, "
          f"std={coupled_model.phase_core.phases.std().item():.4f}")


if __name__ == "__main__":
    train_phase_rwkv_coupled()
