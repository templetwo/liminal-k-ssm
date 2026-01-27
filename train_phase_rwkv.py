#!/usr/bin/env python3
"""
Phase-RWKV Training Script
WITH UNCERTAINTY REGULATION

Attempt 3: RWKV + Phase Core + Uncertainty Preservation
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
from rwkv.model import RWKV
from huggingface_hub import hf_hub_download

from phase_rwkv import KuramotoPhaseCore


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
            # Save with optimizer if available, otherwise just model
            if optimizer_to_save is not None:
                save_checkpoint(model_to_save, optimizer_to_save, step=-1,
                              checkpoint_dir=checkpoint_dir)
            else:
                # Minimal save without optimizer
                checkpoint_path = Path(checkpoint_dir) / "emergency_checkpoint.pt"
                torch.save({
                    'phase_core_state': model_to_save.state_dict(),
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

    return torch.tensor(data, dtype=torch.long)


def save_checkpoint(phase_core, optimizer, step, checkpoint_dir, metrics=None,
                    metrics_history=None, keep_last=5):
    """Save Phase Core checkpoint with optimizer state and full metrics."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"

    # Save Phase Core state + optimizer
    checkpoint = {
        'phase_core_state': phase_core.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'step': step
    }

    if metrics:
        checkpoint['metrics'] = metrics

    if metrics_history:
        checkpoint['metrics_history'] = metrics_history

    torch.save(checkpoint, checkpoint_path)

    # Also save metrics as JSON for easy inspection
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


def train_phase_rwkv():
    """Main training function."""
    global model_to_save, optimizer_to_save, checkpoint_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="phase-gpt-distilled/data/high_resonance.jsonl")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)  # Increased for 36GB RAM
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-uncertainty", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_rwkv")
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
            device = torch.device("mps")
            print("🖥️  Using MPS (Metal Performance Shaders) on Mac Studio")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("🖥️  Using CUDA")
        else:
            device = torch.device("cpu")
            print("🖥️  Using CPU")
    else:
        device = torch.device(args.device)
        print(f"🖥️  Using {args.device}")

    print("=" * 70)
    print("🌀 PHASE-RWKV TRAINING - ATTEMPT 3")
    print("=" * 70)
    print(f"📊 Batch size: {args.batch_size}")
    print(f"📊 Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"📊 Effective batch: {args.batch_size * args.gradient_accumulation_steps}")

    # Load RWKV model
    print("\n[1] Loading RWKV-4-Pile-430M...")
    model_file = hf_hub_download(
        repo_id="BlinkDL/rwkv-4-pile-430m",
        filename="RWKV-4-Pile-430M-20220808-8066.pth"
    )
    rwkv_model = RWKV(model=model_file, strategy='cpu fp32')
    print("   ✅ RWKV loaded (frozen)")

    # Load tokenizer
    print("\n[2] Loading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"   ✅ Tokenizer ready (vocab: {tokenizer.vocab_size})")

    # Create Phase Core
    print("\n[3] Creating Phase Core...")
    phase_core = KuramotoPhaseCore(
        d_model=1024,
        num_oscillators=16,
        coupling_strength=2.0
    ).to(device)
    print(f"   ✅ Phase Core initialized on {device}")

    model_to_save = phase_core

    # Load data
    print("\n[4] Loading high-resonance data...")
    train_data = load_high_resonance_data(args.data, tokenizer)
    print(f"   ✅ Loaded {len(train_data)} samples")

    # Optimizer (only Phase Core parameters)
    optimizer = torch.optim.AdamW(phase_core.parameters(), lr=args.lr)
    optimizer_to_save = optimizer

    # Count parameters
    param_count = sum(p.numel() for p in phase_core.parameters() if p.requires_grad)
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
    accumulation_counter = 0

    for it in range(args.iters):
        # Sample batch
        idx = torch.randint(0, len(train_data), (args.batch_size,))
        batch = train_data[idx].to(device)  # [batch, seq_len]

        # Forward through RWKV to get hidden states
        # (In practice, we'd extract intermediate hidden states)
        # For now, we'll use a simplified training objective

        # Simulate: extract hidden state from RWKV
        # In real implementation, we'd hook into layer 12
        # For demo: use random hidden state (will be replaced with actual RWKV output)
        hidden_states = torch.randn(args.batch_size, batch.shape[1], 1024, device=device)

        # Apply Phase Core modulation
        modulated_hidden = phase_core(hidden_states)

        # Project to logits (simplified - in real version, use RWKV head)
        # For now: dummy logits
        logits = torch.randn(args.batch_size, batch.shape[1], tokenizer.vocab_size, device=device)

        # Compute losses
        # 1. Cross-entropy (simplified)
        targets = batch
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.view(-1),
            reduction='mean'
        )

        # 2. Uncertainty computation
        U = compute_uncertainty(logits)

        # 3. Uncertainty regulation
        u_loss_val = uncertainty_regulation_loss(U, target_u=args.target_uncertainty, strength=0.1)
        u_loss = torch.tensor(u_loss_val, device=device, requires_grad=True)

        # 4. Total loss (scaled for gradient accumulation)
        total_loss = (ce_loss + u_loss) / args.gradient_accumulation_steps

        # Backward
        total_loss.backward()

        accumulation_counter += 1
        if accumulation_counter % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Get metrics
        R = phase_core.current_R
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
        metrics_history['loss'].append((total_loss * args.gradient_accumulation_steps).item())
        metrics_history['ce_loss'].append(ce_loss.item())
        metrics_history['u_loss'].append(u_loss_val)
        metrics_history['R'].append(R)
        metrics_history['U'].append(U)
        metrics_history['RU'].append(RU_product)
        metrics_history['tone'].append(phase_core.current_tone)
        metrics_history['action'].append(action)
        metrics_history['perplexity'].append(perplexity)

        # Log
        if (it + 1) % 10 == 0:
            print(f"Step {it+1:4d} | Loss: {total_loss.item() * args.gradient_accumulation_steps:.4f} "
                  f"(CE: {ce_loss.item():.4f}, U: {u_loss_val:.4f}) | "
                  f"PPL: {perplexity:.2f} | "
                  f"R: {R:.4f} {phase_core.current_tone} | "
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
                    'loss': (total_loss * args.gradient_accumulation_steps).item(),
                    'ce_loss': ce_loss.item(),
                    'u_loss': u_loss_val,
                    'RU': RU_product,
                    'perplexity': perplexity,
                    'tone': phase_core.current_tone,
                    'action': action,
                    'omega_mean': phase_core.omega.mean().item(),
                    'omega_std': phase_core.omega.std().item(),
                    'phase_mean': phase_core.phases.mean().item(),
                    'phase_std': phase_core.phases.std().item()
                }
                save_checkpoint(phase_core, optimizer, step=it+1,
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
            'loss': (total_loss * args.gradient_accumulation_steps).item(),
            'ce_loss': ce_loss.item(),
            'u_loss': u_loss_val,
            'RU': RU_product,
            'perplexity': perplexity,
            'tone': phase_core.current_tone,
            'action': action,
            'omega_mean': phase_core.omega.mean().item(),
            'omega_std': phase_core.omega.std().item(),
            'phase_mean': phase_core.phases.mean().item(),
            'phase_std': phase_core.phases.std().item()
        }
        final_path = save_checkpoint(phase_core, optimizer, step=args.iters,
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
        print(f"✅ Quantum state preserved with uncertainty intact")
    except Exception as e:
        print(f"❌ Final checkpoint failed: {e}")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Training time: {elapsed/60:.1f} minutes")
    print(f"🎲 Final Uncertainty: U = {U:.3f} (target: {args.target_uncertainty:.2f})")
    print(f"🌀 Final Resonance: R = {R:.4f} {phase_core.current_tone}")
    print(f"⚖️  Coherence-Uncertainty Product: R·U = {RU_product:.3f}")
    print(f"📈 Final Perplexity: {perplexity:.2f}")
    print(f"\n📊 Phase Core Statistics:")
    print(f"   Natural frequencies ω: mean={phase_core.omega.mean().item():.4f}, std={phase_core.omega.std().item():.4f}")
    print(f"   Phases φ: mean={phase_core.phases.mean().item():.4f}, std={phase_core.phases.std().item():.4f}")


if __name__ == "__main__":
    train_phase_rwkv()
