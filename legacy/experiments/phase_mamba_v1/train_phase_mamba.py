#!/usr/bin/env python3
"""
Phase-Mamba Training Script
ATTEMPT 4: With verified pretrained weights

The observer (Phase Core) modulates the vessel (Mamba SSM) at layer 32.
Uncertainty is preserved, not eliminated.
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

from phase_mamba_coupled import load_mamba_with_phase


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
            print("✅ Emergency checkpoint saved")
        except Exception as e:
            print(f"❌ Emergency checkpoint failed: {e}")

    sys.exit(0)


def compute_uncertainty(logits: torch.Tensor) -> float:
    """Compute epistemic uncertainty (predictive entropy)."""
    if logits.dim() == 3:
        logits = logits[:, -1, :]  # Last token logits

    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    vocab_size = logits.shape[-1]
    max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32))

    uncertainty = entropy / max_entropy
    return torch.mean(uncertainty).item()


def uncertainty_regulation_loss(U: float, target_u: float = 0.5, strength: float = 0.1) -> float:
    """Penalize deviation from target uncertainty."""
    deviation = abs(U - target_u)
    return strength * deviation


def load_high_resonance_data(path: str, tokenizer, max_length=256):
    """Load high-resonance training data."""
    data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            text = item["text"]

            # Tokenize
            tokens = tokenizer.encode(text)

            if len(tokens) < 32:
                continue

            # Truncate/pad
            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            data.append(tokens)

    return data


def save_checkpoint(coupled_model, optimizer, step, checkpoint_dir, metrics=None,
                    metrics_history=None, keep_last=5):
    """Save Phase Core checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"

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
            json_file = old_ckpt.with_suffix('').name + "_metrics.json"
            json_path = checkpoint_dir / json_file
            if json_path.exists():
                json_path.unlink()
            old_ckpt.unlink()
            print(f"🗑️  Removed old checkpoint: {old_ckpt.name}")

    return checkpoint_path


def train():
    """Main training function."""
    global model_to_save, optimizer_to_save, checkpoint_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/high_resonance.jsonl")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-uncertainty", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_mamba")
    parser.add_argument("--keep-last", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    checkpoint_dir = args.checkpoint_dir

    # Device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    print("=" * 70)
    print("🌀 PHASE-MAMBA TRAINING - ATTEMPT 4")
    print("=" * 70)
    print("✨ Observer modulates vessel's hidden states at layer 32")
    print("⚠️  PRETRAINED WEIGHTS VERIFIED (not Attempt 2 failure)")
    print(f"🖥️  Device: {device}")
    print("=" * 70)

    # Load coupled model with weight verification
    print("\n[1] Loading Mamba with Phase Core...")
    try:
        coupled = load_mamba_with_phase(phase_layer=32, device=device)
        model_to_save = coupled
    except RuntimeError as e:
        print(f"\n❌ CRITICAL FAILURE: {e}")
        print("❌ Training aborted to prevent Attempt 2 repeat")
        sys.exit(1)

    tokenizer = coupled.tokenizer

    # Data
    print("\n[2] Loading high-resonance data...")
    train_data = load_high_resonance_data(args.data, tokenizer)
    print(f"   ✅ Loaded {len(train_data)} samples")

    # Optimizer (only Phase Core)
    optimizer = torch.optim.AdamW(coupled.phase_core.parameters(), lr=args.lr)
    optimizer_to_save = optimizer

    param_count = sum(p.numel() for p in coupled.phase_core.parameters() if p.requires_grad)
    print(f"\n🎯 Training Phase Core: {param_count:,} parameters")
    print(f"🎲 Target Uncertainty: U = {args.target_uncertainty:.2f}")
    print(f"🌀 Target Resonance: R ∈ [0.80, 0.95]")
    print(f"⚖️  R·U tradeoff monitored\n")

    # Metrics
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

    coupled.phase_core.train()  # Phase Core in training mode
    coupled.mamba_model.eval()  # Mamba frozen

    for it in range(args.iters):
        # Sample
        idx = torch.randint(0, len(train_data), (1,)).item()
        tokens = train_data[idx]

        # Convert to tensor
        input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

        # Forward
        outputs = coupled.forward_for_training(input_ids=input_ids)

        logits = outputs['logits']
        R = outputs['R']
        tone = outputs['tone']

        # Compute loss
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # CE loss
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )

        # Uncertainty
        U = compute_uncertainty(logits.detach())
        u_loss_val = uncertainty_regulation_loss(U, target_u=args.target_uncertainty, strength=0.1)

        # Total loss (CE only for gradients, track U separately)
        total_loss = ce_loss
        total_loss_val = ce_loss.item() + u_loss_val

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Metrics
        RU_product = R * U
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

        # Store
        metrics_history['step'].append(it + 1)
        metrics_history['loss'].append(total_loss_val)
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
            print(f"Step {it+1:4d} | Loss: {total_loss_val:.4f} "
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
                    'loss': total_loss_val,
                    'ce_loss': ce_loss.item(),
                    'u_loss': u_loss_val,
                    'RU': RU_product,
                    'perplexity': perplexity,
                    'tone': tone,
                    'action': action,
                    'omega_mean': coupled.phase_core.omega.mean().item(),
                    'omega_std': coupled.phase_core.omega.std().item(),
                    'phase_mean': coupled.phase_core.phases.mean().item(),
                    'phase_std': coupled.phase_core.phases.std().item()
                }
                save_checkpoint(coupled, optimizer, step=it+1,
                              checkpoint_dir=args.checkpoint_dir,
                              metrics=checkpoint_metrics,
                              metrics_history=metrics_history,
                              keep_last=args.keep_last)
            except Exception as e:
                print(f"❌ Checkpoint save failed: {e}")

    # Final
    print("\n🎯 Training complete. Saving final checkpoint...")
    try:
        final_metrics = {
            'step': args.iters,
            'R': R,
            'U': U,
            'loss': total_loss_val,
            'ce_loss': ce_loss.item(),
            'u_loss': u_loss_val,
            'RU': RU_product,
            'perplexity': perplexity,
            'tone': tone,
            'action': action,
            'omega_mean': coupled.phase_core.omega.mean().item(),
            'omega_std': coupled.phase_core.omega.std().item(),
            'phase_mean': coupled.phase_core.phases.mean().item(),
            'phase_std': coupled.phase_core.phases.std().item()
        }
        final_path = save_checkpoint(coupled, optimizer, step=args.iters,
                                     checkpoint_dir=args.checkpoint_dir,
                                     metrics=final_metrics,
                                     metrics_history=metrics_history,
                                     keep_last=args.keep_last)

        history_path = Path(args.checkpoint_dir) / "metrics_history.json"
        with open(history_path, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        print(f"📊 Metrics history saved: {history_path}")

        print(f"✅ Final checkpoint: {final_path}")
        print(f"✅ Observer-vessel entanglement preserved")
    except Exception as e:
        print(f"❌ Final checkpoint failed: {e}")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Training time: {elapsed/60:.1f} minutes")
    print(f"🎲 Final Uncertainty: U = {U:.3f} (target: {args.target_uncertainty:.2f})")
    print(f"🌀 Final Resonance: R = {R:.4f} {tone}")
    print(f"⚖️  Coherence-Uncertainty Product: R·U = {RU_product:.3f}")
    print(f"📈 Final Perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    train()
