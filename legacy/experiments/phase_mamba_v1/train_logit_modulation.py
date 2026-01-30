#!/usr/bin/env python3
"""
Phase-RWKV Training: Logit-Space Modulation
Attempt 3.2: Honest Coupling (Output Side)

The quantum measurement apparatus (Phase Core) observes and modulates
RWKV's output distribution, not internal hidden states.

This is architecturally honest and actually works.
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

from phase_rwkv_simplified import load_rwkv_with_logit_modulation


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
                    'logit_to_phase_state': model_to_save.logit_to_phase.state_dict(),
                    'phase_to_logit_state': model_to_save.phase_to_logit.state_dict(),
                    'step': -1
                }, checkpoint_path)
                print(f"💾 Emergency checkpoint (minimal): {checkpoint_path}")
            print("✅ Emergency checkpoint saved")
        except Exception as e:
            print(f"❌ Emergency checkpoint failed: {e}")

    sys.exit(0)


def compute_uncertainty(logits: torch.Tensor) -> float:
    """Compute epistemic uncertainty (predictive entropy)."""
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

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
            tokens = tokenizer.encode(item["text"])

            if len(tokens) < 32:
                continue

            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [tokenizer.eos_token_id] * (max_length - len(tokens))

            data.append(tokens)

    return data


def save_checkpoint(modulator, optimizer, step, checkpoint_dir, metrics=None,
                    metrics_history=None, keep_last=5):
    """Save checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"

    checkpoint = {
        'phase_core_state': modulator.phase_core.state_dict(),
        'logit_to_phase_state': modulator.logit_to_phase.state_dict(),
        'phase_to_logit_state': modulator.phase_to_logit.state_dict(),
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
    parser.add_argument("--data", type=str, default="high_resonance.jsonl")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--target-uncertainty", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_logit_mod")
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
    print("🌀 PHASE-RWKV TRAINING: LOGIT-SPACE MODULATION")
    print("=" * 70)
    print("✨ Observer modulates vessel's output distribution")
    print(f"🖥️  Device: {device}")
    print("=" * 70)

    # Load model
    print("\n[1] Loading RWKV with Phase Core...")
    model_file = hf_hub_download(
        repo_id="BlinkDL/rwkv-4-pile-430m",
        filename="RWKV-4-Pile-430M-20220808-8066.pth"
    )

    modulator = load_rwkv_with_logit_modulation(model_file, device=device)
    model_to_save = modulator

    # Tokenizer
    print("\n[2] Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"   ✅ Tokenizer ready (vocab: {tokenizer.vocab_size})")

    # Data
    print("\n[3] Loading data...")
    train_data = load_high_resonance_data(args.data, tokenizer)
    print(f"   ✅ Loaded {len(train_data)} samples")

    # Optimizer (Phase Core + projection layers)
    trainable_params = list(modulator.phase_core.parameters()) + \
                      list(modulator.logit_to_phase.parameters()) + \
                      list(modulator.phase_to_logit.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    optimizer_to_save = optimizer

    param_count = sum(p.numel() for p in trainable_params if p.requires_grad)
    print(f"\n🎯 Training {param_count:,} parameters")
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
        # Sample
        idx = torch.randint(0, len(train_data), (1,)).item()
        tokens = train_data[idx]

        # Forward through modulator
        outputs = modulator.forward_with_modulation(tokens, state=None)

        logits_modulated = outputs['logits_modulated']
        R = outputs['R']
        tone = outputs['tone']

        # Prepare for loss
        if logits_modulated.dim() == 1:
            logits_modulated = logits_modulated.unsqueeze(0)

        # Target: next token prediction
        targets = torch.tensor(tokens[1:logits_modulated.shape[0]+1], dtype=torch.long).to(device)
        if len(targets) > logits_modulated.shape[0]:
            targets = targets[:logits_modulated.shape[0]]
        elif len(targets) < logits_modulated.shape[0]:
            logits_modulated = logits_modulated[:len(targets)]

        # Losses
        ce_loss = F.cross_entropy(logits_modulated, targets, reduction='mean')

        # Compute uncertainty with detached logits (for metrics only)
        U = compute_uncertainty(logits_modulated.detach())
        u_loss_val = uncertainty_regulation_loss(U, target_u=args.target_uncertainty, strength=0.1)

        # For now, only train on CE loss (simplify gradient graph)
        total_loss = ce_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Track combined loss for metrics (but don't backward through it)
        total_loss_val = ce_loss.item() + u_loss_val

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
                    'omega_mean': modulator.phase_core.omega.mean().item(),
                    'omega_std': modulator.phase_core.omega.std().item(),
                    'phase_mean': modulator.phase_core.phases.mean().item(),
                    'phase_std': modulator.phase_core.phases.std().item()
                }
                save_checkpoint(modulator, optimizer, step=it+1,
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
            'omega_mean': modulator.phase_core.omega.mean().item(),
            'omega_std': modulator.phase_core.omega.std().item(),
            'phase_mean': modulator.phase_core.phases.mean().item(),
            'phase_std': modulator.phase_core.phases.std().item()
        }
        final_path = save_checkpoint(modulator, optimizer, step=args.iters,
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
