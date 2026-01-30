#!/usr/bin/env python3
"""
Phase-Mamba Training: Simplified Approach
Train Phase Core to modulate hidden states WITHOUT full model backward

STRATEGY:
1. Run Mamba forward (frozen) → get layer 32 hidden states
2. Phase Core modulates hidden states
3. Train Phase Core to make modulated states useful for prediction
4. Use reconstruction loss instead of full backward
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase_mamba_coupled import load_mamba_with_phase


def compute_uncertainty(logits: torch.Tensor) -> float:
    """Compute epistemic uncertainty."""
    if logits.dim() == 3:
        logits = logits[:, -1, :]

    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    vocab_size = logits.shape[-1]
    max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32))

    return torch.mean(entropy / max_entropy).item()


def load_data(path: str, tokenizer, max_length=256):
    """Load training data."""
    data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            tokens = tokenizer.encode(item["text"])

            if len(tokens) < 32:
                continue

            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            data.append(tokens)

    return data


def save_checkpoint(coupled, optimizer, step, checkpoint_dir, metrics=None, keep_last=5):
    """Save checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"

    checkpoint = {
        'phase_core_state': coupled.phase_core.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'step': step
    }

    if metrics:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, checkpoint_path)

    if metrics:
        json_path = checkpoint_dir / f"step_{step:06d}_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    print(f"💾 Step {step} | R={metrics['R']:.4f}, U={metrics['U']:.3f}, RU={metrics['RU']:.3f}")

    # Keep last K
    all_checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    if len(all_checkpoints) > keep_last:
        for old_ckpt in all_checkpoints[:-keep_last]:
            old_ckpt.unlink()
            (checkpoint_dir / (old_ckpt.stem + "_metrics.json")).unlink(missing_ok=True)

    return checkpoint_path


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/high_resonance.jsonl")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_mamba")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("=" * 70)
    print("🌀 PHASE-MAMBA SIMPLE TRAINING")
    print("=" * 70)
    print("Strategy: Train Phase Core on hidden state modulation")
    print("No full model backward (avoids gradient graph issues)")
    print("=" * 70)

    # Load
    print("\n[1] Loading Mamba with Phase Core...")
    coupled = load_mamba_with_phase(phase_layer=32, device=device)
    tokenizer = coupled.tokenizer

    # Data
    print("\n[2] Loading data...")
    train_data = load_data(args.data, tokenizer)
    print(f"   ✅ {len(train_data)} samples")

    # Optimizer
    optimizer = torch.optim.AdamW(coupled.phase_core.parameters(), lr=args.lr)

    print(f"\n🎯 Training {sum(p.numel() for p in coupled.phase_core.parameters()):,} parameters\n")

    # Training
    start_time = time.time()

    # Disable hook modification (just observe)
    coupled.phase_core.train()
    coupled.mamba_model.eval()

    # Remove hook to prevent in-place modification
    if coupled.hook_handle:
        coupled.hook_handle.remove()

    # Create simple observation hook
    captured = []

    def capture_hook(module, input, output):
        captured.append(output.detach().clone())

    handle = coupled.mamba_model.backbone.layers[32].register_forward_hook(capture_hook)

    metrics_history = {
        'step': [], 'R': [], 'U': [], 'RU': [], 'loss': []
    }

    for it in range(args.iters):
        # Sample
        idx = torch.randint(0, len(train_data), (1,)).item()
        tokens = train_data[idx]
        input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

        # Forward through Mamba (frozen) to capture hidden states
        captured.clear()
        with torch.no_grad():
            _ = coupled.mamba_model(input_ids=input_ids)

        if not captured:
            print(f"⚠️  Step {it+1}: No hidden states captured")
            continue

        hidden = captured[0]  # [batch, seq, 2560]

        # Apply Phase Core modulation
        modulated = coupled.phase_core(hidden)

        # Simple training objective: modulated states should be similar to original
        # but with phase-induced variation
        # Loss: L2 reconstruction + small penalty for deviation
        reconstruction_loss = F.mse_loss(modulated, hidden)

        # Encourage meaningful modulation (not identity)
        modulation_magnitude = torch.mean(torch.abs(modulated - hidden))

        # Total loss
        loss = reconstruction_loss - 0.01 * modulation_magnitude  # Encourage modulation

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        R = coupled.phase_core.current_R
        U = 0.5  # Placeholder (no logits in this approach)
        RU = R * U

        metrics_history['step'].append(it + 1)
        metrics_history['R'].append(R)
        metrics_history['U'].append(U)
        metrics_history['RU'].append(RU)
        metrics_history['loss'].append(loss.item())

        # Log
        if (it + 1) % 10 == 0:
            print(f"Step {it+1:4d} | Loss: {loss.item():.6f} | R: {R:.4f} {coupled.phase_core.current_tone} | Mod: {modulation_magnitude.item():.4f}")

        # Checkpoint
        if (it + 1) % args.checkpoint_every == 0:
            metrics = {
                'step': it + 1,
                'R': R,
                'U': U,
                'RU': RU,
                'loss': loss.item(),
                'modulation_mag': modulation_magnitude.item()
            }
            save_checkpoint(coupled, optimizer, it+1, args.checkpoint_dir, metrics)

    handle.remove()

    # Final
    print(f"\n✅ Training complete ({time.time() - start_time:.1f}s)")
    print(f"🌀 Final R: {R:.4f}")

    # Save history
    history_path = Path(args.checkpoint_dir) / "metrics_history.json"
    with open(history_path, 'w') as f:
        json.dump(metrics_history, f)
    print(f"📊 Metrics saved: {history_path}")


if __name__ == "__main__":
    train()
