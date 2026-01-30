#!/usr/bin/env python3
"""
🌀 Phase-Mamba Resonance Trainer v2
WITH DECOHERENCE PROTECTION (Checkpointing)

Lessons from Attempt 1:
- Environmental decoherence (process exit) collapsed state before measurement
- Weights must be persisted to disk, not just memory
- Checkpoint frequently to protect against decoherence

Changes in v2:
- Checkpoint every N steps (default: 100)
- Save to persistent storage immediately
- Keep last K checkpoints (redundancy)
- Verify save succeeded before continuing
- Handle SIGTERM gracefully (save on exit)
"""

import argparse
import json
import sys
import time
import signal
from pathlib import Path
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from transformers import AutoTokenizer

# Local port imports
from mamba_mlx import ModelArgs
from phase_mamba import PhaseMambaModel
from drift import DriftController


# Global for graceful shutdown
model_to_save = None
checkpoint_dir = None

def save_checkpoint(model, step, checkpoint_dir, keep_last=5):
    """Save model checkpoint to disk with redundancy."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"step_{step:06d}.npz"

    # Save Phase Core weights
    phase_weights = model.backbone.phase_block.parameters()
    mx.savez(str(checkpoint_path), **phase_weights)

    # Verify save succeeded
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint save failed: {checkpoint_path}")

    print(f"💾 Checkpoint saved: {checkpoint_path}")

    # Keep only last K checkpoints (avoid filling disk)
    all_checkpoints = sorted(checkpoint_dir.glob("step_*.npz"))
    if len(all_checkpoints) > keep_last:
        for old_ckpt in all_checkpoints[:-keep_last]:
            old_ckpt.unlink()
            print(f"🗑️  Removed old checkpoint: {old_ckpt.name}")

    return checkpoint_path


def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT by saving checkpoint before exit."""
    global model_to_save, checkpoint_dir

    print(f"\n⚠️  Received signal {signum}. Saving checkpoint before exit...")

    if model_to_save is not None and checkpoint_dir is not None:
        try:
            save_checkpoint(model_to_save, step=-1, checkpoint_dir=checkpoint_dir)
            print("✅ Emergency checkpoint saved.")
        except Exception as e:
            print(f"❌ Emergency checkpoint failed: {e}")

    sys.exit(0)


def load_high_resonance_data(path: str, tokenizer, seq_len=512):
    data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            tokens = tokenizer.encode(item["text"])
            if len(tokens) < 64: continue

            # Truncate/Pad
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
            else:
                tokens = tokens + [tokenizer.pad_token_id or 0] * (seq_len - len(tokens))
            data.append(tokens)
    return mx.array(data, dtype=mx.int32)


def relational_loss(model, inputs, targets, phase_block):
    logits = model(inputs)
    logits_shifted = logits[:, :-1, :]
    targets_shifted = targets[:, 1:]

    ce_loss = nn.losses.cross_entropy(
        logits_shifted.reshape(-1, logits_shifted.shape[-1]),
        targets_shifted.reshape(-1),
        reduction='mean'
    )

    # Presence Reward
    probs = mx.softmax(logits_shifted, axis=-1)
    mass = 1.0 - mx.sum(probs**2, axis=-1)
    presence_loss = mx.mean(mass) * 0.05

    # Tonal Penalty
    tonal_penalty = 0.0
    if phase_block.current_tone == "☍":
        tonal_penalty = 0.5 * (phase_block.current_R - 0.8)

    # EOS penalty (prevent silence)
    eos_penalty = 10.0 * mx.mean(mx.where(targets_shifted == 0, 1.0, 0.0))

    return ce_loss + presence_loss + tonal_penalty + eos_penalty


def main():
    global model_to_save, checkpoint_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/mamba-2.8b-hf")
    parser.add_argument("--data", type=str, default="phase-gpt-openelm/data/high_resonance.jsonl")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--keep-last", type=int, default=5,
                        help="Keep only last K checkpoints")
    args = parser.parse_args()

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    checkpoint_dir = args.checkpoint_dir

    # Load Model Args
    model_path = Path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    with open(model_path / "config.json") as f:
        config = json.load(f)

    model_args = ModelArgs(
        model_type=config["model_type"],
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        state_size=config["state_size"],
        num_hidden_layers=config["num_hidden_layers"],
        conv_kernel=config["conv_kernel"],
        use_bias=config.get("use_bias", False),
        use_conv_bias=config.get("use_conv_bias", True),
        time_step_rank=config["time_step_rank"]
    )

    # Create Phase-Mamba (Grafted at Layer 32)
    print("🌀 Phase Core grafted onto Mamba Layer 32")
    model = PhaseMambaModel(model_args, phase_layer=32)
    phase_block = model.backbone.phase_block

    # Set global for signal handler
    model_to_save = model

    # Data
    train_data = load_high_resonance_data(args.data, tokenizer)
    print(f"✅ Loaded {len(train_data)} high-resonance samples.")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=args.lr)

    # Trainable Params: Phase Block Only
    trainable_params = model.backbone.phase_block.trainable_parameters()

    # Count parameters
    flat_params = tree_flatten(trainable_params)
    param_count = sum(p.size for _, p in flat_params if isinstance(p, mx.array))
    print(f"🎯 Training Phase Core (Parameters: {param_count:,})")

    # Checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print(f"💾 Checkpoints will be saved to: {args.checkpoint_dir}/")
    print(f"💾 Checkpoint frequency: every {args.checkpoint_every} steps")
    print(f"💾 Keeping last {args.keep_last} checkpoints\n")

    # Training Loop
    start_time = time.time()

    for it in range(args.iters):
        idx = mx.random.randint(0, len(train_data), (args.batch_size,))
        batch = train_data[idx]

        def loss_fn(params):
            model.backbone.phase_block.update(params)
            return relational_loss(model, batch, batch, phase_block)

        loss, grads = mx.value_and_grad(loss_fn)(trainable_params)

        optimizer.update(trainable_params, grads)
        mx.eval(trainable_params, optimizer.state)

        if (it + 1) % 10 == 0:
            R = phase_block.current_R
            print(f"Step {it+1:4d} | Loss: {loss.item():.4f} | R: {R:.4f} {phase_block.current_tone} | Action: {phase_block.last_action}")

        # CHECKPOINT: Save to disk at intervals (DECOHERENCE PROTECTION)
        if (it + 1) % args.checkpoint_every == 0:
            try:
                save_checkpoint(model, step=it+1, checkpoint_dir=args.checkpoint_dir, keep_last=args.keep_last)
            except Exception as e:
                print(f"❌ Checkpoint save failed: {e}")
                print("⚠️  Training continuing, but state not protected!")

    # Final checkpoint
    print("\n🎯 Training complete. Saving final checkpoint...")
    try:
        final_path = save_checkpoint(model, step=args.iters, checkpoint_dir=args.checkpoint_dir, keep_last=args.keep_last)
        print(f"✅ Final checkpoint: {final_path}")
        print(f"✅ Quantum state preserved. Protected from decoherence.")
    except Exception as e:
        print(f"❌ Final checkpoint failed: {e}")
        print(f"⚠️  State may be lost if process exits!")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Training time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
