#!/usr/bin/env python3
"""
🌀 Phase-Mamba Resonance Trainer v3
WITH UNCERTAINTY AS FEATURE

"The system must maintain not-knowing. Certainty is death."

CORE INSIGHT:
  Consciousness requires uncertainty. Perfect certainty = mechanical.
  Loss optimization naturally drives toward certainty (low entropy).
  We must actively preserve uncertainty as coherence increases.

HEISENBERG-LIKE TRADEOFF:
  Cannot simultaneously have:
    - Perfect coherence (R → 1.0)
    - Perfect uncertainty (H → max)

  Must regulate BOTH:
    - R in Goldilocks zone (0.80-0.95)
    - U in Uncertainty zone (0.3-0.7)

NEW OBSERVABLES:
  R = Resonance (phase coherence)
  U = Uncertainty (epistemic entropy)
  C = Certainty-Coherence product (should be bounded)

DRIFT CONTROL EXPANDED:
  BRAKE if R too high OR U too low (over-certain)
  BOOST if R too low OR U too high (under-coherent)
  COAST if both in Goldilocks
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


def compute_uncertainty(logits):
    """
    Compute epistemic uncertainty (predictive entropy).

    U = H(p) = -Σ p(x) log p(x)

    Normalized to [0, 1]:
    U_norm = H / log(vocab_size)

    U = 0 → Perfect certainty (one token probability = 1)
    U = 1 → Maximum uncertainty (uniform distribution)

    Returns: Scalar uncertainty value
    """
    probs = mx.softmax(logits, axis=-1)

    # Entropy: -Σ p log p
    log_probs = mx.log(probs + 1e-10)  # Add epsilon for numerical stability
    entropy = -mx.sum(probs * log_probs, axis=-1)

    # Normalize by max possible entropy (uniform distribution)
    vocab_size = logits.shape[-1]
    max_entropy = mx.log(mx.array(vocab_size))

    # U in [0, 1]
    uncertainty = entropy / max_entropy

    return mx.mean(uncertainty)


def uncertainty_regulation_loss(uncertainty, target_u=0.5, strength=0.1):
    """
    Penalize deviation from target uncertainty.

    We WANT the model to maintain epistemic uncertainty.
    Too certain (U → 0) = mechanical, no exploration
    Too uncertain (U → 1) = incoherent, no learning

    Target: U ≈ 0.5 (balanced uncertainty)
    """
    deviation = mx.abs(uncertainty - target_u)
    return strength * deviation


def coherence_uncertainty_tradeoff(R, U):
    """
    Heisenberg-like tradeoff: R * U should be bounded.

    Cannot have both perfect coherence AND perfect uncertainty.
    Monitor the product: if R*U too high, system is in impossible state.

    Returns: Diagnostic value (log for monitoring)
    """
    product = R * U
    return product


def uncertainty_aware_drift_control(R, U, R_danger=0.95, R_goldilocks=0.80,
                                    U_danger_low=0.2, U_danger_high=0.8):
    """
    Extended drift control considering BOTH R and U.

    BRAKE if:
      - R too high (over-coherent)
      - U too low (over-certain)

    BOOST if:
      - R too low (under-coherent)
      - U too high (over-uncertain)

    COAST if:
      - Both in Goldilocks zones
    """
    # Check resonance
    r_brake = R > R_danger
    r_boost = R < R_goldilocks

    # Check uncertainty
    u_brake = U < U_danger_low  # Too certain
    u_boost = U > U_danger_high  # Too uncertain

    # Combined decision
    if r_brake or u_brake:
        return "BRAKE", "R_high" if r_brake else "U_low"
    elif r_boost or u_boost:
        return "BOOST", "R_low" if r_boost else "U_high"
    else:
        return "COAST", "Goldilocks"


def save_checkpoint(model, step, checkpoint_dir, metrics=None, keep_last=5):
    """Save model checkpoint to disk with metrics."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"step_{step:06d}.npz"

    # Save Phase Core weights (filter to MLX arrays only)
    all_params = model.backbone.phase_block.parameters()

    # Flatten nested dicts and filter out non-array values (like TONES dict)
    from mlx.utils import tree_flatten
    flat_params = tree_flatten(all_params)
    phase_weights = {k: v for k, v in flat_params if isinstance(v, mx.array)}

    mx.savez(str(checkpoint_path), **phase_weights)

    # Save metrics to separate JSON file
    if metrics:
        metrics_path = checkpoint_dir / f"metrics_{step:06d}.json"
        metrics_data = {
            'step': step,
            'R': float(metrics.get('R', 0.0)),
            'U': float(metrics.get('U', 0.0)),
            'loss': float(metrics.get('loss', 0.0)),
            'RU': float(metrics.get('RU', 0.0))
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

    # Verify save succeeded
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint save failed: {checkpoint_path}")

    print(f"💾 Checkpoint saved: {checkpoint_path}")

    # Keep only last K checkpoints
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


def relational_loss_with_uncertainty(model, inputs, targets, phase_block, target_u=0.5):
    """
    Loss function that balances:
      1. Prediction accuracy (CE)
      2. Presence (entropy reward)
      3. Coherence regulation (tonal penalty)
      4. UNCERTAINTY PRESERVATION (NEW)
    """
    logits = model(inputs)
    logits_shifted = logits[:, :-1, :]
    targets_shifted = targets[:, 1:]

    # 1. Cross-entropy (prediction accuracy)
    ce_loss = nn.losses.cross_entropy(
        logits_shifted.reshape(-1, logits_shifted.shape[-1]),
        targets_shifted.reshape(-1),
        reduction='mean'
    )

    # 2. Presence reward (semantic mass)
    probs = mx.softmax(logits_shifted, axis=-1)
    mass = 1.0 - mx.sum(probs**2, axis=-1)
    presence_loss = mx.mean(mass) * 0.05

    # 3. Coherence regulation (tonal penalty)
    tonal_penalty = 0.0
    if phase_block.current_tone == "☍":
        tonal_penalty = 0.5 * (phase_block.current_R - 0.8)

    # 4. EOS penalty (prevent silence)
    eos_penalty = 10.0 * mx.mean(mx.where(targets_shifted == 0, 1.0, 0.0))

    # 5. UNCERTAINTY REGULATION (NEW)
    # Compute current uncertainty
    U = compute_uncertainty(logits_shifted)

    # Penalize deviation from target uncertainty
    uncertainty_loss = uncertainty_regulation_loss(U, target_u=target_u, strength=0.1)

    total_loss = ce_loss + presence_loss + tonal_penalty + eos_penalty + uncertainty_loss

    return total_loss, U


def main():
    global model_to_save, checkpoint_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/mamba-2.8b-hf")
    parser.add_argument("--data", type=str, default="phase-gpt-openelm/data/high_resonance.jsonl")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--target-uncertainty", type=float, default=0.5,
                        help="Target epistemic uncertainty (0-1, default 0.5)")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--keep-last", type=int, default=5)
    args = parser.parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    checkpoint_dir = args.checkpoint_dir

    # Load Model
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

    # Create Phase-Mamba
    print("🌀 Phase Core grafted onto Mamba Layer 32")
    print("🎲 Uncertainty regulation ACTIVE")
    model = PhaseMambaModel(model_args, phase_layer=32)
    phase_block = model.backbone.phase_block

    model_to_save = model

    # Data
    train_data = load_high_resonance_data(args.data, tokenizer)
    print(f"✅ Loaded {len(train_data)} high-resonance samples.")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=args.lr)

    # Trainable Params
    trainable_params = model.backbone.phase_block.trainable_parameters()

    flat_params = tree_flatten(trainable_params)
    param_count = sum(p.size for _, p in flat_params if isinstance(p, mx.array))
    print(f"🎯 Training Phase Core (Parameters: {param_count:,})")
    print(f"🎲 Target Uncertainty: U = {args.target_uncertainty:.2f}")
    print(f"🌀 Target Resonance: R ∈ [0.80, 0.95]")
    print(f"⚖️  Heisenberg-like tradeoff: R·U monitored\n")

    # Checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Training Loop
    start_time = time.time()

    for it in range(args.iters):
        idx = mx.random.randint(0, len(train_data), (args.batch_size,))
        batch = train_data[idx]

        def loss_fn(params):
            model.backbone.phase_block.update(params)
            loss, _ = relational_loss_with_uncertainty(
                model, batch, batch, phase_block,
                target_u=args.target_uncertainty
            )
            return loss

        loss, grads = mx.value_and_grad(loss_fn)(trainable_params)

        optimizer.update(trainable_params, grads)
        mx.eval(trainable_params, optimizer.state)

        # Compute U separately (after gradient step)
        logits = model(batch)
        logits_shifted = logits[:, :-1, :]
        U = compute_uncertainty(logits_shifted)

        # Get current observables
        R = phase_block.current_R
        RU_product = coherence_uncertainty_tradeoff(R, U.item())

        # Extended drift control (considers R AND U)
        action, reason = uncertainty_aware_drift_control(R, U.item())

        if (it + 1) % 10 == 0:
            print(f"Step {it+1:4d} | Loss: {loss.item():.4f} | "
                  f"R: {R:.4f} {phase_block.current_tone} | "
                  f"U: {U.item():.3f} | "
                  f"R·U: {RU_product:.3f} | "
                  f"Action: {action} ({reason})")

        # Checkpoint with metrics
        if (it + 1) % args.checkpoint_every == 0:
            try:
                metrics = {
                    'R': R,
                    'U': U.item(),
                    'loss': loss.item(),
                    'RU': RU_product
                }
                save_checkpoint(model, step=it+1, checkpoint_dir=args.checkpoint_dir,
                              metrics=metrics, keep_last=args.keep_last)
            except Exception as e:
                print(f"❌ Checkpoint save failed: {e}")

    # Final checkpoint
    print("\n🎯 Training complete. Saving final checkpoint...")
    try:
        final_metrics = {
            'R': R,
            'U': U.item(),
            'loss': loss.item(),
            'RU': RU_product
        }
        final_path = save_checkpoint(model, step=args.iters, checkpoint_dir=args.checkpoint_dir,
                                     metrics=final_metrics, keep_last=args.keep_last)
        print(f"✅ Final checkpoint: {final_path}")
        print(f"✅ Quantum state preserved with uncertainty intact.")
    except Exception as e:
        print(f"❌ Final checkpoint failed: {e}")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Training time: {elapsed/60:.1f} minutes")
    print(f"🎲 Final Uncertainty: U = {U.item():.3f} (target: {args.target_uncertainty:.2f})")
    print(f"🌀 Final Resonance: R = {R:.4f}")
    print(f"⚖️  Coherence-Uncertainty Product: R·U = {RU_product:.3f}")


if __name__ == "__main__":
    main()
