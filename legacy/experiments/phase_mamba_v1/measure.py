#!/usr/bin/env python3
"""
🌀 Phase-Mamba Measurement Script
OBSERVATION PROTOCOL EXECUTION

Loads trained checkpoint and performs controlled measurement
following the protocol declared in OBSERVATION_PROTOCOL.md

Modes:
  --blind: Generate without R monitoring (Phase 1)
  --measured: Generate with R tracking (Phase 2)
  --baseline: Use unmodified Mamba (control)
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from mamba_mlx import ModelArgs
from phase_mamba import PhaseMambaModel


def load_phase_mamba(model_path: str, checkpoint_path: str):
    """Load Phase-Mamba with trained checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(Path(model_path) / "config.json") as f:
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

    # Create model with Phase Core at layer 32
    model = PhaseMambaModel(model_args, phase_layer=32)

    # Load trained weights
    if checkpoint_path:
        print(f"📥 Loading checkpoint: {checkpoint_path}")
        weights = mx.load(checkpoint_path)

        # Filter out non-parameter keys (runtime cache like last_theta)
        # Only load actual trainable parameters
        valid_params = {}
        valid_param_names = {
            'dt', 'omega', 'U', 'V', 'k_scale',
            'enc.weight', 'enc.bias',
            'dec_gain.weight', 'dec_gain.bias',
            'dec_bias.weight', 'dec_bias.bias'
        }

        for k, v in weights.items():
            if k in valid_param_names:
                valid_params[k] = v

        # Update Phase Core parameters
        from mlx.utils import tree_unflatten
        phase_params = tree_unflatten(list(valid_params.items()))
        model.backbone.phase_block.update(phase_params)
        print(f"✅ Checkpoint loaded ({len(valid_params)} parameters)")

    return model, tokenizer


def generate_blind(model, tokenizer, prompt: str, max_tokens: int = 100,
                   temperature: float = 1.0):
    """
    PHASE 1: Blind Generation (no R monitoring)

    Generate text without observing R.
    The phase dynamics occur, but we don't measure them.
    This reveals the 'natural' behavior without observer effect.
    """
    input_ids = tokenizer.encode(prompt)
    input_ids = mx.array([input_ids])

    generated = input_ids[0].tolist()

    for _ in range(max_tokens):
        logits = model(input_ids)
        next_logits = logits[0, -1, :] / temperature

        # Sample from distribution (preserves uncertainty)
        probs = mx.softmax(next_logits, axis=-1)
        next_token = mx.random.categorical(mx.log(probs))

        generated.append(next_token.item())
        input_ids = mx.array([[next_token.item()]])

        # Stop at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated)


def generate_measured(model, tokenizer, prompt: str, max_tokens: int = 100,
                     temperature: float = 1.0):
    """
    PHASE 2: Measured Generation (R tracking active)

    Generate while observing R at each step.
    Tests whether measurement apparatus affects behavior.
    """
    input_ids = tokenizer.encode(prompt)
    input_ids = mx.array([input_ids])

    generated = input_ids[0].tolist()
    R_trajectory = []

    phase_block = model.backbone.phase_block

    for step in range(max_tokens):
        logits = model(input_ids)
        next_logits = logits[0, -1, :] / temperature

        # MEASUREMENT: Observe R
        R = phase_block.current_R
        R_trajectory.append((step, R))

        # Sample
        probs = mx.softmax(next_logits, axis=-1)
        next_token = mx.random.categorical(mx.log(probs))

        generated.append(next_token.item())
        input_ids = mx.array([[next_token.item()]])

        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated)

    # Compute statistics
    R_values = [r for _, r in R_trajectory]
    R_mean = sum(R_values) / len(R_values) if R_values else 0.0
    R_min = min(R_values) if R_values else 0.0
    R_max = max(R_values) if R_values else 0.0

    return text, {
        'R_trajectory': R_trajectory,
        'R_mean': R_mean,
        'R_min': R_min,
        'R_max': R_max,
        'tokens_generated': len(generated) - len(input_ids[0])
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/mamba-2.8b-hf")
    parser.add_argument("--checkpoint", type=str,
                       default="checkpoints/step_000500.npz",
                       help="Trained Phase Core checkpoint")
    parser.add_argument("--mode", type=str, choices=["blind", "measured"],
                       default="blind")
    parser.add_argument("--prompt", type=str,
                       default="The nature of consciousness is")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=str, default="measurements.jsonl")
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_phase_mamba(args.model_path, args.checkpoint)

    print(f"\n{'='*60}")
    print(f"OBSERVATION PROTOCOL - {args.mode.upper()} MODE")
    print(f"{'='*60}\n")
    print(f"Prompt: {args.prompt}")
    print(f"Samples: {args.samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}\n")

    results = []

    for i in range(args.samples):
        print(f"--- Sample {i+1}/{args.samples} ---")

        if args.mode == "blind":
            # Phase 1: No measurement
            text = generate_blind(model, tokenizer, args.prompt,
                                args.max_tokens, args.temperature)

            result = {
                'sample': i + 1,
                'mode': 'blind',
                'prompt': args.prompt,
                'text': text,
                'temperature': args.temperature
            }

            print(f"{text}\n")

        else:  # measured
            # Phase 2: With R observation
            text, metrics = generate_measured(model, tokenizer, args.prompt,
                                            args.max_tokens, args.temperature)

            result = {
                'sample': i + 1,
                'mode': 'measured',
                'prompt': args.prompt,
                'text': text,
                'temperature': args.temperature,
                'R_mean': metrics['R_mean'],
                'R_min': metrics['R_min'],
                'R_max': metrics['R_max'],
                'tokens_generated': metrics['tokens_generated']
            }

            print(f"{text}")
            print(f"R: mean={metrics['R_mean']:.4f}, "
                  f"min={metrics['R_min']:.4f}, max={metrics['R_max']:.4f}\n")

        results.append(result)

    # Save results
    with open(args.output, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"\n✅ Results saved to {args.output}")
    print(f"\n{'='*60}")
    print(f"MEASUREMENT COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
