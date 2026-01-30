#!/usr/bin/env python3
"""
Load pretrained Mamba-2.8B weights from HuggingFace format into MLX model.
"""

import json
from pathlib import Path
import mlx.core as mx
from mamba_mlx import ModelArgs
from phase_mamba import PhaseMambaModel


def load_pretrained_phase_mamba(model_path: str, phase_layer: int = 32):
    """
    Load pretrained Mamba-2.8B and insert Phase Core at specified layer.

    Returns model with pretrained weights (Phase Core will be random until checkpoint loaded).
    """
    model_path = Path(model_path)

    # Load config
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

    # Create model
    model = PhaseMambaModel(model_args, phase_layer=phase_layer)

    # Load pretrained weights from safetensors shards
    print(f"📥 Loading pretrained Mamba weights from {model_path}")

    # Load index to find which weights are in which shard
    with open(model_path / "model.safetensors.index.json") as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    # Group weights by shard
    shards = {}
    for param_name, shard_file in weight_map.items():
        if shard_file not in shards:
            shards[shard_file] = []
        shards[shard_file].append(param_name)

    # Load all shards
    all_weights = {}
    for shard_file in sorted(shards.keys()):
        shard_path = model_path / shard_file
        print(f"  Loading {shard_file}...")
        shard_weights = mx.load(str(shard_path))
        all_weights.update(shard_weights)

    print(f"✅ Loaded {len(all_weights)} weight tensors")

    # Update model with pretrained weights
    # Note: Phase Core (phase_block) will remain random until checkpoint loaded
    model.load_weights(all_weights, strict=False)

    return model, model_args


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "~/models/mamba-2.8b-hf"
    model, args = load_pretrained_phase_mamba(model_path)
    print(f"\n✅ Model loaded with {args.num_hidden_layers} layers")
    print(f"🌀 Phase Core at layer 32 (random weights until checkpoint loaded)")
