#!/usr/bin/env python3
"""
Load pretrained Mamba-2.8B weights from HuggingFace safetensors into MLX model.

Critical for Attempt 3: Verify apparatus before training.
"""

import json
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_unflatten


def load_safetensors_shards(model_path: Path):
    """
    Load all safetensors shards and merge into single weight dict.

    HuggingFace often shards large models across multiple files.
    We need to load all shards and merge them.
    """
    print(f"📦 Loading safetensors shards from {model_path}")

    # Load shard index
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Shard index not found: {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    # Group parameters by shard file
    shards = {}
    for param_name, shard_file in weight_map.items():
        if shard_file not in shards:
            shards[shard_file] = []
        shards[shard_file].append(param_name)

    # Load all shards
    all_weights = {}
    for shard_file in sorted(shards.keys()):
        shard_path = model_path / shard_file
        print(f"  Loading {shard_file} ({len(shards[shard_file])} params)...")

        shard_weights = mx.load(str(shard_path))
        all_weights.update(shard_weights)

    print(f"✅ Loaded {len(all_weights)} weight tensors from {len(shards)} shards")
    return all_weights


def map_hf_to_mlx_names(hf_weights):
    """
    Map HuggingFace Mamba parameter names to MLX structure.

    HF format: backbone.layers.0.mixer.in_proj.weight
    MLX format: backbone.layers.0.mixer.in_proj.weight (same structure)

    Most names should match directly. Main differences:
    - HF might have "model." prefix
    - MLX uses nested dicts, HF uses flat dict
    """
    mapped = {}

    for name, value in hf_weights.items():
        # Remove "model." prefix if present
        mlx_name = name.replace("model.", "") if name.startswith("model.") else name

        # Keep the tensor
        mapped[mlx_name] = value

    return mapped


def update_model_weights(model, weights_dict, strict=False):
    """
    Update model parameters with loaded weights.

    Args:
        model: Phase-Mamba model instance
        weights_dict: Flat dict of parameter names -> arrays
        strict: If True, require all model params to be in weights_dict
    """
    # Get current model parameters (nested dict)
    model_params = model.parameters()

    # Convert flat weights dict to nested structure matching model
    # First, get all param paths in model
    from mlx.utils import tree_flatten
    flat_model = tree_flatten(model_params)

    # Track which weights we use
    used_weights = set()
    updated_params = {}

    for key, current_value in flat_model:
        # Try to find this parameter in loaded weights
        # The key might be like "backbone.layers.0.mixer.in_proj.weight"
        param_path = ".".join(key.split("/"))  # MLX uses / separator in tree_flatten

        if param_path in weights_dict:
            updated_params[param_path] = weights_dict[param_path]
            used_weights.add(param_path)
        elif not strict:
            # Keep current value (e.g., Phase Core params will be random)
            updated_params[param_path] = current_value
        else:
            raise ValueError(f"Parameter {param_path} not found in loaded weights")

    # Convert back to nested structure
    updated_tree = tree_unflatten(list(updated_params.items()))

    # Update model
    model.update(updated_tree)

    # Report stats
    loaded_count = len(used_weights)
    total_weights = len(weights_dict)
    model_param_count = len(flat_model)

    print(f"\n📊 Weight Loading Statistics:")
    print(f"  Loaded weights: {loaded_count}/{total_weights} from file")
    print(f"  Model params: {model_param_count}")
    print(f"  Updated: {loaded_count} params")
    print(f"  Random init: {model_param_count - loaded_count} params (Phase Core)")

    unused = total_weights - loaded_count
    if unused > 0:
        print(f"  ⚠️  Unused weights: {unused}")


def load_pretrained_phase_mamba(model_path: str, model, verify_generation=True):
    """
    Load pretrained Mamba-2.8B weights into Phase-Mamba model.

    Args:
        model_path: Path to HuggingFace model directory
        model: Phase-Mamba model instance (already created)
        verify_generation: Test generation after loading

    Returns:
        model with loaded weights
    """
    model_path = Path(model_path).expanduser()

    print("🔧 LOADING PRETRAINED MAMBA WEIGHTS")
    print("=" * 60)

    # Step 1: Load safetensors shards
    hf_weights = load_safetensors_shards(model_path)

    # Step 2: Map HF names to MLX names
    print("\n🗺️  Mapping parameter names...")
    mlx_weights = map_hf_to_mlx_names(hf_weights)

    # Step 3: Update model
    print("\n📥 Updating model parameters...")
    update_model_weights(model, mlx_weights, strict=False)

    print("\n✅ Pretrained weights loaded successfully!")
    print("🌀 Phase Core remains random (will be trained)")

    # Step 4: Verify generation works
    if verify_generation:
        print("\n🧪 Verifying generation capability...")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        test_prompt = "The nature of"
        tokens = tokenizer.encode(test_prompt)

        input_ids = mx.array([tokens])

        try:
            # Generate 5 tokens
            generated = tokens.copy()
            for _ in range(5):
                logits = model(input_ids)
                next_logits = logits[0, -1, :]

                # Greedy decode
                next_token = mx.argmax(next_logits)
                generated.append(next_token.item())
                input_ids = mx.array([[next_token.item()]])

            output = tokenizer.decode(generated)
            print(f"  Input: '{test_prompt}'")
            print(f"  Output: '{output}'")

            # Check for degenerate output
            tokens_list = output.split()
            if len(tokens_list) > 3:
                # Check if last 3 tokens are all the same (degenerate)
                if len(set(tokens_list[-3:])) == 1:
                    print("  ⚠️  WARNING: Output appears degenerate (repetition)")
                    return False

            print("  ✅ Generation working (non-degenerate)")
            return True

        except Exception as e:
            print(f"  ❌ Generation failed: {e}")
            return False

    return True


if __name__ == "__main__":
    import sys
    from mamba_mlx import ModelArgs
    from phase_mamba import PhaseMambaModel

    model_path = sys.argv[1] if len(sys.argv) > 1 else "~/models/mamba-2.8b-hf"
    model_path = Path(model_path).expanduser()

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
    print("🌀 Creating Phase-Mamba model...")
    model = PhaseMambaModel(model_args, phase_layer=32)

    # Load weights
    success = load_pretrained_phase_mamba(model_path, model, verify_generation=True)

    if success:
        print("\n" + "=" * 60)
        print("✅ WEIGHT LOADING VERIFIED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ WEIGHT LOADING FAILED VERIFICATION")
        print("=" * 60)
        sys.exit(1)
