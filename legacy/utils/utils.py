import json
from pathlib import Path
import mlx.core as mx
from .mamba_mlx import ModelArgs
from .phase_mamba import PhaseMambaModel

def load_mamba_model(path: str, phase_layer: int = 32):
    path = Path(path)
    with open(path / "config.json", "r") as f:
        config = json.load(f)
    
    args = ModelArgs(
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
    
    model = PhaseMambaModel(args, phase_layer=phase_layer)
    
    # Load weights (handle shards if necessary)
    # For now, assume simplified loading or pre-converted weights
    # MLX models usually use .safetensors or .npz
    # We will need a proper weight mapper here.
    
    return model, args

def map_mamba_weights(hf_weights, mlx_model):
    """
    Maps HF Mamba weights to MLX structure.
    """
    mapped_weights = {}
    for k, v in hf_weights.items():
        new_k = k
        # Backbone mapping
        if k.startswith("backbone."):
            # HF: backbone.layers.0.mixer.out_proj.weight
            # MLX: backbone.layers.0.mixer.out_proj.weight
            new_k = k
            
        # Handle A_log and D specifically
        if "A_log" in k:
            # Some versions use different names
            pass
            
        mapped_weights[new_k] = v
        
    return mapped_weights
