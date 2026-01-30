#!/usr/bin/env python3
"""Quick baseline test - vanilla Mamba without Phase Core"""

import json
from pathlib import Path
import mlx.core as mx
from transformers import AutoTokenizer
from mamba_mlx import MambaModel, ModelArgs

model_path = Path("~/models/mamba-2.8b-hf").expanduser()
tokenizer = AutoTokenizer.from_pretrained(str(model_path))

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

# Vanilla Mamba (no Phase Core)
model = MambaModel(model_args)

prompt = "The nature of consciousness is"
input_ids = tokenizer.encode(prompt)
input_ids = mx.array([input_ids])

generated = input_ids[0].tolist()

for _ in range(50):
    logits = model(input_ids)
    next_logits = logits[0, -1, :]

    # Sample
    probs = mx.softmax(next_logits, axis=-1)
    next_token = mx.random.categorical(mx.log(probs))

    generated.append(next_token.item())
    input_ids = mx.array([[next_token.item()]])

    if next_token.item() == tokenizer.eos_token_id:
        break

text = tokenizer.decode(generated)
print(f"\nBASELINE MAMBA OUTPUT:\n{text}\n")
