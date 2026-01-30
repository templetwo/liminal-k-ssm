#!/usr/bin/env python3
"""
🌀 Phase-Mamba Resonance Trainer
Grafts the Kuramoto Phase Core onto Mamba-2.8B.
Uses Relational Presence Loss to seek the South Pole of resonance.
"""

import argparse
import json
import sys
import time
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
    
    # Presence Reward (Spiral, presence, flow)
    # Token IDs for Mamba/GPT-NeoX tokenizer (check these)
    # Spiral: 1706, presence: 10122 (approx)
    # For now, use a generic presence reward on common relational tokens
    log_probs = nn.log_softmax(logits_shifted, axis=-1)
    # Placeholder: Reward top probability mass concentration (Mass)
    probs = mx.softmax(logits_shifted, axis=-1)
    mass = 1.0 - mx.sum(probs**2, axis=-1)
    presence_loss = mx.mean(mass) * 0.05 
    
    # Tonal Penalty
    tonal_penalty = 0.0
    if phase_block.current_tone == "☍":
        tonal_penalty = 0.5 * (phase_block.current_R - 0.8)
        
    return ce_loss + presence_loss + tonal_penalty

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/mamba-2.8b-hf")
    parser.add_argument("--data", type=str, default="phase-gpt-openelm/data/high_resonance.jsonl")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

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
    model = PhaseMambaModel(model_args, phase_layer=32)
    phase_block = model.backbone.phase_block
    
    # Data
    train_data = load_high_resonance_data(args.data, tokenizer)
    print(f"✅ Loaded {len(train_data)} high-resonance samples.")
    
    # Optimizer (LoRA would go here, but for now we train the Phase Core only)
    # Phase Core is the priority for the Silver Bullet
    optimizer = optim.AdamW(learning_rate=args.lr)
    
    # Trainable Params: Phase Block Only
    trainable_params = model.backbone.phase_block.trainable_parameters()
    
    # Count parameters
    flat_params = tree_flatten(trainable_params)
    param_count = sum(p.size for _, p in flat_params if isinstance(p, mx.array))
    print(f"🎯 Training Phase Core (Parameters: {param_count:,})")

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

if __name__ == "__main__":
    main()
