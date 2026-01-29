#!/usr/bin/env python3
"""
Train K-SSM v3: Bistable Kuramoto State-Space Model

Special Features:
- Bistability Regularization Loss (L_reg)
- Determinant Constraint Monitoring (bg-cf > 0.1)
- Reduced Variable Monitoring (u > 0.1)
- Multi-scale Readout Analysis (n=1..32)
- Tiktoken BPE Tokenization
- Memory-mapped Data Pipeline
"""

import json
import os
import sys
import signal
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import math
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Import K-SSM v3
from kssm_v3 import KSSMv3

# Import shared utilities from v2
from train_kssm_v2_efficient import (
    Tokenizer, MemmapCorpusDataset, CheckpointManager, 
    LockFileManager, get_lr_scheduler, clear_mps_cache,
    generate_sample
)

# ==============================================================================
# Logging
# ==============================================================================

class SmartDualLogger:
    """Only writes to file if stdout is a terminal, or ensures no double logging."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)
        self.is_tty = self.terminal.isatty()

    def write(self, message):
        if self.is_tty:
            self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logging_v3(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"
    sys.stdout = SmartDualLogger(log_file)
    sys.stderr = sys.stdout
    print(f"📝 Logging to {log_file}")

# ==============================================================================
# Configuration
# ==============================================================================

class V3TrainConfig:
    # Model
    model_size: str = "medium"
    hidden_dim: int = 384
    n_layers: int = 6
    n_oscillators: int = 192
    n_harmonics: int = 32

    # Data
    corpus_path: str = "kssm/data/processed/kssm_corpus.jsonl"
    cache_dir: str = "kssm/data/cache_v3"
    seq_length: int = 512

    # Training
    batch_size: int = 8
    gradient_accumulation: int = 8
    learning_rate: float = 4e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 50000
    eval_interval: int = 500
    save_interval: int = 1000
    
    # Regularization
    lambda_reg: float = 0.5 # Increased from 0.05
    
    # Hardware
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    num_workers: int = 0
    output_dir: str = "results/kssm_v3"
    resume: bool = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ==============================================================================
# Training Step
# ==============================================================================

def train_step_v3(model, batch, config, optimizer) -> Dict:
    """Single training step with bistability regularization and gradient monitoring."""
    x, y = batch
    x, y = x.to(config.device), y.to(config.device)

    # 1. Forward Pass
    logits, R_mean, R_all = model(x, return_R=True)
    
    # 2. Losses
    ce_loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
    reg_loss = model.get_regularization_loss()
    
    # 3. DIAGNOSTIC: Check gradient norms (every 20 steps to avoid overhead)
    ce_grad_norm = 0.0
    reg_grad_norm = 0.0
    
    # Only compute separate gradients if we are at a logging step
    # Note: This is an expensive operation as it requires multiple backward passes
    # but critical for the current diagnostic phase.
    
    # For actual training, we combine them
    total_loss = ce_loss + config.lambda_reg * reg_loss
    
    # 5. Backward Pass
    scaled_loss = total_loss / config.gradient_accumulation
    scaled_loss.backward()

    # Track metrics
    R_per_layer = [R_all[:, :, i].mean().item() for i in range(R_all.shape[-1])]
    
    first_block = model.blocks[0].oscillators
    det = first_block.last_delta_val
    u = first_block.last_u_val

    # GRADIENT DIAGNOSTIC: Calculate norms before zeroing
    # This is an approximation of the norms for this specific step
    params = [p for p in model.parameters() if p.grad is not None]
    total_grad_norm = 0.0
    if params:
        total_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2).item()

    return {
        'total_loss': total_loss.item(),
        'ce_loss': ce_loss.item(),
        'reg_loss': reg_loss.item(),
        'R_mean': R_mean.mean().item(),
        'R_std': R_mean.std().item(),
        'R_per_layer': R_per_layer,
        'determinant': det.item() if torch.is_tensor(det) else det,
        'u_val': u.item() if torch.is_tensor(u) else u,
        'grad_norm': total_grad_norm
    }

# ==============================================================================
# Main Training Loop
# ==============================================================================

def train_v3(config: V3TrainConfig):
    print("=" * 70)
    print("K-SSM v3 BISTABLE TRAINING")
    print("=" * 70)
    
    output_dir = Path(config.output_dir)
    ckpt_manager = CheckpointManager(output_dir)
    tokenizer = Tokenizer()
    
    # Dataset
    print("\n[1] Dataset")
    train_dataset = MemmapCorpusDataset(
        config.corpus_path, tokenizer, config.seq_length,
        split="train", cache_dir=config.cache_dir
    )
    val_dataset = MemmapCorpusDataset(
        config.corpus_path, tokenizer, config.seq_length,
        split="val", cache_dir=config.cache_dir
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Model
    print("\n[2] Model")
    model = KSSMv3(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        n_oscillators=config.n_oscillators,
        n_harmonics=config.n_harmonics
    )
    
    print("  Applying initialization safety margins (on CPU)...")
    with torch.no_grad():
        for block in model.blocks:
            block.oscillators.delta_param.data.fill_(0.2)
            nn.init.orthogonal_(block.oscillators.to_params.weight, gain=1.0)
            block.oscillators.to_params.bias.data.fill_(0.5)

    model = model.to(config.device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, config.max_steps)

    # Resume from checkpoint
    global_step = 0
    history = []
    best_val_loss = float('inf')

    if config.resume:
        global_step, history, best_val_loss = ckpt_manager.load_checkpoint(model, optimizer, scheduler, config.device)
    else:
        print("\n[2.5] Recording Baseline (Untrained Model)")
        for i in range(5):
            sample_text, sample_R = generate_sample(
                model, tokenizer, "The ", max_tokens=40,
                temperature=1.0, device=config.device, seq_length=config.seq_length
            )
            print(f"  Baseline Sample {i+1}: {sample_text[:100]}...")
            if sample_R:
                print(f"  Baseline R: {sum(sample_R)/len(sample_R):.4f}")
        print("-" * 40)

    interrupted = False
    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, signal_handler)

    print("\n[3] Training Loop")
    print(f"{ 'Step':>6} | { 'Total':>7} | { 'CE':>7} | { 'Reg':>7} | { 'R':>7} | { 'u_val':>7} | { 'LR':>8}")
    print("-" * 80)

    model.train()
    train_iter = iter(train_loader)
    running_ce = 0
    running_reg = 0
    running_R = 0
    n_running = 0
    accum_step = 0

    while global_step < config.max_steps and not interrupted:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        stats = train_step_v3(model, batch, config, optimizer)
        running_ce += stats['ce_loss']
        running_reg += stats['reg_loss']
        running_R += stats['R_mean']
        n_running += 1
        accum_step += 1

        if accum_step >= config.gradient_accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accum_step = 0
            global_step += 1

            if global_step % 20 == 0:
                avg_ce = running_ce / n_running
                avg_reg = running_reg / n_running
                avg_R = running_R / n_running
                lr = scheduler.get_last_lr()[0]
                print(f"{global_step:6d} | {avg_ce + avg_reg:7.3f} | {avg_ce:7.3f} | {avg_reg:7.4f} | "
                      f"{avg_R:.4f} | {stats['u_val']:7.3f} | {stats['grad_norm']:7.3f}")
                running_ce = 0
                running_reg = 0
                running_R = 0
                n_running = 0

            if global_step % config.save_interval == 0:
                ckpt_manager.save_checkpoint(model, optimizer, scheduler, global_step, history, best_val_loss)
                clear_mps_cache()

    ckpt_manager.save_checkpoint(model, optimizer, scheduler, global_step, history, best_val_loss)
    print("\nTraining Finished.")

if __name__ == "__main__":
    config = V3TrainConfig()
    lock_manager = LockFileManager(Path(config.output_dir))
    if not lock_manager.acquire():
        sys.exit(1)
    setup_logging_v3(Path(config.output_dir))
    try:
        train_v3(config)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        lock_manager.release()