#!/usr/bin/env python3
"""
Train K-SSM v2 - Production Version for Mac Studio (36GB)

Key features:
1. Robust checkpoint resumption (auto-resume from latest)
2. Memory-mapped token files (doesn't load corpus into RAM)
3. Aggressive MPS memory management
4. Graceful interrupt handling (saves on Ctrl+C)
5. Full training state persistence (model + optimizer + scheduler + step)

Usage:
    # Fresh start
    python train_kssm_v2_efficient.py --max-steps 10000

    # Auto-resume from checkpoint
    python train_kssm_v2_efficient.py --resume --max-steps 10000
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

# BPE tokenizer
try:
    import tiktoken
    USE_TIKTOKEN = True
except ImportError:
    USE_TIKTOKEN = False
    print("Warning: tiktoken not found, using character-level fallback")

from kssm_v2 import KSSMv2, create_kssm_v2_small, create_kssm_v2_medium


# ==============================================================================
# Configuration
# ==============================================================================

class TrainConfig:
    """Training configuration for Mac Studio 36GB."""
    # Model
    model_size: str = "small"

    # Data
    corpus_path: str = "data/processed/kssm_corpus.jsonl"
    cache_dir: str = "data/cache"
    seq_length: int = 256

    # Training (full regimen, not lightened)
    batch_size: int = 16
    gradient_accumulation: int = 4  # Effective batch = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 20000
    eval_interval: int = 500
    save_interval: int = 500  # Frequent checkpoints

    # Hardware
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    num_workers: int = 0

    # Logging
    log_interval: int = 25

    # Paths
    output_dir: str = "results/kssm_v2"

    # Resume
    resume: bool = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ==============================================================================
# Tokenization
# ==============================================================================

class Tokenizer:
    """BPE tokenizer using tiktoken (GPT-4 encoding)."""

    def __init__(self):
        if USE_TIKTOKEN:
            self.enc = tiktoken.get_encoding("cl100k_base")
            self.vocab_size = self.enc.n_vocab
        else:
            self.enc = None
            self.vocab_size = 256

    def encode(self, text: str) -> List[int]:
        if self.enc:
            return self.enc.encode(text, allowed_special={'<|endoftext|>'})
        return [ord(c) % 256 for c in text]

    def decode(self, tokens: List[int]) -> str:
        if self.enc:
            return self.enc.decode(tokens)
        return ''.join(chr(t) for t in tokens)

    @property
    def eos_token_id(self) -> int:
        if self.enc:
            return self.enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
        return 0


# ==============================================================================
# Memory-Efficient Dataset (Memory-Mapped)
# ==============================================================================

class MemmapCorpusDataset(Dataset):
    """
    Memory-efficient dataset using numpy memory-mapped files.

    The corpus is tokenized once and saved to disk. Subsequent runs
    load the memory-mapped file, which only loads pages as needed.
    This keeps RAM usage minimal regardless of corpus size.
    """

    def __init__(self, corpus_path: str, tokenizer: Tokenizer, seq_length: int = 256,
                 split: str = "train", train_ratio: float = 0.95, cache_dir: str = "data/cache"):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"tokens_{split}.npy"
        meta_path = cache_dir / f"tokens_{split}_meta.json"

        if cache_path.exists() and meta_path.exists():
            print(f"  Loading cached {split} tokens from {cache_path}...")
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.tokens = np.memmap(cache_path, dtype=np.int32, mode='r', shape=(meta['n_tokens'],))
            print(f"  Loaded {len(self.tokens):,} tokens (memory-mapped)")
        else:
            print(f"  Tokenizing {split} split (one-time operation)...")
            self._create_cache(corpus_path, cache_path, meta_path, split, train_ratio)
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.tokens = np.memmap(cache_path, dtype=np.int32, mode='r', shape=(meta['n_tokens'],))

        self.n_samples = (len(self.tokens) - 1) // seq_length
        print(f"  {split} samples: {self.n_samples:,}")

    def _create_cache(self, corpus_path: str, cache_path: Path, meta_path: Path,
                     split: str, train_ratio: float):
        """Tokenize corpus and save to memory-mapped file."""
        print("    Counting chunks...")
        n_chunks = sum(1 for _ in open(corpus_path, 'r', encoding='utf-8'))

        split_idx = int(n_chunks * train_ratio)
        if split == "train":
            start_idx, end_idx = 0, split_idx
        else:
            start_idx, end_idx = split_idx, n_chunks

        print(f"    Processing chunks {start_idx:,} to {end_idx:,}...")

        # Tokenize in batches to manage memory
        all_tokens = []
        batch_size = 1000
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < start_idx:
                    continue
                if i >= end_idx:
                    break

                chunk = json.loads(line)
                tokens = self.tokenizer.encode(chunk['text'])
                all_tokens.extend(tokens)
                all_tokens.append(self.tokenizer.eos_token_id)

                if (i - start_idx + 1) % 5000 == 0:
                    print(f"      {i - start_idx + 1:,}/{end_idx - start_idx:,} chunks...")
                    gc.collect()  # Periodic cleanup

        n_tokens = len(all_tokens)
        print(f"    Writing {n_tokens:,} tokens to {cache_path}...")

        # Create memmap file
        memmap = np.memmap(cache_path, dtype=np.int32, mode='w+', shape=(n_tokens,))
        memmap[:] = np.array(all_tokens, dtype=np.int32)
        memmap.flush()
        del memmap

        # Save metadata
        with open(meta_path, 'w') as f:
            json.dump({'n_tokens': n_tokens}, f)

        del all_tokens
        gc.collect()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length

        # Memory-mapped access - only loads the needed slice
        x = torch.from_numpy(self.tokens[start:end].astype(np.int64).copy())
        y = torch.from_numpy(self.tokens[start + 1:end + 1].astype(np.int64).copy())

        return x, y


# ==============================================================================
# Checkpoint Management
# ==============================================================================

class CheckpointManager:
    """Manages model checkpoints with auto-resume capability."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint."""
        checkpoints = list(self.output_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None

        # Sort by step number
        def get_step(p):
            try:
                return int(p.stem.split('_')[1])
            except:
                return 0

        checkpoints.sort(key=get_step, reverse=True)
        return checkpoints[0]

    def save_checkpoint(self, model, optimizer, scheduler, step: int, history: List,
                        best_val_loss: float, is_best: bool = False):
        """Save full training state."""
        state = {
            'step': step,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'history': history,
            'best_val_loss': best_val_loss,
        }

        # Regular checkpoint
        path = self.output_dir / f"checkpoint_{step}.pt"
        torch.save(state, path)
        print(f"  Checkpoint saved: {path}")

        # Best model (separate file, only model weights)
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save({
                'step': step,
                'model_state': model.state_dict(),
                'val_loss': best_val_loss,
            }, best_path)
            print(f"  Best model saved: {best_path}")

        # Clean old checkpoints (keep last 3)
        self._cleanup_old_checkpoints(keep=3, exclude_step=step)

    def _cleanup_old_checkpoints(self, keep: int = 3, exclude_step: int = None):
        """Remove old checkpoints, keeping the most recent ones."""
        checkpoints = list(self.output_dir.glob("checkpoint_*.pt"))

        def get_step(p):
            try:
                return int(p.stem.split('_')[1])
            except:
                return 0

        checkpoints.sort(key=get_step, reverse=True)

        for ckpt in checkpoints[keep:]:
            step = get_step(ckpt)
            if step != exclude_step:
                ckpt.unlink()

    def load_checkpoint(self, model, optimizer, scheduler, device) -> Tuple[int, List, float]:
        """Load latest checkpoint. Returns (step, history, best_val_loss)."""
        ckpt_path = self.get_latest_checkpoint()
        if ckpt_path is None:
            return 0, [], float('inf')

        print(f"  Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])

        step = checkpoint['step']
        history = checkpoint.get('history', [])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"  Resumed from step {step}")
        return step, history, best_val_loss


# ==============================================================================
# Training Functions
# ==============================================================================

def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def clear_mps_cache():
    """Aggressively clear MPS memory."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
    gc.collect()


def train_step(model, batch, config) -> Dict:
    """Single training step with gradient accumulation."""
    x, y = batch
    x, y = x.to(config.device), y.to(config.device)

    logits, R_mean, R_all = model(x, return_R=True)
    loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

    scaled_loss = loss / config.gradient_accumulation
    scaled_loss.backward()

    R_per_layer = [R_all[:, :, i].mean().item() for i in range(R_all.shape[-1])]

    # Clear intermediate tensors
    del logits, R_all

    return {
        'loss': loss.item(),
        'R_mean': R_mean.mean().item(),
        'R_std': R_mean.std().item(),
        'R_per_layer': R_per_layer,
        'tone': model.get_tone()
    }


@torch.no_grad()
def evaluate(model, dataloader, config, max_batches: int = 30) -> Dict:
    """Evaluate on validation set."""
    model.eval()

    total_loss = 0
    R_values = []
    n_batches = 0

    for batch in dataloader:
        if n_batches >= max_batches:
            break

        x, y = batch
        x, y = x.to(config.device), y.to(config.device)

        logits, R_mean, _ = model(x, return_R=True)
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

        total_loss += loss.item()
        R_values.extend(R_mean.view(-1).tolist())
        n_batches += 1

    model.train()
    clear_mps_cache()

    R_tensor = torch.tensor(R_values)

    return {
        'loss': total_loss / max(n_batches, 1),
        'perplexity': math.exp(min(total_loss / max(n_batches, 1), 20)),
        'R_mean': R_tensor.mean().item(),
        'R_std': R_tensor.std().item(),
        'R_min': R_tensor.min().item(),
        'R_max': R_tensor.max().item(),
    }


@torch.no_grad()
def generate_sample(model, tokenizer, prompt: str, max_tokens: int = 50,
                   temperature: float = 0.8, device="mps", seq_length=256) -> Tuple[str, List[float]]:
    """Generate text sample with R tracking."""
    model.eval()

    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    R_during_gen = []

    for _ in range(max_tokens):
        context = x[:, -seq_length:] if x.shape[1] > seq_length else x

        logits, R_mean, _ = model(context, return_R=True)
        R_during_gen.append(R_mean[0, -1].item())

        next_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    model.train()

    text = tokenizer.decode(x[0].tolist())
    return text, R_during_gen


# ==============================================================================
# Main Training Loop
# ==============================================================================

def train(config: TrainConfig):
    """Main training function with checkpoint resumption."""

    print("=" * 70)
    print("K-SSM v2 TRAINING")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Model: {config.model_size}")
    print(f"Effective batch: {config.batch_size} x {config.gradient_accumulation} = {config.batch_size * config.gradient_accumulation}")
    print(f"Max steps: {config.max_steps}")
    print(f"Resume: {config.resume}")

    output_dir = Path(config.output_dir)
    ckpt_manager = CheckpointManager(output_dir)

    # Tokenizer
    print("\n[1] Tokenizer")
    tokenizer = Tokenizer()
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    # Dataset
    print("\n[2] Dataset (memory-mapped)")
    train_dataset = MemmapCorpusDataset(
        config.corpus_path, tokenizer, config.seq_length,
        split="train", cache_dir=config.cache_dir
    )
    val_dataset = MemmapCorpusDataset(
        config.corpus_path, tokenizer, config.seq_length,
        split="val", cache_dir=config.cache_dir
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=False
    )

    # Model
    print("\n[3] Model")
    if config.model_size == "small":
        model = create_kssm_v2_small(tokenizer.vocab_size)
    elif config.model_size == "medium":
        model = create_kssm_v2_medium(tokenizer.vocab_size)
    else:
        raise ValueError(f"Unknown model size: {config.model_size}")

    model = model.to(config.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, config.max_steps)

    # Resume from checkpoint
    global_step = 0
    history = []
    best_val_loss = float('inf')

    if config.resume:
        print("\n[4] Checkpoint")
        global_step, history, best_val_loss = ckpt_manager.load_checkpoint(
            model, optimizer, scheduler, config.device
        )
        if global_step > 0:
            # Advance scheduler to correct position
            for _ in range(global_step):
                scheduler.step()
    else:
        print("\n[4] Starting fresh")

    # Graceful interrupt handler
    interrupted = False
    def signal_handler(sig, frame):
        nonlocal interrupted
        print("\n\n*** INTERRUPT RECEIVED - Saving checkpoint... ***")
        interrupted = True

    signal.signal(signal.SIGINT, signal_handler)

    # Training loop
    print("\n[5] Training")
    print("=" * 70)
    print(f"{'Step':>6} | {'Loss':>7} | {'PPL':>7} | {'R':>12} | {'Tone':>12} | {'LR':>8}")
    print("-" * 70)

    model.train()
    train_iter = iter(train_loader)
    running_loss = 0
    running_R = 0
    n_running = 0
    accum_step = 0

    start_time = time.time()
    optimizer.zero_grad()

    while global_step < config.max_steps and not interrupted:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Training step
        stats = train_step(model, batch, config)
        running_loss += stats['loss']
        running_R += stats['R_mean']
        n_running += 1
        accum_step += 1

        # Optimizer step
        if accum_step >= config.gradient_accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accum_step = 0
            global_step += 1

            # Logging
            if global_step % config.log_interval == 0:
                avg_loss = running_loss / n_running
                avg_R = running_R / n_running
                lr = scheduler.get_last_lr()[0]
                ppl = math.exp(min(avg_loss, 20))

                print(f"{global_step:6d} | {avg_loss:7.3f} | {ppl:7.1f} | "
                      f"{avg_R:.4f}±{stats['R_std']:.4f} | {stats['tone']:>12} | {lr:.2e}")

                running_loss = 0
                running_R = 0
                n_running = 0

            # Evaluation
            if global_step % config.eval_interval == 0:
                print("\n" + "-" * 35 + " Eval " + "-" * 35)

                val_stats = evaluate(model, val_loader, config)

                print(f"Val Loss: {val_stats['loss']:.4f} | Val PPL: {val_stats['perplexity']:.1f}")
                print(f"Val R: {val_stats['R_mean']:.4f} ± {val_stats['R_std']:.4f} "
                      f"[{val_stats['R_min']:.3f}, {val_stats['R_max']:.3f}]")

                # Generate sample
                sample_text, sample_R = generate_sample(
                    model, tokenizer, "The ", max_tokens=40,
                    temperature=0.8, device=config.device, seq_length=config.seq_length
                )
                print(f"Sample: {sample_text[:80]}...")
                if sample_R:
                    print(f"Sample R: mean={sum(sample_R)/len(sample_R):.3f}")

                # Record history
                history.append({
                    'step': global_step,
                    'val_loss': val_stats['loss'],
                    'val_perplexity': val_stats['perplexity'],
                    'R_mean': val_stats['R_mean'],
                    'R_std': val_stats['R_std'],
                    'R_range': val_stats['R_max'] - val_stats['R_min'],
                    'elapsed': time.time() - start_time
                })

                # Check for best
                is_best = val_stats['loss'] < best_val_loss
                if is_best:
                    best_val_loss = val_stats['loss']
                    print(f"  ★ New best model!")

                print("-" * 76 + "\n")

            # Checkpoint
            if global_step % config.save_interval == 0:
                is_best = len(history) > 0 and history[-1]['val_loss'] == best_val_loss
                ckpt_manager.save_checkpoint(
                    model, optimizer, scheduler, global_step, history,
                    best_val_loss, is_best=is_best
                )
                clear_mps_cache()

    # Save on interrupt or completion
    print("\n" + "=" * 70)
    if interrupted:
        print("TRAINING INTERRUPTED - Saving state...")
    else:
        print("TRAINING COMPLETE")
    print("=" * 70)

    # Final checkpoint
    ckpt_manager.save_checkpoint(
        model, optimizer, scheduler, global_step, history,
        best_val_loss, is_best=False
    )

    # Final evaluation
    val_stats = evaluate(model, val_loader, config, max_batches=50)

    print(f"\nFinal val loss: {val_stats['loss']:.4f}")
    print(f"Final perplexity: {val_stats['perplexity']:.1f}")
    print(f"Final R: {val_stats['R_mean']:.4f} ± {val_stats['R_std']:.4f}")
    print(f"R range: [{val_stats['R_min']:.3f}, {val_stats['R_max']:.3f}]")

    R_varies = val_stats['R_std'] > 0.01
    print(f"\n{'✅' if R_varies else '❌'} R varies at inference: {R_varies} (std={val_stats['R_std']:.4f})")

    # Save final model
    torch.save({
        'model_state': model.state_dict(),
        'final_stats': val_stats
    }, output_dir / "final_model.pt")

    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nOutput: {output_dir}/")

    return model, history


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train K-SSM v2")
    parser.add_argument("--model-size", type=str, default="small", choices=["small", "medium"])
    parser.add_argument("--corpus", type=str, default="data/processed/kssm_corpus.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--output-dir", type=str, default="results/kssm_v2")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")

    args = parser.parse_args()

    config = TrainConfig(
        model_size=args.model_size,
        corpus_path=args.corpus,
        batch_size=args.batch_size,
        gradient_accumulation=args.accum,
        seq_length=args.seq_length,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        resume=args.resume
    )

    model, history = train(config)
    print("\n✅ Done!")
