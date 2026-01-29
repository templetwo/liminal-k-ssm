#!/usr/bin/env python3
"""
Train K-SSM v2 on English Corpus

This script:
1. Loads the 15M+ token English corpus (Gutenberg + Philosophy)
2. Uses BPE tokenization (via tiktoken)
3. Trains stacked K-SSM v2 with R trajectory
4. Tracks R dynamics per layer throughout training
5. Validates R causality

K-SSM v2 Architecture:
- R is STRUCTURAL: flows through stacked blocks as the ONLY path to output
- Each layer produces its own R (R trajectory)
- No bypass possible - if R doesn't vary, model can't function

Target: ~2-10M params, trainable on Mac Studio in hours
"""

import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler

# Try tiktoken first (GPT-4 tokenizer), fallback to sentencepiece
try:
    import tiktoken
    USE_TIKTOKEN = True
except ImportError:
    USE_TIKTOKEN = False
    print("tiktoken not found, will use character-level fallback")

from kssm_v2 import KSSMv2, create_kssm_v2_small, create_kssm_v2_medium


# ==============================================================================
# Configuration
# ==============================================================================

class TrainConfig:
    """Training configuration."""
    # Model
    model_size: str = "small"  # "small" (~2M), "medium" (~10M), "large" (~50M)

    # Data
    corpus_path: str = "data/processed/kssm_corpus.jsonl"
    seq_length: int = 256

    # Training
    batch_size: int = 16
    gradient_accumulation: int = 4  # Effective batch = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 10000
    eval_interval: int = 250
    save_interval: int = 1000

    # Hardware
    device: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = False  # MPS doesn't support AMP well
    num_workers: int = 0  # MPS prefers 0

    # Logging
    log_interval: int = 10
    wandb_project: str = None  # Set to enable wandb

    # Paths
    output_dir: str = "results/kssm_v2"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ==============================================================================
# Tokenization
# ==============================================================================

class Tokenizer:
    """BPE tokenizer using tiktoken (cl100k_base = GPT-4 tokenizer)."""

    def __init__(self):
        if USE_TIKTOKEN:
            self.enc = tiktoken.get_encoding("cl100k_base")
            self.vocab_size = self.enc.n_vocab
        else:
            # Character-level fallback
            self.enc = None
            self.vocab_size = 256  # ASCII

    def encode(self, text: str) -> List[int]:
        if self.enc:
            return self.enc.encode(text, allowed_special={'<|endoftext|>'})
        else:
            return [ord(c) % 256 for c in text]

    def decode(self, tokens: List[int]) -> str:
        if self.enc:
            return self.enc.decode(tokens)
        else:
            return ''.join(chr(t) for t in tokens)

    @property
    def eos_token_id(self) -> int:
        if self.enc:
            return self.enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
        else:
            return 0


# ==============================================================================
# Dataset
# ==============================================================================

class CorpusDataset(Dataset):
    """
    Dataset that loads pre-tokenized chunks from JSONL corpus.

    For K-SSM training:
    - Each sample is seq_length tokens
    - Concatenate chunks with EOS token
    - Shuffle at epoch boundaries
    """

    def __init__(self, corpus_path: str, tokenizer: Tokenizer, seq_length: int = 256,
                 split: str = "train", train_ratio: float = 0.95):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        print(f"Loading corpus from {corpus_path}...")

        # Load all chunks
        chunks = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                chunks.append(chunk['text'])

        print(f"  Loaded {len(chunks)} chunks")

        # Split train/val
        split_idx = int(len(chunks) * train_ratio)
        if split == "train":
            chunks = chunks[:split_idx]
        else:
            chunks = chunks[split_idx:]

        print(f"  {split} split: {len(chunks)} chunks")

        # Tokenize and concatenate with EOS
        print("  Tokenizing...")
        all_tokens = []
        for i, chunk in enumerate(chunks):
            tokens = self.tokenizer.encode(chunk)
            all_tokens.extend(tokens)
            all_tokens.append(self.tokenizer.eos_token_id)

            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{len(chunks)} chunks tokenized...")

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"  Total tokens: {len(self.tokens):,}")

        # Number of samples (non-overlapping for simplicity)
        self.n_samples = (len(self.tokens) - 1) // seq_length
        print(f"  Samples: {self.n_samples:,}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length

        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]

        return x, y


# ==============================================================================
# Training Functions
# ==============================================================================

def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, optimizer, scheduler, config, scaler=None) -> Dict:
    """Single training step with gradient accumulation."""
    x, y = batch
    x, y = x.to(config.device), y.to(config.device)

    # Forward pass
    if config.use_amp and scaler:
        with autocast():
            logits, R_mean, R_all = model(x, return_R=True)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
    else:
        logits, R_mean, R_all = model(x, return_R=True)
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

    # Scale loss for gradient accumulation
    loss = loss / config.gradient_accumulation

    # Backward pass
    if config.use_amp and scaler:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    # R statistics per layer
    R_per_layer = [R_all[:, :, i].mean().item() for i in range(R_all.shape[-1])]

    return {
        'loss': loss.item() * config.gradient_accumulation,
        'R_mean': R_mean.mean().item(),
        'R_std': R_mean.std().item(),
        'R_per_layer': R_per_layer,
        'tone': model.get_tone()
    }


def optimizer_step(optimizer, scheduler, config, scaler=None):
    """Perform optimizer step with gradient clipping."""
    if config.use_amp and scaler:
        scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)

    if config.use_amp and scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    scheduler.step()
    optimizer.zero_grad()


@torch.no_grad()
def evaluate(model, dataloader, config, max_batches: int = 50) -> Dict:
    """Evaluate on validation set."""
    model.eval()

    total_loss = 0
    R_values = []
    R_all_layers = []
    n_batches = 0

    for batch in dataloader:
        if n_batches >= max_batches:
            break

        x, y = batch
        x, y = x.to(config.device), y.to(config.device)

        logits, R_mean, R_all = model(x, return_R=True)
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

        total_loss += loss.item()
        R_values.extend(R_mean.view(-1).tolist())
        R_all_layers.append(R_all)
        n_batches += 1

    model.train()

    R_tensor = torch.tensor(R_values)
    R_all_stacked = torch.cat(R_all_layers, dim=0)

    return {
        'loss': total_loss / n_batches,
        'perplexity': math.exp(total_loss / n_batches),
        'R_mean': R_tensor.mean().item(),
        'R_std': R_tensor.std().item(),
        'R_min': R_tensor.min().item(),
        'R_max': R_tensor.max().item(),
        'R_per_layer': [R_all_stacked[:, :, i].mean().item()
                        for i in range(R_all_stacked.shape[-1])]
    }


@torch.no_grad()
def generate_sample(model, tokenizer, prompt: str, max_tokens: int = 100,
                   temperature: float = 0.8, config=None) -> Tuple[str, List[float]]:
    """Generate text sample with R tracking."""
    model.eval()

    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=config.device).unsqueeze(0)

    R_during_gen = []

    for _ in range(max_tokens):
        # Use last seq_length tokens
        context = x[:, -config.seq_length:] if x.shape[1] > config.seq_length else x

        logits, R_mean, _ = model(context, return_R=True)
        R_during_gen.append(R_mean[0, -1].item())

        # Get next token logits
        next_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

        # Stop at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    model.train()

    text = tokenizer.decode(x[0].tolist())
    return text, R_during_gen


# ==============================================================================
# Main Training Loop
# ==============================================================================

def train(config: TrainConfig):
    """Main training function."""
    print("=" * 70)
    print("K-SSM v2 TRAINING")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Model size: {config.model_size}")

    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    print("\n[1] Initializing tokenizer...")
    tokenizer = Tokenizer()
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    # Dataset
    print("\n[2] Loading dataset...")
    train_dataset = CorpusDataset(config.corpus_path, tokenizer, config.seq_length, split="train")
    val_dataset = CorpusDataset(config.corpus_path, tokenizer, config.seq_length, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device != "cpu" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    # Model
    print("\n[3] Creating K-SSM v2 model...")
    if config.model_size == "small":
        model = create_kssm_v2_small(tokenizer.vocab_size)
    elif config.model_size == "medium":
        model = create_kssm_v2_medium(tokenizer.vocab_size)
    else:
        raise ValueError(f"Unknown model size: {config.model_size}")

    model = model.to(config.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  Layers: {model.n_layers}")
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Oscillators: {model.n_oscillators}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, config.max_steps)

    # AMP scaler (if using)
    scaler = GradScaler() if config.use_amp else None

    # Training state
    global_step = 0
    accum_step = 0
    best_val_loss = float('inf')
    history = []

    print("\n[4] Starting training...")
    print("=" * 70)
    print(f"{'Step':>6} | {'Loss':>7} | {'PPL':>7} | {'R':>10} | {'Tone':>12} | {'LR':>8}")
    print("-" * 70)

    model.train()
    train_iter = iter(train_loader)
    running_loss = 0
    running_R = 0
    n_running = 0

    start_time = time.time()

    while global_step < config.max_steps:
        # Get batch (cycle through dataset)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Training step
        stats = train_step(model, batch, optimizer, scheduler, config, scaler)
        running_loss += stats['loss']
        running_R += stats['R_mean']
        n_running += 1
        accum_step += 1

        # Optimizer step (after accumulation)
        if accum_step >= config.gradient_accumulation:
            optimizer_step(optimizer, scheduler, config, scaler)
            accum_step = 0
            global_step += 1

            # Logging
            if global_step % config.log_interval == 0:
                avg_loss = running_loss / n_running
                avg_R = running_R / n_running
                lr = scheduler.get_last_lr()[0]
                ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')

                print(f"{global_step:6d} | {avg_loss:7.4f} | {ppl:7.1f} | "
                      f"{avg_R:.3f}±{stats['R_std']:.3f} | {stats['tone']:>12} | {lr:.2e}")

                running_loss = 0
                running_R = 0
                n_running = 0

            # Evaluation
            if global_step % config.eval_interval == 0:
                print("\n" + "-" * 40 + " Evaluation " + "-" * 40)

                val_stats = evaluate(model, val_loader, config)

                print(f"Val Loss: {val_stats['loss']:.4f} | Val PPL: {val_stats['perplexity']:.1f}")
                print(f"Val R: {val_stats['R_mean']:.4f} ± {val_stats['R_std']:.4f} "
                      f"[{val_stats['R_min']:.3f}, {val_stats['R_max']:.3f}]")
                print(f"R per layer: {['%.3f' % r for r in val_stats['R_per_layer']]}")

                # Generate sample
                sample_text, sample_R = generate_sample(
                    model, tokenizer, "The ", max_tokens=50,
                    temperature=0.8, config=config
                )
                print(f"Sample: {sample_text[:100]}...")
                print(f"Sample R: mean={sum(sample_R)/len(sample_R):.3f}, "
                      f"std={torch.tensor(sample_R).std().item():.3f}")

                # Record history
                history.append({
                    'step': global_step,
                    'train_loss': avg_loss if n_running == 0 else running_loss / max(n_running, 1),
                    'val_loss': val_stats['loss'],
                    'val_perplexity': val_stats['perplexity'],
                    'R_mean': val_stats['R_mean'],
                    'R_std': val_stats['R_std'],
                    'R_range': val_stats['R_max'] - val_stats['R_min'],
                    'R_per_layer': val_stats['R_per_layer'],
                    'elapsed': time.time() - start_time
                })

                # Save best
                if val_stats['loss'] < best_val_loss:
                    best_val_loss = val_stats['loss']
                    torch.save({
                        'model_state': model.state_dict(),
                        'config': vars(config),
                        'step': global_step,
                        'val_loss': val_stats['loss']
                    }, output_dir / "best_model.pt")
                    print(f"  Saved best model (val_loss={val_stats['loss']:.4f})")

                print("-" * 90 + "\n")

            # Checkpoint
            if global_step % config.save_interval == 0:
                torch.save({
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'config': vars(config),
                    'step': global_step,
                    'history': history
                }, output_dir / f"checkpoint_{global_step}.pt")

    # Final evaluation
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    val_stats = evaluate(model, val_loader, config, max_batches=100)

    print(f"Final val loss: {val_stats['loss']:.4f}")
    print(f"Final perplexity: {val_stats['perplexity']:.1f}")
    print(f"Final R: {val_stats['R_mean']:.4f} ± {val_stats['R_std']:.4f}")
    print(f"R range: [{val_stats['R_min']:.3f}, {val_stats['R_max']:.3f}]")

    # R dynamics check
    R_varies = val_stats['R_std'] > 0.01
    print(f"\n✅ R varies at inference: {R_varies} (std={val_stats['R_std']:.4f})")

    # Save final model and history
    torch.save({
        'model_state': model.state_dict(),
        'config': vars(config),
        'final_stats': val_stats
    }, output_dir / "final_model.pt")

    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print("  - best_model.pt")
    print("  - final_model.pt")
    print("  - training_history.json")

    return model, history


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train K-SSM v2")
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["small", "medium"], help="Model size")
    parser.add_argument("--corpus", type=str, default="data/processed/kssm_corpus.jsonl",
                        help="Path to corpus JSONL")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="results/kssm_v2")

    args = parser.parse_args()

    config = TrainConfig(
        model_size=args.model_size,
        corpus_path=args.corpus,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        output_dir=args.output_dir
    )

    model, history = train(config)

    print("\n✅ K-SSM v2 training complete!")
