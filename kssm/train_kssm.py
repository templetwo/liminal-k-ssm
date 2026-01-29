#!/usr/bin/env python3
"""
Train K-SSM on TinyShakespeare

This script:
1. Downloads TinyShakespeare (character-level)
2. Trains K-SSM
3. Tracks R dynamics throughout training
4. Generates samples with R-modulated temperature
"""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import requests

from kssm_model import KSSM, count_parameters


# ==============================================================================
# Data Loading
# ==============================================================================

def download_tiny_shakespeare(data_dir="data"):
    """Download TinyShakespeare dataset."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "tiny_shakespeare.txt")

    if not os.path.exists(filepath):
        print("Downloading TinyShakespeare...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        with open(filepath, 'w') as f:
            f.write(response.text)
        print(f"Saved to {filepath}")

    with open(filepath, 'r') as f:
        text = f.read()

    return text


class CharDataset(Dataset):
    """Character-level dataset."""

    def __init__(self, text, seq_length=64):
        self.text = text
        self.seq_length = seq_length

        # Build vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

        # Encode full text
        self.data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y

    def decode(self, indices):
        """Convert indices back to string."""
        return ''.join([self.idx_to_char[i.item()] for i in indices])


# ==============================================================================
# Training
# ==============================================================================

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    R_values = []
    n_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, R = model(x, return_R=True)

        loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        R_values.extend(R.mean(dim=1).tolist())  # Mean R per sequence
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "R_mean": sum(R_values) / len(R_values),
        "R_std": torch.tensor(R_values).std().item(),
        "R_min": min(R_values),
        "R_max": max(R_values)
    }


def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    R_values = []
    n_batches = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, R = model(x, return_R=True)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

            total_loss += loss.item()
            R_values.extend(R.mean(dim=1).tolist())
            n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "perplexity": torch.exp(torch.tensor(total_loss / n_batches)).item(),
        "R_mean": sum(R_values) / len(R_values),
        "R_std": torch.tensor(R_values).std().item(),
        "R_range": max(R_values) - min(R_values)
    }


# ==============================================================================
# Generation
# ==============================================================================

def generate(model, dataset, prompt="ROMEO:", max_tokens=200, temperature=1.0,
             use_R_modulation=True, device="cpu"):
    """
    Generate text with optional R-modulated temperature.

    R-modulation: High R → lower temperature (more confident)
                  Low R → higher temperature (more exploratory)
    """
    model.eval()

    # Encode prompt
    tokens = [dataset.char_to_idx[c] for c in prompt]
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    generated = list(tokens)
    R_during_gen = []

    with torch.no_grad():
        for _ in range(max_tokens):
            # Use last seq_length tokens as context
            context = x[:, -64:] if x.shape[1] > 64 else x

            logits, R = model(context, return_R=True)
            R_value = R[0, -1].item()
            R_during_gen.append(R_value)

            # Get logits for next token
            next_logits = logits[0, -1, :]

            # R-modulated temperature
            if use_R_modulation:
                # High R → temp closer to 0.5, Low R → temp closer to 1.5
                effective_temp = 1.0 - 0.5 * R_value + 0.5
            else:
                effective_temp = temperature

            # Sample
            probs = F.softmax(next_logits / effective_temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token.item())
            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

    text = dataset.decode(torch.tensor(generated))
    return text, R_during_gen


# ==============================================================================
# Main
# ==============================================================================

def main():
    # Config
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    config = {
        "embed_dim": 128,
        "n_oscillators": 64,
        "n_harmonics": 8,
        "coupling_strength": 2.0,
        "seq_length": 64,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "n_epochs": 10,
    }

    # Data
    print("\n[1] Loading data...")
    text = download_tiny_shakespeare()
    print(f"Dataset size: {len(text):,} characters")

    # Split
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_dataset = CharDataset(train_text, seq_length=config["seq_length"])
    val_dataset = CharDataset(val_text, seq_length=config["seq_length"])

    # Use same vocab for both
    val_dataset.char_to_idx = train_dataset.char_to_idx
    val_dataset.idx_to_char = train_dataset.idx_to_char
    val_dataset.chars = train_dataset.chars
    val_dataset.vocab_size = train_dataset.vocab_size

    print(f"Vocab size: {train_dataset.vocab_size}")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # Model
    print("\n[2] Creating K-SSM model...")
    model = KSSM(
        vocab_size=train_dataset.vocab_size,
        embed_dim=config["embed_dim"],
        n_oscillators=config["n_oscillators"],
        n_harmonics=config["n_harmonics"],
        coupling_strength=config["coupling_strength"]
    ).to(device)

    print(f"Parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # Training
    print("\n[3] Training K-SSM...")
    print("=" * 70)

    history = []
    best_val_loss = float('inf')

    for epoch in range(config["n_epochs"]):
        start_time = time.time()

        train_stats = train_epoch(model, train_loader, optimizer, device)
        val_stats = evaluate(model, val_loader, device)

        elapsed = time.time() - start_time

        # Log
        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "val_perplexity": val_stats["perplexity"],
            "R_mean": train_stats["R_mean"],
            "R_std": train_stats["R_std"],
            "R_min": train_stats["R_min"],
            "R_max": train_stats["R_max"],
            "R_range": val_stats["R_range"],
            "tone": model.get_tone(train_stats["R_mean"]),
            "time": elapsed
        }
        history.append(epoch_stats)

        print(f"Epoch {epoch+1:2d}/{config['n_epochs']} | "
              f"Loss: {train_stats['loss']:.3f}/{val_stats['loss']:.3f} | "
              f"PPL: {val_stats['perplexity']:.1f} | "
              f"R: {train_stats['R_mean']:.3f}±{train_stats['R_std']:.3f} "
              f"[{train_stats['R_min']:.2f},{train_stats['R_max']:.2f}] | "
              f"{epoch_stats['tone']} | {elapsed:.1f}s")

        # Save best
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "vocab": {
                    "char_to_idx": train_dataset.char_to_idx,
                    "idx_to_char": train_dataset.idx_to_char
                }
            }, "results/best_model.pt")

    print("=" * 70)

    # Save training history
    with open("results/training_log.json", 'w') as f:
        json.dump(history, f, indent=2)

    # Generation samples
    print("\n[4] Generating samples...")
    print("-" * 70)

    prompts = ["ROMEO:", "To be or", "The king", "What light"]

    samples = []
    for prompt in prompts:
        # With R-modulation
        text_mod, R_mod = generate(model, train_dataset, prompt=prompt,
                                   max_tokens=100, use_R_modulation=True, device=device)
        # Without R-modulation
        text_std, R_std = generate(model, train_dataset, prompt=prompt,
                                   max_tokens=100, use_R_modulation=False, device=device)

        print(f"\nPrompt: '{prompt}'")
        print(f"  R-modulated: {text_mod[:80]}...")
        print(f"  Standard:    {text_std[:80]}...")
        print(f"  R during gen (mod): mean={sum(R_mod)/len(R_mod):.3f}, "
              f"std={torch.tensor(R_mod).std().item():.3f}")

        samples.append({
            "prompt": prompt,
            "r_modulated": text_mod,
            "standard": text_std,
            "R_trajectory": R_mod
        })

    # Save samples
    with open("results/samples.json", 'w') as f:
        json.dump(samples, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final val loss: {history[-1]['val_loss']:.3f}")
    print(f"Final perplexity: {history[-1]['val_perplexity']:.1f}")
    print(f"R range during training: [{min(h['R_min'] for h in history):.3f}, "
          f"{max(h['R_max'] for h in history):.3f}]")
    print(f"R std (mean): {sum(h['R_std'] for h in history)/len(history):.3f}")

    # R dynamics check
    R_varies = history[-1]["R_std"] > 0.05
    print(f"\n✅ R varies at inference: {R_varies} (std={history[-1]['R_std']:.3f})")

    print(f"\nSaved to results/")
    print("  - best_model.pt")
    print("  - training_log.json")
    print("  - samples.json")


if __name__ == "__main__":
    main()
