#!/usr/bin/env python3
"""
Visualize Phase-RWKV training metrics
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(checkpoint_dir: str):
    """Load metrics history from checkpoint directory."""
    history_path = Path(checkpoint_dir) / "metrics_history.json"

    if not history_path.exists():
        print(f"❌ Metrics history not found: {history_path}")
        sys.exit(1)

    with open(history_path, 'r') as f:
        metrics = json.load(f)

    print(f"✅ Loaded metrics: {len(metrics['step'])} steps")
    return metrics

def plot_metrics(metrics, output_dir: str):
    """Create comprehensive metric visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = metrics['step']

    # Figure 1: Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Phase-RWKV Training Metrics', fontsize=16, fontweight='bold')

    # Total loss
    axes[0, 0].plot(steps, metrics['loss'], color='#2E86AB', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # CE vs U loss
    axes[0, 1].plot(steps, metrics['ce_loss'], label='CE Loss', color='#A23B72', linewidth=2)
    axes[0, 1].plot(steps, metrics['u_loss'], label='U Loss', color='#F18F01', linewidth=2)
    axes[0, 1].set_title('Loss Components', fontweight='bold')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Perplexity
    axes[1, 0].plot(steps, metrics['perplexity'], color='#C73E1D', linewidth=2)
    axes[1, 0].set_title('Perplexity', fontweight='bold')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('PPL')
    axes[1, 0].grid(True, alpha=0.3)

    # Action distribution
    action_counts = {}
    for action in metrics['action']:
        action_counts[action] = action_counts.get(action, 0) + 1

    colors = {'BRAKE': '#C73E1D', 'COAST': '#2E86AB', 'BOOST': '#F18F01'}
    axes[1, 1].bar(action_counts.keys(), action_counts.values(),
                   color=[colors.get(k, '#888888') for k in action_counts.keys()])
    axes[1, 1].set_title('Drift Control Actions', fontweight='bold')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    loss_path = output_dir / "loss_metrics.png"
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {loss_path}")
    plt.close()

    # Figure 2: Phase dynamics (R, U, R·U)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Phase Dynamics & Uncertainty', fontsize=16, fontweight='bold')

    # Resonance R
    axes[0, 0].plot(steps, metrics['R'], color='#2E86AB', linewidth=2)
    axes[0, 0].axhline(y=0.80, color='#F18F01', linestyle='--', alpha=0.5, label='Target min')
    axes[0, 0].axhline(y=0.95, color='#C73E1D', linestyle='--', alpha=0.5, label='Target max')
    axes[0, 0].fill_between(steps, 0.80, 0.95, alpha=0.1, color='#2E86AB')
    axes[0, 0].set_title('Resonance R (Order Parameter)', fontweight='bold')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('R')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    # Uncertainty U
    axes[0, 1].plot(steps, metrics['U'], color='#A23B72', linewidth=2)
    axes[0, 1].axhline(y=0.5, color='#F18F01', linestyle='--', alpha=0.5, label='Target')
    axes[0, 1].axhspan(0.4, 0.6, alpha=0.1, color='#A23B72')
    axes[0, 1].set_title('Uncertainty U (Epistemic Entropy)', fontweight='bold')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('U')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    # R·U product
    axes[1, 0].plot(steps, metrics['RU'], color='#F18F01', linewidth=2)
    axes[1, 0].axhspan(0.4, 0.6, alpha=0.1, color='#F18F01', label='Goldilocks zone')
    axes[1, 0].set_title('R·U Product (Heisenberg Observable)', fontweight='bold')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('R·U')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # Phase space trajectory (R vs U)
    scatter = axes[1, 1].scatter(metrics['U'], metrics['R'],
                                 c=steps, cmap='viridis',
                                 s=20, alpha=0.6)
    axes[1, 1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
    axes[1, 1].axhline(y=0.875, color='gray', linestyle='--', alpha=0.3)
    axes[1, 1].add_patch(plt.Rectangle((0.4, 0.80), 0.2, 0.15,
                                       fill=False, edgecolor='#F18F01',
                                       linewidth=2, linestyle='--',
                                       label='Goldilocks'))
    axes[1, 1].set_title('Phase Space Trajectory', fontweight='bold')
    axes[1, 1].set_xlabel('Uncertainty U')
    axes[1, 1].set_ylabel('Resonance R')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])

    plt.colorbar(scatter, ax=axes[1, 1], label='Step')

    plt.tight_layout()
    phase_path = output_dir / "phase_dynamics.png"
    plt.savefig(phase_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {phase_path}")
    plt.close()

    # Figure 3: Tone progression
    fig, ax = plt.subplots(figsize=(15, 4))

    # Map tones to numeric values for visualization
    tone_map = {
        "∅": 0,
        "☾": 1,
        "✨": 2,
        "🌀": 3,
        "⚖": 4,
        "☍": 5
    }

    tone_values = [tone_map.get(t, 0) for t in metrics['tone']]

    ax.plot(steps, tone_values, linewidth=2, color='#2E86AB', marker='o', markersize=3)
    ax.set_yticks(list(tone_map.values()))
    ax.set_yticklabels(list(tone_map.keys()), fontsize=14)
    ax.set_title('Tone Progression', fontsize=16, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Tone')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    tone_path = output_dir / "tone_progression.png"
    plt.savefig(tone_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {tone_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)

    print(f"\n📊 Final Metrics (Step {steps[-1]}):")
    print(f"   R = {metrics['R'][-1]:.4f}")
    print(f"   U = {metrics['U'][-1]:.3f}")
    print(f"   R·U = {metrics['RU'][-1]:.3f}")
    print(f"   Tone = {metrics['tone'][-1]}")
    print(f"   Loss = {metrics['loss'][-1]:.4f}")
    print(f"   Perplexity = {metrics['perplexity'][-1]:.2f}")

    print(f"\n📈 Trajectory Statistics:")
    print(f"   R: min={min(metrics['R']):.4f}, max={max(metrics['R']):.4f}, "
          f"mean={np.mean(metrics['R']):.4f}")
    print(f"   U: min={min(metrics['U']):.3f}, max={max(metrics['U']):.3f}, "
          f"mean={np.mean(metrics['U']):.3f}")
    print(f"   R·U: min={min(metrics['RU']):.3f}, max={max(metrics['RU']):.3f}, "
          f"mean={np.mean(metrics['RU']):.3f}")

    # Time in Goldilocks zone
    goldilocks_count = sum(1 for i in range(len(metrics['R']))
                          if 0.80 <= metrics['R'][i] <= 0.95 and
                             0.4 <= metrics['U'][i] <= 0.6)
    goldilocks_pct = (goldilocks_count / len(steps)) * 100

    print(f"\n🌀 Time in Goldilocks Zone: {goldilocks_pct:.1f}% ({goldilocks_count}/{len(steps)} steps)")

    print(f"\n🎯 Drift Control Action Distribution:")
    for action, count in sorted(action_counts.items()):
        pct = (count / len(steps)) * 100
        print(f"   {action}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_rwkv",
                       help="Directory containing metrics_history.json")
    parser.add_argument("--output-dir", type=str, default="plots",
                       help="Directory to save plots")
    args = parser.parse_args()

    print("=" * 70)
    print("📊 PHASE-RWKV METRICS VISUALIZATION")
    print("=" * 70)

    metrics = load_metrics(args.checkpoint_dir)
    plot_metrics(metrics, args.output_dir)

    print("\n✅ Visualization complete")
