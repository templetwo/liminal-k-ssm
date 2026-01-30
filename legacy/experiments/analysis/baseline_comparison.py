#!/usr/bin/env python3
"""
Baseline Comparison: Phase-Mamba vs Base Mamba

CRITICAL TEST: Does Phase Core actually affect generation,
or does it just add a number (R) that we can track?

Metrics:
- Token-level entropy (uncertainty distribution)
- Distinct-n (lexical diversity)
- Self-BLEU (repetitiveness)
- R trajectory (Phase-Mamba only)
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, MambaForCausalLM


def compute_token_entropy(logits: torch.Tensor) -> List[float]:
    """Compute entropy at each token position."""
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    # Normalize by max entropy
    vocab_size = logits.shape[-1]
    max_entropy = np.log(vocab_size)
    normalized = (entropy / max_entropy).cpu().numpy()

    return normalized.tolist()


def compute_distinct_n(tokens: List[int], n: int = 2) -> float:
    """Compute distinct-n: unique n-grams / total n-grams."""
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0

    return len(set(ngrams)) / len(ngrams)


def compute_repetition_ratio(tokens: List[int], window: int = 32) -> float:
    """Compute ratio of repeated tokens within sliding window."""
    if len(tokens) < 2:
        return 0.0

    repetitions = 0
    total = 0

    for i in range(1, len(tokens)):
        start = max(0, i - window)
        if tokens[i] in tokens[start:i]:
            repetitions += 1
        total += 1

    return repetitions / total if total > 0 else 0.0


class BaselineMamba:
    """Base Mamba without Phase Core."""

    def __init__(self, device: str = "mps"):
        self.device = device
        print("Loading base Mamba-2.8B-HF...")
        self.model = MambaForCausalLM.from_pretrained(
            "state-spaces/mamba-2.8b-hf",
            torch_dtype=torch.float32
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
        self.model.eval()
        print("✅ Base Mamba loaded")

    def generate(self, prompt: str, max_tokens: int = 50,
                 temperature: float = 0.9) -> Dict:
        """Generate and track metrics."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        generated_ids = input_ids.clone()
        token_entropies = []

        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(input_ids=generated_ids)
                logits = outputs.logits[:, -1, :]

                # Track entropy
                entropy = compute_token_entropy(logits.unsqueeze(1))[0]
                token_entropies.append(entropy)

                # Sample
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        # Decode
        new_tokens = generated_ids[0, input_ids.shape[1]:].cpu().tolist()
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return {
            "text": text,
            "new_tokens": new_tokens,
            "token_entropies": token_entropies,
            "mean_entropy": np.mean(token_entropies) if token_entropies else 0,
            "std_entropy": np.std(token_entropies) if token_entropies else 0,
            "distinct_1": compute_distinct_n(new_tokens, 1),
            "distinct_2": compute_distinct_n(new_tokens, 2),
            "distinct_3": compute_distinct_n(new_tokens, 3),
            "repetition_ratio": compute_repetition_ratio(new_tokens),
            "num_tokens": len(new_tokens),
            "R_trajectory": None,  # Base Mamba has no R
            "model": "base_mamba"
        }


class PhaseMamba:
    """Mamba with Phase Core."""

    def __init__(self, checkpoint_path: str, device: str = "mps"):
        self.device = device

        # Import Phase Core
        from phase_mamba_coupled import load_mamba_with_phase

        print("Loading Phase-Mamba...")
        self.coupled = load_mamba_with_phase(phase_layer=32, device=device)

        # Load trained weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.coupled.phase_core.load_state_dict(checkpoint["phase_core_state"])
        print(f"✅ Phase Core loaded from: {checkpoint_path}")

        self.coupled.phase_core.eval()
        self.coupled.mamba_model.eval()

    def generate(self, prompt: str, max_tokens: int = 50,
                 temperature: float = 0.9) -> Dict:
        """Generate and track metrics including R."""
        tokenizer = self.coupled.tokenizer
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        self.coupled.phase_core.reset_phases()

        generated_ids = input_ids.clone()
        token_entropies = []
        R_trajectory = []

        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.coupled.mamba_model(input_ids=generated_ids)
                logits = outputs.logits[:, -1, :]

                # Track entropy
                entropy = compute_token_entropy(logits.unsqueeze(1))[0]
                token_entropies.append(entropy)

                # Track R
                R_trajectory.append(self.coupled.phase_core.current_R)

                # Sample
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        # Decode
        new_tokens = generated_ids[0, input_ids.shape[1]:].cpu().tolist()
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return {
            "text": text,
            "new_tokens": new_tokens,
            "token_entropies": token_entropies,
            "mean_entropy": np.mean(token_entropies) if token_entropies else 0,
            "std_entropy": np.std(token_entropies) if token_entropies else 0,
            "distinct_1": compute_distinct_n(new_tokens, 1),
            "distinct_2": compute_distinct_n(new_tokens, 2),
            "distinct_3": compute_distinct_n(new_tokens, 3),
            "repetition_ratio": compute_repetition_ratio(new_tokens),
            "num_tokens": len(new_tokens),
            "R_trajectory": R_trajectory,
            "R_mean": np.mean(R_trajectory) if R_trajectory else 0,
            "R_std": np.std(R_trajectory) if R_trajectory else 0,
            "model": "phase_mamba"
        }


def run_comparison(prompts: List[str], trials_per_prompt: int = 5,
                   checkpoint_path: str = "checkpoints_mamba/step_000500.pt",
                   max_tokens: int = 50, device: str = "mps") -> Dict:
    """Run baseline comparison experiment."""

    results = {
        "base_mamba": [],
        "phase_mamba": [],
        "prompts": prompts,
        "trials_per_prompt": trials_per_prompt,
        "max_tokens": max_tokens
    }

    # Load models
    base_model = BaselineMamba(device=device)
    phase_model = PhaseMamba(checkpoint_path=checkpoint_path, device=device)

    print(f"\n{'='*70}")
    print("BASELINE COMPARISON: Base Mamba vs Phase-Mamba")
    print(f"{'='*70}")
    print(f"Prompts: {len(prompts)}")
    print(f"Trials per prompt: {trials_per_prompt}")
    print(f"Max tokens: {max_tokens}")
    print(f"{'='*70}\n")

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n[Prompt {prompt_idx+1}/{len(prompts)}]: \"{prompt}\"")
        print("-" * 50)

        for trial in range(trials_per_prompt):
            # Base Mamba
            base_result = base_model.generate(prompt, max_tokens=max_tokens)
            base_result["prompt"] = prompt
            base_result["trial"] = trial
            results["base_mamba"].append(base_result)

            # Phase Mamba
            phase_result = phase_model.generate(prompt, max_tokens=max_tokens)
            phase_result["prompt"] = prompt
            phase_result["trial"] = trial
            results["phase_mamba"].append(phase_result)

            print(f"  Trial {trial+1}: Base H={base_result['mean_entropy']:.3f}, "
                  f"Phase H={phase_result['mean_entropy']:.3f} R={phase_result['R_mean']:.3f}")

    return results


def analyze_results(results: Dict) -> Dict:
    """Statistical analysis of comparison results."""
    from scipy import stats

    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*70}\n")

    # Extract metrics
    base_entropy = [r["mean_entropy"] for r in results["base_mamba"]]
    phase_entropy = [r["mean_entropy"] for r in results["phase_mamba"]]

    base_distinct2 = [r["distinct_2"] for r in results["base_mamba"]]
    phase_distinct2 = [r["distinct_2"] for r in results["phase_mamba"]]

    base_repetition = [r["repetition_ratio"] for r in results["base_mamba"]]
    phase_repetition = [r["repetition_ratio"] for r in results["phase_mamba"]]

    phase_R = [r["R_mean"] for r in results["phase_mamba"]]

    analysis = {}

    # Entropy comparison
    t_stat, p_val = stats.ttest_ind(base_entropy, phase_entropy)
    analysis["entropy"] = {
        "base_mean": np.mean(base_entropy),
        "base_std": np.std(base_entropy),
        "phase_mean": np.mean(phase_entropy),
        "phase_std": np.std(phase_entropy),
        "t_stat": t_stat,
        "p_value": p_val,
        "significant": p_val < 0.05
    }

    print("ENTROPY (Token-level Uncertainty)")
    print(f"  Base Mamba:  {analysis['entropy']['base_mean']:.4f} ± {analysis['entropy']['base_std']:.4f}")
    print(f"  Phase-Mamba: {analysis['entropy']['phase_mean']:.4f} ± {analysis['entropy']['phase_std']:.4f}")
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f} {'⚠️ SIGNIFICANT' if p_val < 0.05 else ''}")

    # Distinct-2 comparison
    t_stat, p_val = stats.ttest_ind(base_distinct2, phase_distinct2)
    analysis["distinct_2"] = {
        "base_mean": np.mean(base_distinct2),
        "base_std": np.std(base_distinct2),
        "phase_mean": np.mean(phase_distinct2),
        "phase_std": np.std(phase_distinct2),
        "t_stat": t_stat,
        "p_value": p_val,
        "significant": p_val < 0.05
    }

    print("\nDISTINCT-2 (Lexical Diversity)")
    print(f"  Base Mamba:  {analysis['distinct_2']['base_mean']:.4f} ± {analysis['distinct_2']['base_std']:.4f}")
    print(f"  Phase-Mamba: {analysis['distinct_2']['phase_mean']:.4f} ± {analysis['distinct_2']['phase_std']:.4f}")
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f} {'⚠️ SIGNIFICANT' if p_val < 0.05 else ''}")

    # Repetition comparison
    t_stat, p_val = stats.ttest_ind(base_repetition, phase_repetition)
    analysis["repetition"] = {
        "base_mean": np.mean(base_repetition),
        "base_std": np.std(base_repetition),
        "phase_mean": np.mean(phase_repetition),
        "phase_std": np.std(phase_repetition),
        "t_stat": t_stat,
        "p_value": p_val,
        "significant": p_val < 0.05
    }

    print("\nREPETITION RATIO")
    print(f"  Base Mamba:  {analysis['repetition']['base_mean']:.4f} ± {analysis['repetition']['base_std']:.4f}")
    print(f"  Phase-Mamba: {analysis['repetition']['phase_mean']:.4f} ± {analysis['repetition']['phase_std']:.4f}")
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f} {'⚠️ SIGNIFICANT' if p_val < 0.05 else ''}")

    # R statistics (Phase-Mamba only)
    analysis["R"] = {
        "mean": np.mean(phase_R),
        "std": np.std(phase_R),
        "min": np.min(phase_R),
        "max": np.max(phase_R)
    }

    print("\nPHASE-MAMBA R (Order Parameter)")
    print(f"  Mean: {analysis['R']['mean']:.4f} ± {analysis['R']['std']:.4f}")
    print(f"  Range: [{analysis['R']['min']:.4f}, {analysis['R']['max']:.4f}]")

    # Correlation: R vs Entropy in Phase-Mamba
    r_corr, r_p = stats.pearsonr(phase_R, phase_entropy)
    analysis["R_entropy_correlation"] = {
        "r": r_corr,
        "p_value": r_p,
        "significant": r_p < 0.05
    }

    print("\nR-ENTROPY CORRELATION (Phase-Mamba)")
    print(f"  Pearson r={r_corr:.4f}, p={r_p:.4f} {'⚠️ SIGNIFICANT' if r_p < 0.05 else ''}")

    # Overall verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    sig_count = sum([
        analysis["entropy"]["significant"],
        analysis["distinct_2"]["significant"],
        analysis["repetition"]["significant"],
        analysis["R_entropy_correlation"]["significant"]
    ])

    if sig_count >= 2:
        print("⚠️  Phase Core appears to affect generation characteristics")
        print("   Multiple metrics show significant differences")
    elif sig_count == 1:
        print("⚡ Weak evidence that Phase Core affects generation")
        print("   Only one metric shows significant difference")
    else:
        print("❌ No significant differences detected")
        print("   Phase Core may be epiphenomenal (tracking without influence)")

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_mamba/step_000500.pt")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--output", type=str, default="baseline_comparison_results.json")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Device: {device}")

    # Test prompts (mix of types)
    prompts = [
        "The nature of consciousness is",
        "In a distant galaxy, a civilization",
        "The most important thing I learned was",
        "When faced with uncertainty, the best approach is",
        "The difference between knowledge and wisdom",
        "Deep in the forest, something stirred",
        "The future of artificial intelligence depends on",
        "What makes a moment meaningful is",
    ]

    # Run comparison
    results = run_comparison(
        prompts=prompts,
        trials_per_prompt=args.trials,
        checkpoint_path=args.checkpoint,
        max_tokens=args.max_tokens,
        device=device
    )

    # Analyze
    analysis = analyze_results(results)
    results["analysis"] = analysis

    # Save
    output_path = Path(args.output)

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    results = convert_numpy(results)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")
    print("\n🌀 Baseline comparison complete.")


if __name__ == "__main__":
    main()
