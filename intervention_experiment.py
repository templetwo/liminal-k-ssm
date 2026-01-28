#!/usr/bin/env python3
"""
Intervention Experiment: Force R to specific values during generation

CRITICAL TEST: Can R affect generation if we make it move?

If forcing R to different values produces different outputs:
  → R is causal but needs external control during inference
  → The architecture could work with a control mechanism

If forcing R doesn't change output:
  → R is completely disconnected from generation
  → Layers 33-63 ignore the modulation at layer 32
  → Architecture needs fundamental redesign
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def compute_token_entropy(logits: torch.Tensor) -> float:
    """Compute normalized entropy."""
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    vocab_size = logits.shape[-1]
    max_entropy = np.log(vocab_size)
    return (entropy / max_entropy).mean().item()


def compute_distinct_n(tokens: List[int], n: int = 2) -> float:
    """Compute distinct-n metric."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


class ForcedRPhaseMamba:
    """Phase-Mamba with forced R values during generation."""

    def __init__(self, checkpoint_path: str, device: str = "mps"):
        self.device = device

        from phase_mamba_coupled import load_mamba_with_phase
        print("Loading Phase-Mamba for intervention...")
        self.coupled = load_mamba_with_phase(phase_layer=32, device=device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.coupled.phase_core.load_state_dict(checkpoint["phase_core_state"])
        print(f"✅ Loaded from: {checkpoint_path}")

        self.coupled.phase_core.eval()
        self.coupled.mamba_model.eval()

    def generate_with_forced_R(self, prompt: str, target_R: float,
                                max_tokens: int = 50,
                                temperature: float = 0.9) -> Dict:
        """
        Generate while forcing R to a specific value.

        We do this by overriding the oscillator phases to produce
        the desired R value before each forward pass.
        """
        tokenizer = self.coupled.tokenizer
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        phase_core = self.coupled.phase_core

        # Store original phases
        original_phases = phase_core.phases.clone()

        generated_ids = input_ids.clone()
        token_entropies = []
        actual_R_values = []

        with torch.no_grad():
            for step in range(max_tokens):
                # INTERVENTION: Force phases to achieve target R
                # R = |mean(exp(i*theta))|
                # To get R = target, we set all phases equal (gives R=1)
                # then add controlled dispersion to reduce R

                if target_R >= 0.99:
                    # All phases aligned → R ≈ 1
                    forced_phases = torch.zeros_like(phase_core.phases)
                elif target_R <= 0.01:
                    # Phases uniformly distributed → R ≈ 0
                    n = phase_core.phases.shape[0]
                    forced_phases = torch.linspace(0, 2*np.pi*(1-1/n), n).to(self.device)
                else:
                    # Intermediate R: use von Mises-like distribution
                    # Higher concentration κ → higher R
                    # Approximate: R ≈ I₁(κ)/I₀(κ) for von Mises
                    # We'll use a simpler approach: linear interpolation of spread

                    n = phase_core.phases.shape[0]
                    # Spread from 0 to 2π as R goes from 1 to 0
                    spread = 2 * np.pi * (1 - target_R)
                    forced_phases = torch.linspace(-spread/2, spread/2, n).to(self.device)

                # Apply forced phases
                phase_core.phases.data = forced_phases

                # Forward pass
                outputs = self.coupled.mamba_model(input_ids=generated_ids)
                logits = outputs.logits[:, -1, :]

                # Track actual R achieved
                actual_R = phase_core.current_R
                actual_R_values.append(actual_R)

                # Track entropy
                entropy = compute_token_entropy(logits)
                token_entropies.append(entropy)

                # Sample next token
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        # Restore original phases
        phase_core.phases.data = original_phases

        # Decode
        new_tokens = generated_ids[0, input_ids.shape[1]:].cpu().tolist()
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return {
            "text": text,
            "new_tokens": new_tokens,
            "target_R": target_R,
            "actual_R_mean": np.mean(actual_R_values),
            "actual_R_std": np.std(actual_R_values),
            "mean_entropy": np.mean(token_entropies),
            "std_entropy": np.std(token_entropies),
            "distinct_1": compute_distinct_n(new_tokens, 1),
            "distinct_2": compute_distinct_n(new_tokens, 2),
            "num_tokens": len(new_tokens)
        }

    def generate_free_running(self, prompt: str, max_tokens: int = 50,
                              temperature: float = 0.9) -> Dict:
        """Generate without forcing R (control condition)."""
        tokenizer = self.coupled.tokenizer
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        self.coupled.phase_core.reset_phases()

        generated_ids = input_ids.clone()
        token_entropies = []
        R_values = []

        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.coupled.mamba_model(input_ids=generated_ids)
                logits = outputs.logits[:, -1, :]

                R_values.append(self.coupled.phase_core.current_R)
                token_entropies.append(compute_token_entropy(logits))

                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        new_tokens = generated_ids[0, input_ids.shape[1]:].cpu().tolist()
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return {
            "text": text,
            "new_tokens": new_tokens,
            "target_R": "free",
            "actual_R_mean": np.mean(R_values),
            "actual_R_std": np.std(R_values),
            "mean_entropy": np.mean(token_entropies),
            "std_entropy": np.std(token_entropies),
            "distinct_1": compute_distinct_n(new_tokens, 1),
            "distinct_2": compute_distinct_n(new_tokens, 2),
            "num_tokens": len(new_tokens)
        }


def run_intervention_experiment(prompts: List[str],
                                 target_R_values: List[float],
                                 trials_per_condition: int = 5,
                                 checkpoint_path: str = "checkpoints_mamba/step_000500.pt",
                                 max_tokens: int = 50,
                                 device: str = "mps") -> Dict:
    """Run the intervention experiment."""

    results = {
        "conditions": {},
        "prompts": prompts,
        "target_R_values": target_R_values,
        "trials_per_condition": trials_per_condition
    }

    model = ForcedRPhaseMamba(checkpoint_path=checkpoint_path, device=device)

    print(f"\n{'='*70}")
    print("INTERVENTION EXPERIMENT: Forcing R during generation")
    print(f"{'='*70}")
    print(f"Target R values: {target_R_values}")
    print(f"Prompts: {len(prompts)}")
    print(f"Trials per condition: {trials_per_condition}")
    print(f"{'='*70}\n")

    # Add free-running condition
    all_conditions = ["free"] + target_R_values

    for condition in all_conditions:
        condition_key = f"R={condition}" if condition != "free" else "free"
        results["conditions"][condition_key] = []

        print(f"\n[Condition: {condition_key}]")
        print("-" * 40)

        for prompt in prompts:
            for trial in range(trials_per_condition):
                if condition == "free":
                    result = model.generate_free_running(prompt, max_tokens=max_tokens)
                else:
                    result = model.generate_with_forced_R(prompt, target_R=condition,
                                                          max_tokens=max_tokens)

                result["prompt"] = prompt
                result["trial"] = trial
                results["conditions"][condition_key].append(result)

                print(f"  {prompt[:30]}... trial {trial+1}: "
                      f"H={result['mean_entropy']:.3f}, "
                      f"R_actual={result['actual_R_mean']:.3f}")

    return results


def analyze_intervention_results(results: Dict) -> Dict:
    """Analyze whether forcing R changes output characteristics."""
    from scipy import stats

    print(f"\n{'='*70}")
    print("INTERVENTION ANALYSIS")
    print(f"{'='*70}\n")

    analysis = {}

    # Extract metrics by condition
    conditions = list(results["conditions"].keys())
    metrics_by_condition = {}

    for cond in conditions:
        trials = results["conditions"][cond]
        metrics_by_condition[cond] = {
            "entropy": [t["mean_entropy"] for t in trials],
            "distinct_2": [t["distinct_2"] for t in trials],
            "actual_R": [t["actual_R_mean"] for t in trials]
        }

    # Print summary stats
    print("SUMMARY BY CONDITION:")
    print("-" * 60)
    print(f"{'Condition':<12} {'Entropy':>12} {'Distinct-2':>12} {'Actual R':>12}")
    print("-" * 60)

    for cond in conditions:
        m = metrics_by_condition[cond]
        print(f"{cond:<12} {np.mean(m['entropy']):>12.4f} "
              f"{np.mean(m['distinct_2']):>12.4f} "
              f"{np.mean(m['actual_R']):>12.4f}")

    print("-" * 60)

    # Statistical tests: compare each forced condition to free-running
    analysis["comparisons"] = {}
    free_entropy = metrics_by_condition["free"]["entropy"]
    free_distinct = metrics_by_condition["free"]["distinct_2"]

    print("\nCOMPARISONS TO FREE-RUNNING:")
    print("-" * 60)

    significant_entropy = 0
    significant_distinct = 0

    for cond in conditions:
        if cond == "free":
            continue

        cond_entropy = metrics_by_condition[cond]["entropy"]
        cond_distinct = metrics_by_condition[cond]["distinct_2"]

        # t-tests
        t_ent, p_ent = stats.ttest_ind(free_entropy, cond_entropy)
        t_dist, p_dist = stats.ttest_ind(free_distinct, cond_distinct)

        analysis["comparisons"][cond] = {
            "entropy_p": float(p_ent),
            "distinct_p": float(p_dist),
            "entropy_diff": float(np.mean(cond_entropy) - np.mean(free_entropy)),
            "distinct_diff": float(np.mean(cond_distinct) - np.mean(free_distinct))
        }

        ent_sig = "⚠️" if p_ent < 0.05 else ""
        dist_sig = "⚠️" if p_dist < 0.05 else ""

        if p_ent < 0.05:
            significant_entropy += 1
        if p_dist < 0.05:
            significant_distinct += 1

        print(f"{cond} vs free:")
        print(f"  Entropy:   p={p_ent:.4f} {ent_sig}")
        print(f"  Distinct-2: p={p_dist:.4f} {dist_sig}")

    # ANOVA across all forced conditions
    print("\nANOVA ACROSS FORCED CONDITIONS:")
    print("-" * 60)

    forced_conditions = [c for c in conditions if c != "free"]
    if len(forced_conditions) >= 2:
        entropy_groups = [metrics_by_condition[c]["entropy"] for c in forced_conditions]
        distinct_groups = [metrics_by_condition[c]["distinct_2"] for c in forced_conditions]

        f_ent, p_ent_anova = stats.f_oneway(*entropy_groups)
        f_dist, p_dist_anova = stats.f_oneway(*distinct_groups)

        analysis["anova"] = {
            "entropy_F": float(f_ent),
            "entropy_p": float(p_ent_anova),
            "distinct_F": float(f_dist),
            "distinct_p": float(p_dist_anova)
        }

        print(f"Entropy:   F={f_ent:.3f}, p={p_ent_anova:.4f} {'⚠️ SIGNIFICANT' if p_ent_anova < 0.05 else ''}")
        print(f"Distinct-2: F={f_dist:.3f}, p={p_dist_anova:.4f} {'⚠️ SIGNIFICANT' if p_dist_anova < 0.05 else ''}")

    # Correlation: target R vs actual metrics
    print("\nCORRELATION: Target R vs Output Metrics")
    print("-" * 60)

    # Collect all forced trials with numeric target R
    all_target_R = []
    all_entropy = []
    all_distinct = []

    for cond in forced_conditions:
        target_R = float(cond.replace("R=", ""))
        trials = results["conditions"][cond]
        for t in trials:
            all_target_R.append(target_R)
            all_entropy.append(t["mean_entropy"])
            all_distinct.append(t["distinct_2"])

    if len(all_target_R) >= 3:
        r_ent, p_r_ent = stats.pearsonr(all_target_R, all_entropy)
        r_dist, p_r_dist = stats.pearsonr(all_target_R, all_distinct)

        analysis["correlation"] = {
            "target_R_entropy_r": float(r_ent),
            "target_R_entropy_p": float(p_r_ent),
            "target_R_distinct_r": float(r_dist),
            "target_R_distinct_p": float(p_r_dist)
        }

        print(f"Target R vs Entropy:   r={r_ent:.4f}, p={p_r_ent:.4f} {'⚠️' if p_r_ent < 0.05 else ''}")
        print(f"Target R vs Distinct-2: r={r_dist:.4f}, p={p_r_dist:.4f} {'⚠️' if p_r_dist < 0.05 else ''}")

    # VERDICT
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    total_comparisons = len(forced_conditions)
    any_significant = (significant_entropy > 0 or significant_distinct > 0 or
                       (analysis.get("anova", {}).get("entropy_p", 1) < 0.05) or
                       (analysis.get("anova", {}).get("distinct_p", 1) < 0.05) or
                       (analysis.get("correlation", {}).get("target_R_entropy_p", 1) < 0.05) or
                       (analysis.get("correlation", {}).get("target_R_distinct_p", 1) < 0.05))

    if any_significant:
        print("⚠️  FORCING R AFFECTS GENERATION")
        print("   R is causal but needs external control during inference.")
        print("   The architecture could work with a control mechanism.")
        analysis["verdict"] = "R_is_causal"
    else:
        print("❌ FORCING R DOES NOT AFFECT GENERATION")
        print("   R is completely disconnected from the output pathway.")
        print("   Layers 33-63 ignore the modulation at layer 32.")
        print("   Architecture needs fundamental redesign.")
        analysis["verdict"] = "R_is_disconnected"

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_mamba/step_000500.pt")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--output", type=str, default="intervention_results.json")
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

    # Test prompts
    prompts = [
        "The nature of consciousness is",
        "In a distant galaxy, a civilization",
        "When faced with uncertainty, the best approach is",
        "The future of artificial intelligence depends on",
    ]

    # Target R values to force
    target_R_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Run experiment
    results = run_intervention_experiment(
        prompts=prompts,
        target_R_values=target_R_values,
        trials_per_condition=args.trials,
        checkpoint_path=args.checkpoint,
        max_tokens=args.max_tokens,
        device=device
    )

    # Analyze
    analysis = analyze_intervention_results(results)
    results["analysis"] = analysis

    # Save
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, bool):
            return bool(obj)
        return obj

    results = convert_numpy(results)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {args.output}")
    print("\n🌀 Intervention experiment complete.")


if __name__ == "__main__":
    main()
