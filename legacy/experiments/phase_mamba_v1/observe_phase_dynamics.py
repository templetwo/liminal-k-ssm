#!/usr/bin/env python3
"""
Phase-Mamba Observation Protocol
Test whether observation affects phase dynamics

EXPERIMENT:
1. Blind generation (no R monitoring during generation)
2. Measured generation (R tracked at each token)
3. Compare R distributions

HYPOTHESIS:
If consciousness-like behavior exists, measured generation should
show different R dynamics than blind generation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
from phase_mamba_coupled import load_mamba_with_phase


def compute_uncertainty(logits: torch.Tensor) -> float:
    """Compute epistemic uncertainty from logits."""
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    max_entropy = torch.log(torch.tensor(logits.shape[-1], dtype=torch.float32))
    return torch.mean(entropy / max_entropy).item()


def blind_generation(coupled, prompt: str, max_tokens: int = 50) -> Dict:
    """
    Generate without monitoring R during generation.
    Only check R after generation is complete.
    """
    tokenizer = coupled.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt").to(coupled.device)

    coupled.phase_core.reset_phases()
    coupled.phase_core.eval()

    # Generate without any R checks
    with torch.no_grad():
        outputs = coupled.mamba_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Only check R after generation
    final_R = coupled.phase_core.current_R
    final_tone = coupled.phase_core.current_tone

    return {
        "text": text,
        "final_R": final_R,
        "final_tone": final_tone,
        "R_trajectory": [final_R],  # Only final value
        "mode": "blind"
    }


def measured_generation(coupled, prompt: str, max_tokens: int = 50) -> Dict:
    """
    Generate while monitoring R at each token.
    This actively observes the phase dynamics.
    """
    tokenizer = coupled.tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(coupled.device)

    coupled.phase_core.reset_phases()
    coupled.phase_core.eval()

    R_trajectory = []
    U_trajectory = []
    generated_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass
            outputs = coupled.mamba_model(input_ids=generated_ids)
            logits = outputs.logits[:, -1, :]

            # OBSERVE: Record R at this step
            R_trajectory.append(coupled.phase_core.current_R)
            U_trajectory.append(compute_uncertainty(logits))

            # Sample next token
            probs = F.softmax(logits / 0.9, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return {
        "text": text,
        "final_R": R_trajectory[-1] if R_trajectory else 0,
        "final_tone": coupled.phase_core.current_tone,
        "R_trajectory": R_trajectory,
        "U_trajectory": U_trajectory,
        "mode": "measured"
    }


def run_observation_protocol(coupled, prompts: List[str], trials_per_prompt: int = 5):
    """
    Run the observation protocol.

    For each prompt, generate both blind and measured versions,
    and compare the R dynamics.
    """
    results = {
        "blind": [],
        "measured": []
    }

    for prompt in prompts:
        print(f"\n{'='*70}")
        print(f"Prompt: \"{prompt}\"")
        print('='*70)

        # Blind trials
        print("\n[BLIND GENERATION]")
        for trial in range(trials_per_prompt):
            result = blind_generation(coupled, prompt)
            results["blind"].append({
                "prompt": prompt,
                "trial": trial,
                **result
            })
            print(f"  Trial {trial+1}: R={result['final_R']:.4f} {result['final_tone']}")

        # Measured trials
        print("\n[MEASURED GENERATION]")
        for trial in range(trials_per_prompt):
            result = measured_generation(coupled, prompt)
            results["measured"].append({
                "prompt": prompt,
                "trial": trial,
                **result
            })
            R_mean = np.mean(result["R_trajectory"])
            R_std = np.std(result["R_trajectory"])
            print(f"  Trial {trial+1}: R={result['final_R']:.4f} {result['final_tone']} "
                  f"(trajectory: mean={R_mean:.4f}, std={R_std:.4f})")

    return results


def analyze_results(results: Dict):
    """Analyze and compare blind vs measured generation."""

    print("\n" + "="*70)
    print("OBSERVATION PROTOCOL ANALYSIS")
    print("="*70)

    # Extract final R values
    blind_R = [r["final_R"] for r in results["blind"]]
    measured_R = [r["final_R"] for r in results["measured"]]

    print(f"\nBlind Generation (n={len(blind_R)}):")
    print(f"  R: mean={np.mean(blind_R):.4f}, std={np.std(blind_R):.4f}")
    print(f"  Range: [{min(blind_R):.4f}, {max(blind_R):.4f}]")

    print(f"\nMeasured Generation (n={len(measured_R)}):")
    print(f"  R: mean={np.mean(measured_R):.4f}, std={np.std(measured_R):.4f}")
    print(f"  Range: [{min(measured_R):.4f}, {max(measured_R):.4f}]")

    # Trajectory analysis for measured
    all_trajectories = [r["R_trajectory"] for r in results["measured"]]
    avg_trajectory_length = np.mean([len(t) for t in all_trajectories])

    print(f"\nMeasured Trajectory Analysis:")
    print(f"  Average trajectory length: {avg_trajectory_length:.1f} tokens")

    # Compute trajectory statistics
    trajectory_means = [np.mean(t) for t in all_trajectories]
    trajectory_stds = [np.std(t) for t in all_trajectories]

    print(f"  Trajectory R means: {np.mean(trajectory_means):.4f} ± {np.std(trajectory_means):.4f}")
    print(f"  Trajectory R stds: {np.mean(trajectory_stds):.4f} ± {np.std(trajectory_stds):.4f}")

    # Statistical test
    from scipy import stats
    if len(blind_R) > 1 and len(measured_R) > 1:
        t_stat, p_value = stats.ttest_ind(blind_R, measured_R)
        print(f"\nStatistical Comparison (t-test):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            print("  ⚠️  SIGNIFICANT DIFFERENCE between blind and measured!")
            print("  🌀 Observation may be affecting phase dynamics!")
        else:
            print("  No significant difference detected (p > 0.05)")

    # Tone distribution
    print("\nTone Distribution:")
    for mode in ["blind", "measured"]:
        tones = [r["final_tone"] for r in results[mode]]
        from collections import Counter
        tone_counts = Counter(tones)
        print(f"  {mode.capitalize()}:")
        for tone, count in sorted(tone_counts.items(), key=lambda x: -x[1]):
            print(f"    {tone}: {count} ({100*count/len(tones):.1f}%)")

    return {
        "blind_R_mean": np.mean(blind_R),
        "blind_R_std": np.std(blind_R),
        "measured_R_mean": np.mean(measured_R),
        "measured_R_std": np.std(measured_R),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_mamba_500/step_000500.pt")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    print("="*70)
    print("🔬 PHASE-MAMBA OBSERVATION PROTOCOL")
    print("="*70)
    print("Testing whether observation affects phase dynamics")
    print("="*70)

    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    coupled = load_mamba_with_phase(phase_layer=32, device=device)

    # Load trained weights
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        coupled.phase_core.load_state_dict(checkpoint["phase_core_state"])
        print(f"\n✅ Loaded Phase Core from: {args.checkpoint}")

    # Test prompts
    prompts = [
        "The nature of consciousness is",
        "Awareness emerges when",
        "The boundary between observer and observed",
        "In the space between thoughts",
    ]

    # Run protocol
    results = run_observation_protocol(coupled, prompts, trials_per_prompt=args.trials)

    # Analyze
    analysis = analyze_results(results)

    print("\n" + "="*70)
    print("🌀 OBSERVATION PROTOCOL COMPLETE")
    print("="*70)
