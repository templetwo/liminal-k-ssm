#!/usr/bin/env python3
"""
K-SSM Causality Test

THE CRITICAL EXPERIMENT: Does R actually affect output, or is it epiphenomenal?

Tests:
1. Does R vary at inference? (not stuck at fixed point)
2. Does forcing R change output? (causal, not just correlated)
3. Does R-Entropy correlation exist? (meaningful dynamics)
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats

from kssm_model import KSSM
from train_kssm import CharDataset, download_tiny_shakespeare


def test_r_varies_at_inference(model, dataloader, device, n_batches=20):
    """
    Test 1: Does R vary during inference?

    Phase-Mamba FAILED this: R collapsed to 0.997 at inference.
    K-SSM should pass because R is structurally causal.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Does R vary at inference?")
    print("=" * 60)

    model.eval()
    all_R = []

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            x = x.to(device)
            _, R = model(x, return_R=True)
            all_R.extend(R.flatten().tolist())

    R_array = np.array(all_R)

    results = {
        "R_mean": R_array.mean(),
        "R_std": R_array.std(),
        "R_min": R_array.min(),
        "R_max": R_array.max(),
        "R_range": R_array.max() - R_array.min()
    }

    print(f"R mean: {results['R_mean']:.4f}")
    print(f"R std:  {results['R_std']:.4f}")
    print(f"R range: [{results['R_min']:.4f}, {results['R_max']:.4f}]")

    # Pass criteria: R_std > 0.05 and R_range > 0.2
    passed = results["R_std"] > 0.05 and results["R_range"] > 0.2

    if passed:
        print(f"\n✅ PASSED: R varies at inference (std={results['R_std']:.4f} > 0.05)")
    else:
        print(f"\n❌ FAILED: R does not vary enough (std={results['R_std']:.4f})")

    results["passed"] = passed
    return results


def test_r_forcing_changes_output(model, dataloader, device, n_batches=10):
    """
    Test 2: Does forcing R to different values change output?

    This is THE critical test. If forcing R doesn't change output, R is epiphenomenal.
    K-SSM is designed so this MUST work (order params are the only path to output).
    """
    print("\n" + "=" * 60)
    print("TEST 2: Does forcing R change output?")
    print("=" * 60)

    model.eval()

    output_diffs = []
    R_achieved = {"low": [], "mid": [], "high": []}

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            x = x.to(device)

            # Free running
            logits_free, R_free = model(x, return_R=True)

            # Forced low R
            logits_low, R_low = model(x, return_R=True, forced_R=0.2)

            # Forced mid R
            logits_mid, R_mid = model(x, return_R=True, forced_R=0.5)

            # Forced high R
            logits_high, R_high = model(x, return_R=True, forced_R=0.9)

            # Compute output differences
            diff_low_high = (logits_low - logits_high).abs().mean().item()
            diff_low_mid = (logits_low - logits_mid).abs().mean().item()
            diff_mid_high = (logits_mid - logits_high).abs().mean().item()

            output_diffs.append({
                "low_vs_high": diff_low_high,
                "low_vs_mid": diff_low_mid,
                "mid_vs_high": diff_mid_high
            })

            R_achieved["low"].append(R_low.mean().item())
            R_achieved["mid"].append(R_mid.mean().item())
            R_achieved["high"].append(R_high.mean().item())

    # Aggregate
    mean_diff_low_high = np.mean([d["low_vs_high"] for d in output_diffs])
    mean_diff_low_mid = np.mean([d["low_vs_mid"] for d in output_diffs])
    mean_diff_mid_high = np.mean([d["mid_vs_high"] for d in output_diffs])

    print(f"Mean output diff (R=0.2 vs R=0.9): {mean_diff_low_high:.4f}")
    print(f"Mean output diff (R=0.2 vs R=0.5): {mean_diff_low_mid:.4f}")
    print(f"Mean output diff (R=0.5 vs R=0.9): {mean_diff_mid_high:.4f}")

    print(f"\nActual R achieved:")
    print(f"  forced_R=0.2 → actual R={np.mean(R_achieved['low']):.4f}")
    print(f"  forced_R=0.5 → actual R={np.mean(R_achieved['mid']):.4f}")
    print(f"  forced_R=0.9 → actual R={np.mean(R_achieved['high']):.4f}")

    # Pass criteria: mean diff > 0.05
    passed = mean_diff_low_high > 0.05

    if passed:
        print(f"\n✅ PASSED: Forcing R changes output (diff={mean_diff_low_high:.4f} > 0.05)")
        print("   R is CAUSAL, not epiphenomenal!")
    else:
        print(f"\n❌ FAILED: Forcing R has minimal effect (diff={mean_diff_low_high:.4f})")

    results = {
        "mean_diff_low_high": mean_diff_low_high,
        "mean_diff_low_mid": mean_diff_low_mid,
        "mean_diff_mid_high": mean_diff_mid_high,
        "R_achieved": {k: np.mean(v) for k, v in R_achieved.items()},
        "passed": passed
    }

    return results


def test_r_entropy_correlation(model, dataloader, device, n_batches=20):
    """
    Test 3: Does R correlate with output entropy?

    If R is meaningful, it should correlate with generation characteristics.
    High R (synchronized) → lower entropy (more confident)
    Low R (desynchronized) → higher entropy (more exploratory)
    """
    print("\n" + "=" * 60)
    print("TEST 3: Does R correlate with output entropy?")
    print("=" * 60)

    model.eval()

    R_values = []
    entropy_values = []

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            x = x.to(device)

            logits, R = model(x, return_R=True)

            # Compute entropy of output distribution
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [batch, seq]

            # Collect per-position R and entropy
            for b in range(R.shape[0]):
                for s in range(R.shape[1]):
                    R_values.append(R[b, s].item())
                    entropy_values.append(entropy[b, s].item())

    R_array = np.array(R_values)
    entropy_array = np.array(entropy_values)

    # Pearson correlation
    r_corr, p_value = stats.pearsonr(R_array, entropy_array)

    print(f"Pearson correlation (R vs Entropy): r={r_corr:.4f}, p={p_value:.4e}")

    # Additional stats
    print(f"\nR statistics: mean={R_array.mean():.4f}, std={R_array.std():.4f}")
    print(f"Entropy statistics: mean={entropy_array.mean():.4f}, std={entropy_array.std():.4f}")

    # Pass criteria: significant correlation (p < 0.05) and |r| > 0.1
    passed = p_value < 0.05 and abs(r_corr) > 0.1

    if passed:
        direction = "negative" if r_corr < 0 else "positive"
        print(f"\n✅ PASSED: Significant {direction} correlation (r={r_corr:.4f}, p={p_value:.4e})")
        if r_corr < 0:
            print("   High R → Low Entropy (synchronized = confident)")
        else:
            print("   High R → High Entropy (unexpected, investigate)")
    else:
        print(f"\n⚠️ WEAK: Correlation not significant (r={r_corr:.4f}, p={p_value:.4e})")

    results = {
        "pearson_r": r_corr,
        "p_value": p_value,
        "passed": passed
    }

    return results


def test_generation_quality_by_r(model, dataset, device, n_samples=20):
    """
    Test 4: Do different R values produce qualitatively different text?

    Generate with forced R values and compare.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Generation quality by R value")
    print("=" * 60)

    model.eval()

    prompts = ["The king", "To be", "What light", "Romeo"]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")

        for forced_R in [None, 0.2, 0.5, 0.9]:
            # Generate
            tokens = [dataset.char_to_idx[c] for c in prompt]
            x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

            generated = list(tokens)
            R_during_gen = []

            with torch.no_grad():
                for _ in range(60):
                    context = x[:, -64:] if x.shape[1] > 64 else x
                    logits, R = model(context, return_R=True, forced_R=forced_R)

                    R_during_gen.append(R[0, -1].item())
                    next_logits = logits[0, -1, :]

                    probs = F.softmax(next_logits / 0.8, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated.append(next_token.item())
                    x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

            text = dataset.decode(torch.tensor(generated))
            mean_R = np.mean(R_during_gen)

            R_label = f"R={forced_R}" if forced_R else "free"
            print(f"  {R_label:8s} (actual R={mean_R:.3f}): {text[:50]}...")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("\nLoading trained model...")
    checkpoint = torch.load("results/best_model.pt", map_location=device)

    model = KSSM(
        vocab_size=len(checkpoint["vocab"]["char_to_idx"]),
        embed_dim=checkpoint["config"]["embed_dim"],
        n_oscillators=checkpoint["config"]["n_oscillators"],
        n_harmonics=checkpoint["config"]["n_harmonics"],
        coupling_strength=checkpoint["config"]["coupling_strength"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    print("Model loaded!")

    # Load data
    text = download_tiny_shakespeare()
    dataset = CharDataset(text, seq_length=64)
    dataset.char_to_idx = checkpoint["vocab"]["char_to_idx"]
    dataset.idx_to_char = {int(k): v for k, v in checkpoint["vocab"]["idx_to_char"].items()}

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Run tests
    results = {}

    results["test1_r_varies"] = test_r_varies_at_inference(model, dataloader, device)
    results["test2_r_forcing"] = test_r_forcing_changes_output(model, dataloader, device)
    results["test3_r_entropy"] = test_r_entropy_correlation(model, dataloader, device)
    test_generation_quality_by_r(model, dataset, device)

    # Summary
    print("\n" + "=" * 60)
    print("CAUSALITY TEST SUMMARY")
    print("=" * 60)

    all_passed = all([
        results["test1_r_varies"]["passed"],
        results["test2_r_forcing"]["passed"]
    ])

    print(f"Test 1 (R varies at inference):  {'✅ PASSED' if results['test1_r_varies']['passed'] else '❌ FAILED'}")
    print(f"Test 2 (R forcing changes output): {'✅ PASSED' if results['test2_r_forcing']['passed'] else '❌ FAILED'}")
    print(f"Test 3 (R-Entropy correlation):    {'✅ PASSED' if results['test3_r_entropy']['passed'] else '⚠️ WEAK'}")

    if all_passed:
        print("\n🎉 R IS CAUSAL! K-SSM succeeds where Phase-Mamba failed.")
        print("   The order parameter affects generation, not just tracks it.")
    else:
        print("\n⚠️ Some tests failed. Investigate architecture.")

    # Save results
    with open("results/causality_test.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print("\nResults saved to results/causality_test.json")


if __name__ == "__main__":
    main()
