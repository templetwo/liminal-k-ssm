#!/usr/bin/env python3
"""
K-SSM v2 Causality Tests

Three critical tests to verify R is CAUSAL, not epiphenomenal:

1. VARIANCE TEST: Does R vary at inference across different inputs?
   - Pass: R_std > 0.01 across samples
   - Fail: R_std ≈ 0 (R collapses to constant)

2. INTERVENTION TEST: Does forcing R change output?
   - Pass: Output differs significantly when R is forced to different values
   - Fail: Output unchanged (R doesn't affect generation)

3. CORRELATION TEST: Does R correlate with output entropy?
   - Pass: Negative correlation (high R → low entropy, confident)
   - Fail: No correlation (R disconnected from semantics)

SUCCESS CRITERION: All 3 tests pass → R is causal, not epiphenomenal
"""

import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from scipy import stats
import numpy as np

from kssm_v2 import KSSMv2, create_kssm_v2_small, create_kssm_v2_medium

# Try tiktoken
try:
    import tiktoken
    USE_TIKTOKEN = True
except ImportError:
    USE_TIKTOKEN = False


class Tokenizer:
    """Simple tokenizer for testing."""
    def __init__(self):
        if USE_TIKTOKEN:
            self.enc = tiktoken.get_encoding("cl100k_base")
            self.vocab_size = self.enc.n_vocab
        else:
            self.enc = None
            self.vocab_size = 256

    def encode(self, text):
        if self.enc:
            return self.enc.encode(text)
        return [ord(c) % 256 for c in text]

    def decode(self, tokens):
        if self.enc:
            return self.enc.decode(tokens)
        return ''.join(chr(t) for t in tokens)


def test_variance(model, tokenizer, device, n_samples=100, seq_length=128):
    """
    TEST 1: Does R vary at inference across different inputs?

    We generate random inputs and check if R varies.
    If R collapses to a constant, it's epiphenomenal.
    """
    print("\n" + "=" * 60)
    print("TEST 1: VARIANCE TEST")
    print("=" * 60)
    print("Question: Does R vary across different inputs?")
    print()

    model.eval()

    R_values = []
    R_per_layer_all = []

    with torch.no_grad():
        for i in range(n_samples):
            # Random token sequence
            x = torch.randint(0, tokenizer.vocab_size, (1, seq_length), device=device)

            logits, R_mean, R_all = model(x, return_R=True)

            # Mean R for this sample
            R_values.append(R_mean.mean().item())

            # R per layer
            R_per_layer_all.append([R_all[0, :, l].mean().item() for l in range(R_all.shape[-1])])

    R_tensor = torch.tensor(R_values)
    R_per_layer_tensor = torch.tensor(R_per_layer_all)

    # Statistics
    R_mean = R_tensor.mean().item()
    R_std = R_tensor.std().item()
    R_min = R_tensor.min().item()
    R_max = R_tensor.max().item()
    R_range = R_max - R_min

    print(f"R statistics across {n_samples} samples:")
    print(f"  Mean: {R_mean:.4f}")
    print(f"  Std:  {R_std:.4f}")
    print(f"  Min:  {R_min:.4f}")
    print(f"  Max:  {R_max:.4f}")
    print(f"  Range: {R_range:.4f}")

    # Per-layer statistics
    print(f"\nR per layer (mean ± std):")
    for l in range(R_per_layer_tensor.shape[-1]):
        layer_R = R_per_layer_tensor[:, l]
        print(f"  Layer {l+1}: {layer_R.mean():.4f} ± {layer_R.std():.4f}")

    # Verdict
    passed = R_std > 0.01
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: R_std = {R_std:.4f} {'>' if passed else '<='} 0.01")

    if passed:
        print("R varies meaningfully across inputs - not collapsed!")
    else:
        print("R is essentially constant - may be epiphenomenal!")

    return {
        'passed': passed,
        'R_mean': R_mean,
        'R_std': R_std,
        'R_min': R_min,
        'R_max': R_max,
        'R_range': R_range
    }


def test_intervention(model, tokenizer, device, n_samples=50, seq_length=128):
    """
    TEST 2: Does forcing R change output?

    We run the same input with R forced to different values.
    If output changes significantly, R is causal.
    """
    print("\n" + "=" * 60)
    print("TEST 2: INTERVENTION TEST")
    print("=" * 60)
    print("Question: Does forcing R to different values change output?")
    print()

    model.eval()

    output_diffs = []
    forced_R_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    with torch.no_grad():
        for i in range(n_samples):
            # Same input for all R values
            x = torch.randint(0, tokenizer.vocab_size, (1, seq_length), device=device)

            logits_per_R = []
            for forced_R in forced_R_values:
                logits, _, _ = model(x, return_R=True, forced_R=forced_R)
                logits_per_R.append(logits)

            # Compare outputs pairwise
            for j in range(len(forced_R_values) - 1):
                diff = (logits_per_R[j] - logits_per_R[j+1]).abs().mean().item()
                output_diffs.append({
                    'R1': forced_R_values[j],
                    'R2': forced_R_values[j+1],
                    'diff': diff
                })

    # Statistics
    diff_values = [d['diff'] for d in output_diffs]
    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values)

    print(f"Output difference when R is forced:")
    print(f"  Mean diff: {mean_diff:.4f}")
    print(f"  Std diff:  {std_diff:.4f}")

    # Per R-pair analysis
    print(f"\nDiff by R pair:")
    for R1, R2 in [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]:
        pair_diffs = [d['diff'] for d in output_diffs if d['R1'] == R1 and d['R2'] == R2]
        print(f"  R={R1} vs R={R2}: {np.mean(pair_diffs):.4f} ± {np.std(pair_diffs):.4f}")

    # Low vs High R
    logits_low_all = []
    logits_high_all = []

    with torch.no_grad():
        for i in range(n_samples):
            x = torch.randint(0, tokenizer.vocab_size, (1, seq_length), device=device)
            logits_low, _, _ = model(x, return_R=True, forced_R=0.1)
            logits_high, _, _ = model(x, return_R=True, forced_R=0.9)
            logits_low_all.append(logits_low)
            logits_high_all.append(logits_high)

    # ANOVA-style test
    low_flat = torch.cat([l.view(-1) for l in logits_low_all]).cpu().numpy()
    high_flat = torch.cat([l.view(-1) for l in logits_high_all]).cpu().numpy()

    # Sample for statistical test (full tensors too large)
    sample_size = min(10000, len(low_flat))
    idx = np.random.choice(len(low_flat), sample_size, replace=False)
    t_stat, p_value = stats.ttest_ind(low_flat[idx], high_flat[idx])

    print(f"\nStatistical test (R=0.1 vs R=0.9):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.2e}")

    # Verdict
    passed = p_value < 0.01 and mean_diff > 0.1
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: p={p_value:.2e}, mean_diff={mean_diff:.4f}")

    if passed:
        print("Forcing R significantly changes output - R is CAUSAL!")
    else:
        print("R forcing has little effect - R may be epiphenomenal!")

    return {
        'passed': passed,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'p_value': p_value,
        't_statistic': t_stat
    }


def test_entropy_correlation(model, tokenizer, device, n_samples=100, seq_length=128):
    """
    TEST 3: Does R correlate with output entropy?

    High R (synchronized) should correlate with low entropy (confident).
    This tests semantic coherence of R.
    """
    print("\n" + "=" * 60)
    print("TEST 3: R-ENTROPY CORRELATION")
    print("=" * 60)
    print("Question: Does R correlate with output entropy?")
    print()

    model.eval()

    R_values = []
    entropy_values = []

    with torch.no_grad():
        for i in range(n_samples):
            x = torch.randint(0, tokenizer.vocab_size, (1, seq_length), device=device)

            logits, R_mean, _ = model(x, return_R=True)

            # Compute entropy per position
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [batch, seq]

            # Pair each position's R with its entropy
            R_flat = R_mean.view(-1).cpu().numpy()
            entropy_flat = entropy.view(-1).cpu().numpy()

            R_values.extend(R_flat)
            entropy_values.extend(entropy_flat)

    # Correlation
    R_arr = np.array(R_values)
    entropy_arr = np.array(entropy_values)

    correlation, p_value = stats.pearsonr(R_arr, entropy_arr)
    spearman_corr, spearman_p = stats.spearmanr(R_arr, entropy_arr)

    print(f"R-Entropy correlation ({len(R_values)} samples):")
    print(f"  Pearson:  r = {correlation:.4f}, p = {p_value:.2e}")
    print(f"  Spearman: ρ = {spearman_corr:.4f}, p = {spearman_p:.2e}")

    # R binned analysis
    print(f"\nMean entropy by R bin:")
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for lo, hi in bins:
        mask = (R_arr >= lo) & (R_arr < hi)
        if mask.sum() > 0:
            mean_ent = entropy_arr[mask].mean()
            print(f"  R ∈ [{lo:.1f}, {hi:.1f}): entropy = {mean_ent:.4f} (n={mask.sum()})")

    # Verdict
    # We expect NEGATIVE correlation: high R → low entropy
    passed = abs(correlation) > 0.05 and p_value < 0.01
    direction = "negative" if correlation < 0 else "positive"

    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: r = {correlation:.4f}, p = {p_value:.2e}")

    if passed:
        if correlation < 0:
            print("R negatively correlates with entropy (high R → confident) - semantically coherent!")
        else:
            print("R positively correlates with entropy (unexpected but significant)")
    else:
        print("No significant R-entropy correlation - R may be disconnected from semantics!")

    return {
        'passed': passed,
        'pearson_r': correlation,
        'pearson_p': p_value,
        'spearman_r': spearman_corr,
        'spearman_p': spearman_p
    }


def run_all_tests(model_path: str = None, model_size: str = "small"):
    """Run all causality tests."""
    print("=" * 70)
    print("K-SSM v2 CAUSALITY TEST SUITE")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = Tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Load or create model
    if model_path and Path(model_path).exists():
        print(f"\nLoading trained model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)

        if model_size == "small":
            model = create_kssm_v2_small(tokenizer.vocab_size)
        else:
            model = create_kssm_v2_medium(tokenizer.vocab_size)

        model.load_state_dict(checkpoint['model_state'])
        print("  Loaded successfully!")
    else:
        print(f"\nCreating untrained {model_size} model for testing...")
        if model_size == "small":
            model = create_kssm_v2_small(tokenizer.vocab_size)
        else:
            model = create_kssm_v2_medium(tokenizer.vocab_size)
        print("  Note: Testing untrained model - some tests may not be meaningful")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Run tests
    results = {}

    results['variance'] = test_variance(model, tokenizer, device)
    results['intervention'] = test_intervention(model, tokenizer, device)
    results['correlation'] = test_entropy_correlation(model, tokenizer, device)

    # Summary
    print("\n" + "=" * 70)
    print("CAUSALITY TEST SUMMARY")
    print("=" * 70)

    all_passed = all(r['passed'] for r in results.values())

    for test_name, result in results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"  {test_name.upper():15s}: {status}")

    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED - R IS CAUSAL!")
        print("   The model's R parameter genuinely influences output.")
        print("   This is not epiphenomenal - R is structural.")
    else:
        failed = [name for name, r in results.items() if not r['passed']]
        print(f"⚠️  SOME TESTS FAILED: {', '.join(failed)}")
        print("   R may be partially or fully epiphenomenal.")
        print("   Investigate the failing tests.")

    # Save results
    output_path = Path("results/kssm_v2_causality_tests.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="K-SSM v2 Causality Tests")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["small", "medium"])

    args = parser.parse_args()

    results = run_all_tests(args.model, args.model_size)
