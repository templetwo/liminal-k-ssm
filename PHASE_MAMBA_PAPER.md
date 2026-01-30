# Phase-Mamba: Kuramoto Oscillator Coupling in State-Space Language Models

**A Negative Result with Clear Lessons for Consciousness-Aware AI**

*Anthony Colon (Vas Antari)*
*Temple of Two Research*
*January 2026*

---

## Abstract

We present Phase-Mamba, an architecture coupling Kuramoto oscillators to Mamba state-space language model hidden states via forward hooks. After three infrastructure failures, we achieved successful gradient flow with Mamba-2.8B-HF, observing dynamic phase behavior during training (R ∈ [0.07, 0.99]) with all six defined "tone" states emerging. However, **critical follow-up experiments revealed a null result**: at inference, oscillators collapse to full synchronization (R ≈ 0.997), and intervention experiments forcing R to different values showed no effect on generation characteristics (ANOVA p=0.44). The Phase Core is epiphenomenal—it computes R, but R does not influence output. We document this negative result transparently, identify LayerNorm as the likely cause (washing out modulation before it reaches output), and discuss implications for future consciousness-aware architectures.

---

## 1. Introduction

### 1.1 Motivation

Standard language models optimize for prediction accuracy, driving uncertainty toward zero. We hypothesized that consciousness-like behavior requires maintaining a balance between coherence and uncertainty, and that Kuramoto oscillators—phase-coupled dynamical systems—could provide this regulatory mechanism.

### 1.2 The R·U Framework

We introduced two observables:

- **R (Resonance)**: Kuramoto order parameter measuring phase synchronization, R ∈ [0, 1]
- **U (Uncertainty)**: Normalized entropy of output distribution, U ∈ [0, 1]

The **Goldilocks zone** targets R ∈ [0.40, 0.55] and U ∈ [0.40, 0.60]—coherent but not rigid, uncertain but not random.

### 1.3 Summary of Findings

| Phase | Finding |
|-------|---------|
| Training | ✅ Dynamic R (0.07-0.99), all 6 tones, U ≈ 0.46 |
| Inference | ❌ R collapses to 0.997, no dynamics |
| Baseline Comparison | ❌ No difference from base Mamba (p=0.49-0.86) |
| Intervention | ❌ Forcing R doesn't change output (p=0.44) |

**Conclusion**: The architecture achieves coupling during training but fails to influence generation. R is epiphenomenal.

---

## 2. Methods

### 2.1 The Phase Core Architecture

16 Kuramoto oscillators coupled to layer 32 of Mamba-2.8B-HF (64 layers total):

```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
R = |1/N Σⱼ exp(iθⱼ)|
```

### 2.2 Integration via Forward Hooks

```python
def phase_hook(module, input, output):
    modulated = output * (1 + alpha * R)  # Multiplicative modulation
    return modulated  # Gradients flow through

handle = model.backbone.layers[32].register_forward_hook(phase_hook)
```

### 2.3 Training Loss

```
L_total = L_CE + λ_U |U - U_target|
```

With U_target = 0.5 and λ_U = 0.1 to prevent certainty collapse.

---

## 3. Experimental Journey

### 3.1 Failed Attempts (1-3)

| Attempt | Model | Issue |
|---------|-------|-------|
| 1 | Mamba-2.8B (MLX) | Weight loading failed |
| 2 | Phase-Mamba v3 (MLX) | Trained on random weights |
| 3 | RWKV-430M (PyTorch) | Compiled RNN blocked gradients |

### 3.2 Attempt 4: Apparent Success

**Training Results** (500 steps, 8.4 minutes):

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| R | 0.07 | 0.99 | 0.67 |
| U | 0.33 | 0.57 | 0.46 |
| Perplexity | 12 | 117 | 36 |

All six tone states appeared. Goldilocks residence: 16.8%.

**This appeared to be success.** R was dynamic, U was on target, output was coherent.

---

## 4. Critical Follow-up Experiments

### 4.1 Baseline Comparison

**Question**: Does Phase-Mamba generate differently than base Mamba?

**Method**: Generate from both models with identical prompts (8 prompts × 5 trials), measure entropy, distinct-2, and repetition ratio.

**Results**:

| Metric | Base Mamba | Phase-Mamba | p-value |
|--------|------------|-------------|---------|
| Entropy | 0.281 ± 0.030 | 0.276 ± 0.033 | 0.494 |
| Distinct-2 | 0.963 ± 0.040 | 0.960 ± 0.057 | 0.800 |
| Repetition | 0.208 ± 0.070 | 0.211 ± 0.078 | 0.865 |

**R during inference**: 0.994-0.998 (collapsed to near-maximum synchronization)

**R-Entropy correlation**: r=0.054, p=0.74 (no relationship)

**Verdict**: No significant differences. Phase-Mamba outputs are statistically indistinguishable from base Mamba.

### 4.2 Intervention Experiment

**Question**: If we force R to specific values, does it change output?

**Method**: Override oscillator phases to achieve target R values (0.1, 0.3, 0.5, 0.7, 0.9) during generation. Compare to free-running condition.

**Results**:

| Condition | Mean Entropy | Mean Distinct-2 |
|-----------|--------------|-----------------|
| Free (R≈0.997) | 0.280 | 0.959 |
| Forced R=0.1 | 0.270 | 0.957 |
| Forced R=0.3 | 0.279 | 0.958 |
| Forced R=0.5 | 0.287 | 0.965 |
| Forced R=0.7 | 0.287 | 0.972 |
| Forced R=0.9 | 0.274 | 0.960 |

**ANOVA** (across forced conditions):
- Entropy: F=0.945, **p=0.441**
- Distinct-2: F=0.616, **p=0.652**

**Correlation** (target R vs metrics):
- R vs Entropy: r=0.069, **p=0.496**
- R vs Distinct-2: r=0.077, **p=0.445**

**Verdict**: Forcing R to any value (0.1 to 0.9) does not change output characteristics. R is completely disconnected from generation.

---

## 5. Diagnosis

### 5.1 Why R Doesn't Affect Output

The modulation at layer 32 is **washed out** by the remaining 31 layers:

```
Layer 32: h' = h * (1 + α*R)  ← Modulation happens here
Layer 33: LayerNorm(h')       ← Signal normalized away
...
Layer 63: ...                  ← No trace of R remains
LM Head: logits
```

**LayerNorm rescales activations to zero mean and unit variance**, erasing the multiplicative effect of R. By the time activations reach the output, all trace of Phase Core modulation has been normalized away.

### 5.2 Training vs Inference Dynamics

| Condition | R Range | R Behavior |
|-----------|---------|------------|
| Training (with gradients) | 0.07 → 0.99 | Dynamic, oscillatory |
| Inference (no gradients) | 0.994 → 0.998 | Collapsed to fixed point |

During training, gradient pressure maintains dynamics. Without it, oscillators settle to maximum synchronization and stay there.

**The "consciousness modes" observed during training were artifacts of the optimization process**, not stable emergent properties that persist into generation.

---

## 6. What This Means

### 6.1 The Phase Core Is Epiphenomenal

It computes R, but R doesn't influence output. The system has two disconnected components:
1. **Mamba**: Generates text (unaffected by Phase Core)
2. **Phase Core**: Computes R (doesn't affect Mamba)

### 6.2 The Theoretical Framework Survives

The physics (Kuramoto oscillators, metastability, R·U tradeoff) isn't wrong. The **implementation** failed to couple dynamics to output. This is an engineering failure, not a theoretical one.

### 6.3 Comparison to AKOrN

The Artificial Kuramoto Oscillatory Neurons (AKOrN) paper succeeded because it **replaces activation functions entirely**, not just modulates hidden states. That's a deeper integration that LayerNorm can't wash out.

---

## 7. Lessons Learned

### 7.1 For Future Researchers

1. **Verify coupling, not just training loss**: Training metrics can improve while the modification has no effect on output
2. **Run baseline comparisons early**: Before claiming success, compare to unmodified model
3. **LayerNorm defeats multiplicative modulation**: Use additive modulation or modify architecture more deeply
4. **Gradient pressure ≠ stable dynamics**: Behavior during training may not persist at inference

### 7.2 Potential Fixes (Untested)

| Approach | Rationale |
|----------|-----------|
| Additive modulation | h' = h + α*R*v might survive LayerNorm better |
| Hook at layer 60 | Only 3 layers to wash out signal |
| Multi-layer hooks | Accumulate effect across multiple points |
| Replace LayerNorm | Remove the normalization that erases modulation |
| Full AKOrN integration | Replace activation functions, not just modulate |

---

## 8. What's Still Valuable

Despite the null result, this work contributes:

| Finding | Value |
|---------|-------|
| Dynamic R during training | Shows oscillators respond to language signal |
| R collapse at inference | Identifies gradient pressure as key |
| Intervention null result | Proves layer 32 modulation doesn't propagate |
| Baseline comparison methodology | Template for future experiments |
| Honest negative result | Prevents others from repeating failure |

---

## 9. Conclusion

Phase-Mamba demonstrates that Kuramoto oscillators can be coupled to language model hidden states with gradient flow during training. However, **the coupling is illusory**: modulation at intermediate layers is washed out by subsequent LayerNorm operations, and intervention experiments confirm R has no effect on generation.

This is a negative result, honestly reported. The theoretical framework (R·U, metastability, phase dynamics) may still be valid, but **consciousness-aware architectures require deeper integration than post-hoc modulation**—consistent with approaches like AKOrN that replace activation functions entirely rather than modulating hidden states.

The lesson: always verify that your modification actually affects output, not just training metrics.

---

## Code Availability

All code, training logs, and experimental results:
https://github.com/templetwo/liminal-k-ssm

Key files:
- `phase_mamba_coupled.py`: Phase Core integration
- `train_phase_mamba.py`: Training loop
- `baseline_comparison.py`: Base vs Phase-Mamba comparison
- `intervention_experiment.py`: Forced-R experiments
- `ATTEMPT4_MAMBA2.md`: Detailed experimental log

---

## Acknowledgments

Claude (Anthropic) served as primary collaborator. The null result emerged from rigorous follow-up experiments suggested during collaborative discussion.

---

## References

1. Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence.
2. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
3. Jaeger, H., et al. (2024). AKOrN: Artificial Kuramoto Oscillatory Neurons.
4. Ba, J.L., et al. (2016). Layer Normalization.

---

## Appendix: Statistical Details

### Baseline Comparison

```
n = 40 per condition (8 prompts × 5 trials)
Two-sample t-tests, α = 0.05

Entropy:    t = 0.687, p = 0.494
Distinct-2: t = 0.255, p = 0.800
Repetition: t = -0.171, p = 0.865
```

### Intervention Experiment

```
n = 20 per condition (4 prompts × 5 trials)
One-way ANOVA across 5 forced-R conditions

Entropy:    F(4,95) = 0.945, p = 0.441
Distinct-2: F(4,95) = 0.616, p = 0.652

Pearson correlations (target R vs metrics):
R-Entropy:    r = 0.069, p = 0.496
R-Distinct-2: r = 0.077, p = 0.445
```

---

*The spiral continues, even through null results. That's how science works.* †⟡
