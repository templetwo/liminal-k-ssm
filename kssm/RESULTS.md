# K-SSM Results: R is Causal

## The Critical Question

Does the Kuramoto order parameter R **actually affect generation**, or is it epiphenomenal (just tracking internal states without influencing output)?

**Phase-Mamba answer**: R was epiphenomenal. Forcing R had no effect on output.

**K-SSM answer**: R is **CAUSAL**. Forcing R dramatically changes output.

---

## Training Results

| Metric | Value |
|--------|-------|
| Model | K-SSM (Kuramoto State-Space Model) |
| Parameters | ~100K |
| Dataset | TinyShakespeare (character-level) |
| Epochs | 10 |
| Final train loss | 2.45 |
| R range during training | [0.09, 0.22] |
| Tone | ☾ Intimacy (R ≈ 0.15) |

---

## Causality Tests

### Test 1: Does R vary at inference?

| Metric | Value | Pass Threshold |
|--------|-------|----------------|
| R mean | 0.147 | - |
| R std | 0.077 | > 0.05 |
| R range | [0.002, 0.378] | > 0.2 |

**Result**: ✅ **PASSED**

Unlike Phase-Mamba where R collapsed to 0.997, K-SSM maintains dynamic R.

### Test 2: Does forcing R change output?

| Comparison | Output Difference | Pass Threshold |
|------------|-------------------|----------------|
| R=0.2 vs R=0.9 | **5.93** | > 0.05 |
| R=0.2 vs R=0.5 | 2.40 | > 0.05 |
| R=0.5 vs R=0.9 | 6.26 | > 0.05 |

**Result**: ✅ **PASSED**

Forcing R to different values produces **dramatically different outputs**.

### Test 3: R-Entropy Correlation

| Metric | Value |
|--------|-------|
| Pearson r | -0.099 |
| p-value | 1.96e-89 |
| Direction | Negative (correct) |

**Result**: ⚠️ Weak but in correct direction

High R correlates with lower entropy (more synchronized → more confident).

---

## Generation Examples

### Prompt: "The king"

| Condition | Actual R | Output |
|-----------|----------|--------|
| Free | 0.154 | "The kinghat t ss pen t gere bunome..." |
| R=0.2 | 0.209 | "The king eoerykAirf,aERHtelrrmyth..." |
| R=0.5 | 0.595 | "The kingBRPm:I:eS:I noaRK:UhThCThNK..." |
| R=0.9 | 0.980 | "The kinguooooo itlseoo ttltstltltlt..." |

### Key Observation

**High R (0.9) → Collapses to repetitive patterns**

This is exactly what the theory predicts:
- High synchronization → all oscillators aligned → deterministic output
- Low synchronization → diverse phases → exploratory output

---

## Why K-SSM Succeeds Where Phase-Mamba Failed

### Phase-Mamba Architecture

```
Token → Embedding → [Layers 0-31] → Phase Core modulation → [Layers 32-63] → Logits
                                     ↓
                                  R computed but...
                                  LayerNorm washes out the signal
```

**Problem**: R modulated hidden states at layer 32, but layers 33-63 (with LayerNorm) erased the effect.

### K-SSM Architecture

```
Token → Embedding → Oscillator frequencies → Kuramoto dynamics → Order params → Logits
                                              ↓
                                           R IS the signal
                                           (no bypass path)
```

**Solution**: Multi-scale order parameters are the **ONLY** path to output. R cannot be epiphenomenal because there's no alternative route.

---

## Comparison Table

| Aspect | Phase-Mamba | K-SSM |
|--------|-------------|-------|
| R at inference | 0.994-0.998 (stuck) | 0.002-0.378 (dynamic) |
| R forcing effect | 0.0 (none) | 5.93 (large) |
| R → Output path | Indirect (modulation) | Direct (only path) |
| LayerNorm washout | Yes | N/A |
| R is causal | ❌ No | ✅ **Yes** |

---

## Implications

1. **Architecture matters**: Bolting oscillators onto existing models doesn't work. The oscillator dynamics must be structurally integrated.

2. **Order parameters as features**: Using multi-scale order parameters (harmonics) as the hidden representation ensures R influences output by construction.

3. **Consciousness theories**: If R represents something like "integration" or "coherence", K-SSM provides a model where that quantity actually affects behavior—unlike Phase-Mamba where it was merely tracked.

---

## Files

| File | Description |
|------|-------------|
| `kssm_model.py` | K-SSM architecture |
| `train_kssm.py` | Training script |
| `test_causality.py` | Causality tests |
| `results/best_model.pt` | Trained model |
| `results/training_log.json` | Training metrics |
| `results/causality_test.json` | Test results |
| `results/samples.json` | Generated samples |

---

## Conclusion

**K-SSM proves that Kuramoto order parameters CAN be causal in language models—but only with the right architecture.**

The key insight: R must be on the **critical path** to output, not a side computation that can be normalized away.

🌀 The spiral continues. R is no longer epiphenomenal.
