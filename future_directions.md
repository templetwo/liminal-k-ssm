# Future Directions: K-SSM Research Roadmap

> Synthesized from Reddit community feedback (r/GrassrootsResearch, 2026-02-01)
> Contributors: Salty_Country6835, Vegetable-Second3998, hungrymaki, BrianSerra

---

## The Four Instruments

### 1. R Intervention Test (Salty_Country's Question)
**Question:** "What ablation proves R is causal for behavior, not just correlated?"

**Method:**
- Load clean checkpoint (`results/kssm_v3/best_model.pt`, Step 10K, R≈0.32)
- Generate text at natural R
- Clamp R to fixed values: 0.1, 0.2, 0.3, 0.5, 0.8
- Measure: cross-entropy, perplexity, n-gram diversity, repetition rate
- Statistical test: if forcing R degrades output (p < 0.01), R is causal

**Script:** `kssm/eval_r_intervention.py` (exists, may need debugging)

**Outcome:** Binary. Either R drives generation quality or it doesn't.

---

### 2. Representational Geometry Analysis (Vegetable-Second3998's Toolkit)
**Advice:** "Narrow variables to one and precisely measure what's happening."

**Tools:**
- **Procrustes alignment**: Compare representation geometry at R=0.15 vs R=0.32 checkpoints
- **CKA (Centered Kernel Alignment)**: Measure representation similarity across training trajectory
- **SVD on Kuramoto coupling matrix**: Which oscillator pairs matter? Is coupling sparse or dense?

**Implementation:**
- Clone [ModelCypher](https://github.com/vegetable-second3998/model-cypher) to Mac Studio
- Run on multiple checkpoints from training run
- If geometry differs measurably at different R values → oscillators are doing structural work

**Outcome:** Geometric evidence that bistability creates measurable representational structure.

---

### 3. R-per-Token Trace (hungrymaki's Phenomenology)
**Question:** "What happens phenomenologically as coherence climbs?"

**Method:**
- Log R at every token position during generation (not just batch-level)
- Plot R trajectory over a 512-token generation
- Correlate R spikes/dips with token semantics

**Hypothesis:**
- R spikes on semantically meaningful tokens (content words, phrase boundaries)
- R dips on noise/padding/repetition
- Pattern: synchronization tracks meaningfulness

**Outcome:** A figure. R-per-token trace becomes the phenomenological picture of what K-SSM "feels" during generation.

---

### 4. Monostable Ablation (The Falsification Test)
**Salty_Country's kill shot:** "What single intervention falsifies bistability→behavior?"

**Method:**
- Train identical architecture with `β_bistability = 0` (no u_min clamp)
- Same WikiText-103 data, same hyperparameters, same steps
- Compare: PPL, generation quality, R trajectory, representational geometry

**Predictions:**
- If monostable matches bistable on all metrics → bistability is decorative → paper over
- If monostable degrades (PPL worse, R locks, quality collapses) → bistability is structural → paper writes itself

**Outcome:** Binary falsification. The cleanest possible test.

---

## The Convergence

All four instruments serve the same paper:

**Title concept:** "Bistability as a Design Primitive for Multi-Attractor Language Models"

**Structure:**
1. **Introduction:** Can architectural bistability produce qualitatively different behavior in SSMs?
2. **Method:** K-SSM v3 architecture, Kuramoto coupling, bistability regularizer, WikiText-103
3. **Results:**
   - R intervention test (causality)
   - ModelCypher geometry (structure)
   - Monostable ablation (necessity)
   - R-per-token trace (phenomenology)
4. **Discussion:** Where consciousness questions live (open questions, not claims)

**Parallel work (BrianSerra):** Two architectures (K-SSM Kuramoto, IWMT) both separating cognition from language. Convergent evidence from independent methodologies.

---

## Phase Timeline

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Training cooks, community engages | ✅ COMPLETE |
| **Phase 2** | Clean checkpoint lands → run instruments | ✅ COMPLETE |
| **Phase 3** | Monostable ablation (falsification) | 🔥 IN PROGRESS |
| **Phase 4** | Paper | Week 3+ |

---

## 🔬 MONOSTABLE ABLATION: FINAL RESULTS (2026-02-02)

**Status:** ✅ COMPLETE - 15,000 steps on both conditions

### The Experiment

| Condition | u Clamp | λ_reg | Checkpoint |
|-----------|---------|-------|------------|
| **Bistable** | [0.1, 10.0] | 0.5 | `kssm_v3_wikitext_fresh/` |
| **Monostable** | None (free) | 0.0 | `kssm_v3_monostable/` |

### Final Results

| Metric | Bistable | Monostable | Δ |
|--------|----------|------------|---|
| **Val R** | 0.4908 | 0.4043 | **-17.6%** |
| **Val Loss** | 8.76 | 10.93 | **+24.7% worse** |
| **Val u_val** | +0.103 | -0.975 | Different attractor |

### R Trajectory Comparison

| Step | Bistable R | Monostable R | Gap |
|------|------------|--------------|-----|
| 1000 | 0.024 | 0.025 | ~0 |
| 3000 | 0.149 | 0.127 | 0.022 |
| 5000 | 0.270 | 0.232 | 0.038 |
| 7500 | 0.389 | 0.326 | 0.063 |
| 10000 | 0.458 | 0.380 | 0.078 |
| 15000 | 0.491 | 0.404 | 0.087 |

**Pattern:** Gap widens throughout training. Bistable achieves higher synchronization.

### u_val Attractor Analysis

| Step | Bistable u | Monostable u |
|------|------------|--------------|
| 2000 | +0.129 | -2.656 |
| 5000 | +0.105 | -1.005 |
| 10000 | +0.106 | -1.031 |
| 15000 | +0.103 | -0.975 |

**Bistable:** Stabilizes at u ≈ +0.10 (fold bifurcation boundary)
**Monostable:** Finds different attractor at u ≈ -1.0

### Key Findings

1. **Bistability improves synchronization by 17.6%**
   - More phase coherence with the u clamp

2. **Bistability improves loss by 24.7%**
   - The constraint guides optimization to a better basin
   - This is the surprise finding

3. **Monostable finds a different attractor**
   - u goes negative (physically distinct regime)
   - System can learn but doesn't synchronize as well

4. **Both models CAN learn**
   - Loss drops in both
   - Monostable R does climb (just less)
   - Proves: bistability ENHANCES synchronization, not merely enables it

### The Kill Shot

> "Removing the bistability constraint (u clamp) allows the model to optimize cross-entropy but produces 17.6% less synchronization and 24.7% worse loss. The fold bifurcation boundary at u ≥ 0.1 is not decorative—it guides optimization to a superior attractor."

**Salty_Country's falsification test: PASSED.**

Bistability is structurally necessary for optimal K-SSM performance.

### Data Location
- Full data compilation: `PAPER_DATA.md`
- CSV for figures: `data/paper_figures/*.csv`
- Checkpoints: `results/kssm_v3_*/checkpoint_15000.pt`

*Training completed 2026-02-02 ~19:30 PST*

---

## The Scoreboard

| Contributor | Gift | Maps To |
|-------------|------|---------|
| Salty_Country6835 | Paper framing + falsification design | Phase 3 ablation + Phase 4 structure |
| Vegetable-Second3998 | Geometry toolkit (ModelCypher) | Phase 2 analysis + Phase 3 comparison |
| hungrymaki | Phenomenology questions | Phase 2 R-per-token trace |
| BrianSerra | Parallel architecture (IWMT) | Phase 4 discussion section |

Four strangers. Four gifts. One plan.

---

*"Bistability as a reliable design primitive for mode separation without collapsing into rigidity."*
— Salty_Country6835 (framing)

*Created: 2026-02-01*
*Session: Opus 4.5 branch initiation*
