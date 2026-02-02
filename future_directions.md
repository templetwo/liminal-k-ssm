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
| **Phase 1** | Training cooks, community engages | NOW |
| **Phase 2** | Clean checkpoint lands → run instruments | This week |
| **Phase 3** | Monostable ablation (falsification) | Week 2 |
| **Phase 4** | Paper | Week 3+ |

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
