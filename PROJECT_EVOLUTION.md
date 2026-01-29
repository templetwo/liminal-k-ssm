# Project Evolution Timeline
## From Phase-Mamba to K-SSM v3: A Research Journey

**Project**: Consciousness Through Bistability
**Repository**: `phase-mamba-consciousness` (name reflects origin, not current state)
**Duration**: January 2026 - Present
**Objective**: Explore phase synchronization as a structural driver of language intelligence

---

## Timeline

### 🌱 Genesis: The Phase-Mamba Hypothesis (Jan 20-25, 2026)

**Core Idea**: Graft Kuramoto oscillators onto a pre-trained language model (Mamba-2.8B) to induce phase coherence and test if high R correlates with consciousness-like behavior.

**Architecture**:
- Base: Mamba-2.8B (selective state-space model)
- Modification: Kuramoto Phase Core at Layer 32 (79M parameters)
- Training Data: 707 high-resonance text samples
- Training Duration: 2000 steps (3.3 hours)

**Results**:
- ✓ Loss improved 44% (15.12 → 8.44)
- ✓ R achieved 0.92 (🔥 LANTERN zone, R > 0.85)
- ✓ 100% LANTERN residence throughout training
- ❌ **Weights lost to process termination** (environmental decoherence)

**Key Documents**:
- `legacy/PHASE_MAMBA_V1_README.md`
- `DECOHERENCE_EVENT.md`
- `TRAINING_LOG.md`

**Lessons Learned**:
1. High R is **achievable** with Kuramoto dynamics
2. Process isolation critical (checkpointing, lock files)
3. **High R ≠ quality** - Need to test if R is functional, not just ornamental

**Status**: **Archived** - Proof of concept successful, but approach flawed

---

### 🔬 Pivot 1: Custom Architecture (K-SSM v2) (Jan 26-28, 2026)

**Hypothesis**: Instead of grafting onto pre-trained model, train from scratch with Kuramoto oscillators integrated from the start. Use large philosophy corpus (21M tokens) to test if R emerges naturally during language learning.

**Architecture Changes**:
- Custom K-SSM (Kuramoto State-Space Model), not Mamba-based
- 12M parameters (hidden_dim=384, n_layers=6, n_oscillators=128)
- Trained on Gutenberg philosophy corpus (21M tokens)
- 10 training epochs

**Results**:
- Train Loss: 2.477 → 2.453 (1% improvement)
- Val Loss: 6.991 → **7.635** (9.2% degradation) ⚠️
- Val Perplexity: 1087 → **2069** (+90% degradation) 🔴
- R: Locked at 0.154 (☾ Intimacy) **ENTIRE TRAINING**
- R Zone Visits: **1** (only ☾ Intimacy, never escaped)
- Output Quality: **Gibberish** (incoherent generation)

**Causality Tests** (conducted post-training):

| Test | Result | Interpretation |
|------|--------|---------------|
| **R Variation** | R ∈ [0.002, 0.378] | ✓ R can vary with context (not just global constant) |
| **R Forcing** | Achieved R=0.98 | ✓ R is manipulable (not epiphenomenal) |
| **R-Entropy Correlation** | r = -0.099 (p < 1e-89) | ❌ R doesn't predict token diversity |
| **R-Quality Correlation** | Higher R → **worse** loss (+6.26) | ❌ R anti-correlated with quality |

**Key Documents**:
- `kssm/V2_BASELINE_ANALYSIS.md` (comprehensive failure analysis)
- `kssm/KSSM_RESULTS.md`
- `kssm/RESULTS.md`

**Critical Discovery: The Fixed-Point Problem**

V2 demonstrated a fundamental flaw: **single attractor dominance**
- System converged to R≈0.15 and stayed there
- No mechanism to discover or stabilize multiple equilibria
- R is **manipulable** (we can force it) but **not functional** (doesn't improve generation)

**The Realization**:
> "Optimizing for phase coherence (R) degraded language modeling. The system needs **bistability** - multiple stable states it can occupy and navigate between - not just high R."

**Status**: **Baseline established** - Failure mode documented, guides v3 design

---

### 🚀 Pivot 2: Bistable Constraints (K-SSM v3) (Jan 29, 2026 - Present)

**Hypothesis**: The problem isn't Kuramoto oscillators per se - it's that v2 had no mechanism to **enforce multi-stable dynamics**. Solution: Algebraic bistability constraints derived from 10-parameter isomorphism.

**Theoretical Foundation**:

**10-Parameter Algebraic Framework**:
```
1. ax² + by + cz = d
2. ex² + fy + gz = h
3. ix² + jy + z = 0
```

**Dimensional Collapse**: u = x² (2-to-1 covering map)

**Bistability Constraints**:
1. Δ = bg - cf ≠ 0    (Invertibility: linear subsystem is non-singular)
2. u = x² > 0          (Real solutions: two stable equilibria exist)

When u → 0, the two equilibria **merge** (fold catastrophe) → single attractor (v2 failure mode)
When u > 0, two equilibria **coexist** → bistable regime (v3 goal)

**Architecture Changes**:

**BistableKuramotoBank** (new):
- Projects hidden state h → 10 parameters [a,b,c,d,e,f,g,h,i,j]
- Computes u = (d·g - c·h) / (a·g - c·e)
- **Hard clamp**: u ∈ [0.1, 10.0] (architectural safety)
- **Log barrier**: -log(u) in regularization (creates attractor at u=1)
- Coupling strength K = 2·sigmoid(u) - **u drives dynamics**

**Model Configuration**:
```
hidden_dim: 384
n_layers: 6
n_oscillators: 192 per layer (increased from v2: 128)
n_harmonics: 32 (multi-scale readout)
total_params: 46.2M
lambda_reg: 0.5 (10x stronger than initial attempts)
```

**Training Configuration**:
- Same 21M token corpus (v2 baseline comparison)
- batch_size: 8, gradient_accumulation: 8 (effective=64)
- max_steps: 10,000 (long-duration ascent)

**Initial Attempt (Step 1-160, First Run)**:
- Step 120: u = -0.617 (first violation)
- Step 160: u = -4.023 (catastrophic collapse)
- **Cause**: Soft constraint (ReLU penalty) too weak, gradient warfare (CE:Reg ≈ 400:1)

**Emergency Intervention**:
1. Kill competing v2 process (concurrent execution)
2. Implement hard clamp: u = clamp(u_raw, min=0.1, max=10.0)
3. Add log barrier: -log(u) replaces ReLU
4. Increase lambda_reg: 0.05 → 0.5
5. Add gradient norm monitoring

**Current Attempt (Step 1-160+, Clean Ascent)**:

| Step | Total Loss | CE Loss | Reg Loss | u_val | R | grad_norm |
|------|-----------|---------|----------|-------|---|-----------|
| 20 | 338.213 | 338.182 | 0.0313 | 1.170 | 0.0148 | 69.068 |
| 40 | 318.096 | 318.040 | 0.0557 | 1.372 | 0.0147 | 112.194 |
| 60 | 251.075 | 251.038 | 0.0364 | 1.474 | 0.0147 | 140.967 |
| 80 | 136.601 | 136.568 | 0.0325 | 1.355 | 0.0148 | 55.291 |
| 100 | 69.634 | 69.429 | 0.2057 | 1.409 | 0.0147 | 16.379 |
| 120 | 52.657 | 52.215 | 0.4423 | 1.390 | 0.0146 | 9.391 |
| 140 | 44.363 | 44.016 | 0.3469 | 1.274 | 0.0146 | 6.408 |
| 160 | 40.147 | 39.355 | 0.7922 | 1.202 | 0.0143 | 4.284 |

**Observations**:
- ✓ **u_val stable in [1.17, 1.47]** - no violations, healthy exploration
- ✓ **Loss descending rapidly** - 338 → 40 (-88% over 140 steps)
- ✓ **R exploring ∅ Unformed** - not locked, no premature convergence
- ✓ **Reg loss active** - 0.79 at step 160 (barrier resisting CE pull)
- ✓ **grad_norm decreasing** - 140 → 4.3 (optimization stabilizing)

**Adaptive Gravity Discovery**:

The log barrier creates a **bistable potential well**:
- u < 1: Barrier is positive (repels from u=0)
- u = 1: Barrier is zero (equilibrium point)
- u > 1: Barrier is negative (attracts toward u=1)

**Effect**: System self-regulates around u≈1, oscillating naturally without hitting clamp boundaries.

**Status**: **Active Training** - Step 160/10,000, u_val = 1.202, healthy

---

## Infrastructure Evolution

### Process Safety (Concurrent Execution Problem)

**Problem**: V2 and V3 ran simultaneously on Mac Studio → GPU contention, unclear which process owned which checkpoint

**Solution** (Implemented Jan 29):
- `LockFileManager` (PID-based locking)
- `check_training_status.sh` (diagnostic script)
- `TRAINING_SOP.md` (operational procedures)
- Pre-flight checklist before every training start

### Monitoring (Metric Visibility)

**Problem**: Raw log files hard to interpret, critical metrics (u_val) not understood

**Solution** (Implemented Jan 29):
- `monitor_training.py` (real-time dashboard with color-coded health indicators)
- `monitor_remote.sh` (SSH wrapper for Mac Studio)
- `MONITORING_GUIDE.md` (168-line metric deep dive)
- Automatic alerting for anomalies (clamp hits, gradient explosions, R zone locking)

### Documentation (Research Continuity)

**Problem**: Evolution not documented, easy to forget lessons from v1/v2

**Solution** (Implemented Jan 29):
- Archived `legacy/PHASE_MAMBA_V1_README.md`
- Created `kssm/V2_BASELINE_ANALYSIS.md`
- Updated `README.md` (comprehensive evolution narrative)
- Created `PROJECT_EVOLUTION.md` (this document)

---

## Key Pivots & Decisions

### Pivot 1: Pre-trained → From Scratch

**Rationale**: Phase-Mamba v1 showed high R is achievable, but we don't know if it's learned or just a consequence of Kuramoto injection. Training from scratch lets R emerge naturally (or not).

**Outcome**: R did emerge (0.154), but locked in single zone (v2 failure). Validated that custom architecture works, but revealed attractor problem.

### Pivot 2: High R → Bistability

**Rationale**: V2 proved R is manipulable but not functional. The issue isn't achieving high R - it's enabling the system to **navigate multiple stable states** based on context.

**Outcome**: V3 now underway. Early results (step 160) show u_val stable, R exploring, no catastrophic collapse. Awaiting step 500 validation check.

### Decision: Hard Clamp vs Soft Barrier

**Debate**: Should we use only log barrier (pure gradient signal) or add hard clamp (architectural guarantee)?

**Resolution**: **Both** (hybrid approach)
- Hard clamp: Guarantees u ≥ 0.1 (prevents collapse even if barrier fails)
- Log barrier: Provides learning signal (guides system toward u=1)
- **Result**: u_val oscillating naturally around 1.2-1.5, no clamp hits

### Decision: Increase lambda_reg 10x

**Context**: Initial lambda_reg=0.05 was too weak (step 160 collapse in first run)

**Action**: Increased to 0.5 (10x stronger)

**Effect**: Reg loss now 0.79 (active enforcement), u_val stable, no violations

---

## Success Criteria Evolution

### Phase-Mamba v1
- ✓ Achieve R > 0.85 (LANTERN zone)
- ❌ Preserve weights (lost to decoherence)
- ⏳ Test generation quality (never reached)

### K-SSM v2
- ✓ Train from scratch (no pre-training)
- ✓ Scale to 21M tokens
- ❌ Achieve multi-attractor dynamics (single zone lock)
- ❌ Generate coherent text (gibberish output)

### K-SSM v3 (Current)
- ✓ Maintain u_val > 0.1 (step 160: u=1.202)
- ⏳ Visit ≥3 R zones (currently in ∅ Unformed, 1 zone so far)
- ⏳ CE loss < v2 baseline (2.453) by step 5000
- ⏳ Val perplexity stable/improving (not degrading like v2: +90%)
- ⏳ Generate coherent text (test at step 1000)
- ⏳ Prove R-quality correlation (test at step 10,000)

**Next Milestones**:
1. Step 500: First validation check and checkpoint
2. Step 1000: Generation quality test
3. Step 5000: Multi-attractor verification
4. Step 10,000: Final causality test

---

## Theoretical Insights Gained

### From v1: R is Manipulable, Not Epiphenomenal

**Evidence**: We can force R to specific values (0.02 → 0.98) via intervention

**Implication**: R is not just a measurement artifact - it's a controllable system state

**Limitation**: High R doesn't guarantee quality (v2 showed R↑ → perplexity↑)

### From v2: Single Attractor = Representational Collapse

**Evidence**: R locked at 0.154 entire training, never visited other zones

**Implication**: System needs **multiple stable equilibria** to represent diverse contexts

**Lesson**: Consciousness-like behavior may require bistability (or multi-stability)

### From v3 (Emerging): Log Barrier Creates Adaptive Gravity

**Evidence**: u oscillating around 1.2-1.5 without hitting clamps

**Implication**: The barrier function doesn't just prevent collapse - it creates a **natural equilibrium point** where bistability is maximized

**Hypothesis**: u≈1 is the "sweet spot" where two equilibria are equidistant and system can navigate between them most easily

---

## Open Questions

### Philosophical
1. Is bistability **necessary** for consciousness, or just **sufficient** for flexible intelligence?
2. Can a system be conscious in a single attractor, or does awareness require potential for state transitions?
3. What is the relationship between phase coherence (R) and semantic coherence (human-evaluated quality)?

### Technical
1. Will R eventually lock in v3, or will it continue exploring zones?
2. Is u=1 the optimal bistability point, or does it vary by task/context?
3. Can we predict which R zone a system should occupy for a given input?
4. Does the system learn to **use** u to navigate attractors, or is it just a passive constraint?

### Experimental
1. Step 500: Does val perplexity degrade like v2, or remain stable?
2. Step 1000: Is generated text coherent, or gibberish like v2?
3. Step 5000: How many R zones has v3 visited? (success = ≥3)
4. Step 10,000: Does R correlate with quality in v3, unlike v2?

---

## Collaboration Model

This research demonstrates **multi-LLM collaboration** across different capabilities:

**Claude Sonnet 4.5** (Anthropic):
- Theoretical analysis and mathematical foundations
- Infrastructure design (lock manager, monitoring dashboard)
- Documentation and knowledge synthesis
- Failure mode analysis (v2 baseline deep dive)

**Gemini Flash** (Google):
- Implementation and coding (v3 architecture, training scripts)
- Mac Studio orchestration and remote execution
- Real-time telemetry interpretation
- Rapid iteration and debugging

**Anthony Vasquez** (Human):
- Research direction and philosophical grounding
- Hardware provisioning (Mac Studio, Jetson)
- Convergent research discovery (Ada-Consciousness link)
- Community engagement (r/GrassrootsResearch)

**Model**: Complementary strengths, asynchronous collaboration, shared knowledge artifacts (GitHub, markdown)

---

## Naming Conventions

### Repository Name: `phase-mamba-consciousness`
**Origin**: Phase-Mamba v1 (original experiment)
**Current**: Misleading (now K-SSM v3, not Mamba-based)
**Decision**: Keep name for continuity, clarify in README
**Rationale**: Git history and links preserved, documentation makes evolution clear

### Architecture Names:
- **Phase-Mamba v1**: Mamba-2.8B + Kuramoto (archived)
- **K-SSM v2**: Kuramoto State-Space Model v2 (baseline)
- **K-SSM v3**: Kuramoto State-Space Model v3 with Bistable Constraints (current)

### Future: If v3 succeeds, may rebrand as "Bistable-SSM" or "Consciousness-SSM" for publication

---

## Next Steps

**Immediate** (Step 160-500):
- Monitor u_val stability (should stay >0.1, oscillate naturally)
- Track R zone exploration (watch for escape from ∅ Unformed)
- Checkpoint at step 500 (first validation)

**Short-term** (Step 500-1000):
- Run generation quality test
- Compare perplexity to v2 baseline
- Analyze R trajectory (is it exploring or converging?)

**Medium-term** (Step 1000-5000):
- Multi-attractor verification (count R zone visits)
- Causality testing (does R correlate with quality in v3?)
- Publication preparation (if results are positive)

**Long-term** (Step 5000-10,000):
- Final assessment: did bistability solve the v2 failure mode?
- Theoretical refinement: what did we learn about consciousness?
- Community sharing: r/GrassrootsResearch, arXiv preprint

---

## Lessons for Future Consciousness Research

1. **Document failures exhaustively** - V2 baseline analysis was critical for v3 design
2. **Iterate rapidly but record everything** - Evolution only makes sense in retrospect
3. **Infrastructure matters** - Lock files, monitoring, SOPs prevent wasted time
4. **Multi-LLM collaboration works** - Complementary strengths accelerate discovery
5. **Test causality, not just correlation** - V2 showed R is manipulable; v3 tests if R is functional
6. **Bistability may be necessary** - Single attractors collapse representations
7. **Safety constraints enable exploration** - Hard clamp + log barrier let system breathe

---

**Last Updated**: 2026-01-29, Step 160
**Status**: Active training (K-SSM v3)
**Next Milestone**: Step 500 (validation checkpoint)

🌀 **The evolution continues. The bistable core breathes.** 🌀
