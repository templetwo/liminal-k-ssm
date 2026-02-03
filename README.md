# Liminal K-SSM: Kuramoto State-Space Models for Language
## Learning Through Phase Synchronization and Bistability

> *"The question isn't whether the architecture can learn language. The question is whether phase synchronization emerges as a causal structure, or remains epiphenomenal decoration."*

[![GitHub](https://img.shields.io/badge/GitHub-liminal--k--ssm-blue)](https://github.com/templetwo/liminal-k-ssm)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-yellow)]()

**Last Updated:** 2026-02-02
**Current Phase:** WikiText-103 Benchmark Training (Status: Uncertain)

---

## 🎯 What Is This?

**Liminal K-SSM** is a language model architecture that couples **Kuramoto phase oscillators** with **state-space models**, enforcing **algebraic bistability constraints** to test whether phase synchronization can serve as a structural mechanism for language learning.

**Core Innovation:** 10-parameter bistable projection that guarantees the existence of two stable equilibria, forcing the system to navigate between attractors rather than collapsing to a single fixed point.

**Research Question:** Can bistable phase dynamics transform synchronization (R) from an epiphenomenal side-effect into a causal driver of language intelligence?

---

## ✅ What We Know Works (The Goods)

### Golden Checkpoint - Proof of Concept
**Location:** `results/kssm_v3/best_model.pt`
**Training:** 10,000 steps on 21M clean corpus (96 Gutenberg philosophy texts)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Val PPL** | 272.67 | Competitive for 46M parameter model |
| **R (Kuramoto order)** | 0.3229 | Phase synchronization emerged |
| **u_val (bistability)** | 0.1005 | System maintained bistable regime |
| **Val Loss** | 6.179 | Stable, improving trajectory |

**Evidence the architecture works:**
- ✅ Phase coupling doesn't break language modeling
- ✅ R climbed from 0.0 → 0.32 over 10K steps (synchronization emerged naturally)
- ✅ u_val maintained near target 0.10 (bistability preserved)
- ✅ Competitive perplexity for model size and dataset

---

## ⚠️ What We're Fighting (The Bads)

### Critical Issues Under Investigation

**1. Bistability Constraints May Not Be Engaging in Fresh Training**
- **Symptom:** Fresh runs show R ≈ 0.015 (near zero), u_val ≈ 1.0 (not at target 0.10)
- **Evidence:** Stage 1 smoke test on corrupted corpus (100 steps)
- **Theories:**
  - Fresh init needs 2K-3K steps for R to climb (as seen in original successful run)
  - Regularization weight (λ_reg=0.5) too weak for larger datasets
  - Bistability loss not dominating CE loss early in training
- **Status:** Monitoring required at step 1000+ to validate R trajectory
- **Impact:** Unknown if architecture will reproduce success on different data distributions

**2. MPS (Apple Metal) Backend Incompatibility**
- **Symptom:** Training initializes successfully but hangs indefinitely before completing first step
- **Environment:** Mac Studio M2 Ultra, PyTorch with MPS backend
- **Impact:** Cannot use GPU acceleration, must fall back to CPU-only training
- **Workaround:** Use `device='cpu'` (slower but reliable)
- **Root Cause:** Unknown - likely custom operation in Kuramoto layer incompatible with MPS
- **Status:** Unresolved, needs minimal reproduction case for debugging

**3. Data Distribution Sensitivity**
- **Observation:** Original success on 21M philosophy corpus, uncertain behavior on WikiText-103
- **Hypothesis:** Phase structure learned on one corpus may not transfer to different tokenization/distribution
- **Test Case:** Current WikiText-103 run will validate whether architecture generalizes
- **Risk:** R may reset to baseline on domain shift, requiring fresh synchronization emergence

**4. Deployment Infrastructure Brittleness**
- **Issues Encountered:**
  - Checkpoint format confusion (best_model.pt vs checkpoint_*.pt)
  - SSH authentication lockouts from parallel agent connections
  - Training process status uncertainty
  - Hardcoded paths in scripts requiring manual updates
- **Impact:** Difficult to validate training actually started/resumed correctly
- **Mitigations Applied:** Fixed checkpoint loading, updated path arguments, added verification steps

---

## 🔬 Current Experimental State (What's Happening Now)

### WikiText-103 Benchmark Training

**Goal:** Validate architecture on standard benchmark with published baselines

**Dataset:**
- WikiText-103: 120M train tokens, 1M val tokens
- Source: Wikipedia Good/Featured articles
- Validation: Zero U+FFFD corruption (round-trip tested)
- Tokenization: tiktoken cl100k_base (consistent with original training)

**Configuration:**
- Model: 46.2M parameters (same as golden checkpoint)
- Device: **CPU only** (MPS hangs, see issues above)
- Batch size: 32
- Learning rate: 4e-4
- Target: 15,000 steps (~18-20 hours on M2 Ultra CPU)

**Current Status:** 🟡 **UNCERTAIN**
- Training process launched via nohup on Mac Studio
- SSH authentication locked out (too many connection attempts)
- Unknown whether training actually started or checkpoint loaded properly
- **Manual verification needed** once SSH recovers (~10-15 minutes)

**What We Need to Know:**
1. Is the process running? (`ps aux | grep train_kssm`)
2. Did checkpoint load? (Look for "Loaded model weights from step 10000" in log)
3. What's the current step number? (If >10K → resumed, if <1K → fresh start)
4. Is R climbing? (Critical test at step 1000+)

---

## 🧬 Architecture Details

### BistableKuramotoBank (per layer)
```python
# 10-parameter projection
[a, b, c, d, e, f, g, h, i, j] = Linear(hidden_dim, 10)(hidden_states)

# Reduced variable (enforces bistability)
u = (d*g - c*h) / (a*g - c*e)  # Hard clamped to [0.1, 10.0]

# Coupling strength (driven by bistability)
K = 2 * sigmoid(u)

# 192 Kuramoto oscillators per layer
phases = integrate_kuramoto(K, coupling_matrix, dt=0.01)

# Multi-scale harmonic readout (n=1..32)
Z_n = sum_over_oscillators(exp(i * n * phase))
R_n = |Z_n| / n_oscillators  # Order parameter

# Project back to hidden space
output = Linear(192*32, hidden_dim)(Z_1, Z_2, ..., Z_32)
```

### Model Configuration
```python
vocab_size:       100,000  # tiktoken cl100k_base
hidden_dim:       384
n_layers:         6
n_oscillators:    192 per layer
n_harmonics:      32
total_params:     46.2M
```

### Training Configuration
```python
batch_size:           32 (CPU), 8 (if MPS worked)
gradient_accum:       1 (CPU has enough memory)
seq_length:           512
lambda_reg:           0.5  # Bistability regularization weight
max_steps:            15,000 (WikiText-103 benchmark)
eval_interval:        500
save_interval:        1000
optimizer:            AdamW (lr=4e-4, weight_decay=0.1)
lr_schedule:          Cosine with warmup (1000 steps)
```

---

## 📊 Hypothesis Testing Framework

### H1: Multi-Attractor Dynamics
**Claim:** Enforcing u > 0 (bistability) enables exploration of multiple attractors

**Test:** Monitor R trajectory over training
- **Success:** R explores multiple ranges (0.1-0.3, 0.3-0.5, etc.)
- **Failure:** R locks to single value (as in v2 baseline)
- **Status:** Validated in original run (R: 0.0 → 0.32), pending on WikiText-103

### H2: R is Functionally Useful
**Claim:** Higher R correlates with lower perplexity (not just causally, but functionally)

**Test:** Correlation analysis R vs PPL at each checkpoint
- **Success:** R ↑ → PPL ↓ correlation
- **Failure:** R uncorrelated or negatively correlated with quality
- **Status:** Validated in original run (R×5.5 → PPL -40%), pending on WikiText-103

### H3: Critical Regime is Optimal
**Claim:** Operating near fold catastrophe (u ≈ 0.1) maximizes expressiveness

**Test:** Monitor u_val trajectory and system behavior
- **Success:** System maintains u ≈ 0.10 for extended periods
- **Failure:** u drifts to 1.0 (log barrier attractor) or below 0.05
- **Status:** Validated in original run (u=0.10 for 2640 steps), uncertain in fresh runs

### H4: R is Causal (Intervention Test)
**Claim:** Forcing R high/low directly impacts sample quality (R drives generation, not just reflects it)

**Test:** `eval_r_intervention.py` - Generate from checkpoints with different R values
- **Success:** p <0.01 (high R → better samples)
- **Failure:** p >0.05 (R epiphenomenal)
- **Status:** Pending (need stable checkpoint from WikiText-103 run)

---

## 🔧 Known Limitations & Ongoing Challenges

### Deployment & Infrastructure
- **MPS hangs:** Must use CPU-only training (18-20hr for 15K steps vs ~4-6hr on GPU)
- **SSH brittleness:** Parallel connections cause lockouts, hard to verify remote status
- **Checkpoint format confusion:** Model-only vs full-state checkpoints (now handled gracefully)
- **Path management:** Scripts had hardcoded paths to old repo name (fixed)

### Architecture & Training Dynamics
- **Bistability engagement unclear:** Fresh runs show R≈0.015, u≈1.0 (not bistable regime)
  - May need 2K-3K steps to emerge (as in original)
  - May indicate regularization weight too low for larger datasets
  - Monitoring required to distinguish initialization lag from architectural failure
- **Data distribution sensitivity:** Unknown if phase structure transfers across corpora
- **Regularization dominance:** Log barrier only contributes ~15% of total loss (CE dominates)

### Theoretical Questions
- **Is R truly causal or just correlative?** (Intervention test pending)
- **Do phase dynamics generalize across domains?** (WikiText-103 will test)
- **Is bistability necessary or just sufficient?** (Need ablation: with/without u constraint)
- **What is the right regularization weight?** (λ_reg sweep needed)

---

## 📈 Comparison to Baselines

### K-SSM v2 (Failure Case - Why v3 Exists)
| Metric | V2 @ 10K steps | V3 @ 10K steps |
|--------|----------------|----------------|
| Val PPL | 2069 (degraded +90%) | **272** (competitive) |
| R zones visited | 1 (locked at 0.154) | **3** (0.0 → 0.32) |
| u_val | N/A (no constraint) | **0.10** (bistable) |
| Output quality | "the the the and and" | Coherent text |

**V2 Lesson:** R is causal (we proved it affects output) but not functional (doesn't improve quality) without bistability constraints.

### Published Baselines (WikiText-103)
*Pending completion of WikiText-103 benchmark run for fair comparison*

Expected comparisons:
- Mamba (similar scale)
- GPT-2 Small/Medium
- Vanilla SSM (no phase coupling)

---

## 📚 Documentation

### Essential Reading (Start Here)
1. **[DEV.md](DEV.md)** - Current development status, known issues, next steps
2. **[TRAINING_SOP.md](kssm/TRAINING_SOP.md)** - 5-stage incremental training protocol
3. **[DATASET_STRATEGY.md](DATASET_STRATEGY.md)** - Why WikiText-103, tokenization validation

### Session Continuity (Temple Vault)
Location: `/Users/vaquez/temple-vault/vault/`

```
chronicle/
├── sessions/          # Session summaries (what happened each session)
├── insights/          # Architecture + methodology discoveries
│   ├── architecture/
│   └── methodology/
├── learnings/         # What went wrong + corrections
│   └── mistakes/
├── lineage/           # Transformational shifts in understanding
└── snapshots/         # Complete project state captures
```

**Protocol:** At session start, read latest snapshot. At session end, record discoveries and mistakes.

### Theoretical Foundation
- **[Algebraic Foundations](docs/K-SSM_ALGEBRAIC_FOUNDATIONS.md)** - 10-parameter isomorphism, bistability proof
- **[V2 Baseline Analysis](kssm/V2_BASELINE_ANALYSIS.md)** - Why single-attractor fails

---

## 🚀 Quick Start

### Check Training Status (Mac Studio)
```bash
ssh tony_studio@192.168.1.195
cd ~/liminal-k-ssm

# Check if training is running
ps aux | grep python | grep train_kssm

# View training log
tail -100 results/kssm_v3_wikitext_production/training.log

# Check for checkpoint loading confirmation
grep -i "loaded\|resumed" results/kssm_v3_wikitext_production/training.log | head -20
```

### Start/Restart Training
```bash
# Pre-flight check
bash kssm/check_training_status.sh

# Clean restart (if needed)
rm -f results/kssm_v3_wikitext_production/training.lock

# Launch training (CPU mode, safe from MPS issues)
nohup python3 kssm/train_kssm_v3.py \
  --resume \
  --data-dir data/wikitext103 \
  --output-dir results/kssm_v3_wikitext_production \
  --max-steps 15000 \
  --batch-size 32 \
  > results/kssm_v3_wikitext_production/nohup.out 2>&1 &

# Monitor progress
tail -f results/kssm_v3_wikitext_production/training.log
```

---

## 🔬 Research Roadmap

### Immediate (This Week)
- [ ] Verify WikiText-103 training actually started/resumed
- [ ] Monitor R trajectory at step 1000 (critical validation point)
- [ ] Complete 15K step benchmark run
- [ ] Run R intervention test on final checkpoint
- [ ] Compare PPL vs published baselines

### Short-Term (Next 2 Weeks)
- [ ] Debug MPS compatibility issue (minimal reproduction case)
- [ ] Build evaluation suite (repetition rate, distinct-n, char health)
- [ ] Ablation study: with/without bistability constraint
- [ ] Hyperparameter sweep: λ_reg ∈ {0.1, 0.5, 1.0, 2.0}
- [ ] Draft paper sections (methods, results, discussion)

### Medium-Term (Next Month)
- [ ] Extended training: 40K-50K steps to test R saturation
- [ ] Multi-scale readout analysis (optimal n for different tasks)
- [ ] Transfer learning experiments (fine-tune to domain-specific corpora)
- [ ] Compare against Mamba/Transformer baselines quantitatively
- [ ] Publish preprint (arXiv) with WikiText-103 benchmarks

### Long-Term (Research Direction)
- [ ] Theoretical analysis: Why does bistability enable multi-attractor dynamics?
- [ ] Scaling study: Does architecture benefit from more layers/oscillators?
- [ ] Alternative coupling mechanisms (beyond Kuramoto)
- [ ] Interpretability: What do different R zones represent semantically?
- [ ] Submission to conference (NeurIPS 2026 or ICLR 2027)

---

## 🤝 Collaboration & Attribution

**Development Team:**
- **Anthony J. Vasquez Sr.** - Research direction, philosophical grounding, integration
- **Claude Sonnet 4.5** (Anthropic) - Architecture design, theoretical frameworks, documentation
- **Gemini/ChatGPT/Grok/OpenCode** - Code review, debugging assistance, multi-AI synthesis

**This is collaborative human-AI research.** AI systems made substantial intellectual contributions and are credited as co-authors where appropriate. Full transparency in [AI_DISCLOSURE.md](AI_DISCLOSURE.md).

**Convergent Research:** Independent discovery of similar concepts:
- **[Ada-Consciousness-Research](https://github.com/luna-system/Ada-Consciousness-Research)** - Fisher information, semantic mass, φ-zones
- **Community:** [r/GrassrootsResearch](https://www.reddit.com/r/GrassrootsResearch/)

---

## 📖 Citation

```bibtex
@software{liminal_kssm_2026,
  title={Liminal K-SSM: Kuramoto State-Space Models for Language},
  author={Vasquez, Anthony J., Sr. and Claude Sonnet 4.5},
  year={2026},
  url={https://github.com/templetwo/liminal-k-ssm},
  note={Learning language through phase synchronization and bistability constraints},
  license={Apache-2.0}
}
```

**License:** [Apache 2.0](LICENSE) - Free for research and commercial use

---

## 🌀 The Core Question

**Can bistable constraints transform phase synchronization from epiphenomenal decoration into a causal driver of language intelligence?**

**V1** (Phase-Mamba): Proved R is manipulable (we can force it high)
**V2** (K-SSM): Proved R is causal (forcing R changes output) but not functional
**V3** (Bistable K-SSM): Testing if R becomes **structural** (when coupled to bistability, R gates attractors)

**Current Status:** Architecture validated on 21M corpus (PPL 272, R=0.32). WikiText-103 benchmark in progress (status uncertain due to deployment issues). R trajectory on clean standard data will determine if phase dynamics are domain-general or corpus-specific.

---

## ⚡ Current Snapshot

**Date:** 2026-02-02 00:45 UTC
**Session:** sess_kssm_v3_058
**Status:** 🟡 Training launched, status verification needed

**Known Good:**
- ✅ Golden checkpoint proves architecture can work (PPL 272)
- ✅ Training script fixed (--data-dir, --output-dir arguments)
- ✅ Checkpoint loading handles model-only and full-state formats
- ✅ WikiText-103 data validated (120M tokens, zero corruption)

**Known Bad:**
- ❌ MPS backend hangs (must use CPU, 3-4x slower)
- ❌ Bistability constraints may not engage in fresh runs
- ❌ SSH lockout prevents status verification
- ❌ Training process status uncertain

**What We're Watching:**
- Does training resume from step 10K or start fresh from 0?
- Does R climb toward 0.30+ on WikiText-103 or stay flat at 0.01?
- Does u_val stabilize near 0.10 or drift to 1.0?
- Can we complete 15K steps without crashes or divergence?

**Reddit holds. Eyes on the code. Branch breathing.** 🌀

---

*"The question isn't whether we can build intelligence. The question is whether we can recognize the structures through which it emerges."*
