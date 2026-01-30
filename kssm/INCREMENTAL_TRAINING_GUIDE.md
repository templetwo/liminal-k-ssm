# K-SSM Incremental Training Quick Reference

**Golden Rule**: Never jump straight to 10,000 steps. Validate at each milestone.

---

## Training Progression Flowchart

```
START
  ↓
[Stage 1] 100 steps (2 min)
  ↓
  Smoke test passed? ───No──→ Fix bugs, restart Stage 1
  ↓ Yes
[Stage 2] 500 steps (10 min)
  ↓
  Quality check passed? ───No──→ Debug, adjust hyperparams, restart Stage 2
  ↓ Yes
[Stage 3] 1500 steps (30 min)
  ↓
  Milestone metrics healthy? ───No──→ Major debugging or architecture revision
  ↓ Yes
[Stage 4] 5000 steps (2 hours)
  ↓
  Multi-attractor confirmed? ───No──→ Investigate, possibly restart Stage 4
  ↓ Yes
[Stage 5] 10,000 steps (4-6 hours)
  ↓
COMPLETE → Full evaluation suite
```

---

## Quick Command Reference

### Stage 1: Smoke Test (100 steps)
```bash
python3 kssm/train_kssm_v3.py --max-steps 100 --output-dir results/stage1
```
**Pass criteria**: No crashes, loss descending

### Stage 2: Short Validation (500 steps)
```bash
python3 kssm/train_kssm_v3.py --max-steps 500 --output-dir results/stage2
```
**Pass criteria**: Val PPL < 1000, R > 0.01, u_val stable

### Stage 3: First Milestone (1500 steps)
```bash
python3 kssm/train_kssm_v3.py --max-steps 1500 --output-dir results/stage3
```
**Pass criteria**: Val PPL < 500, R exploring, samples coherent

### Stage 4: Extended Run (5000 steps)
```bash
tmux new -s stage4
python3 kssm/train_kssm_v3.py --max-steps 5000 --output-dir results/stage4
# CTRL+B, D to detach
```
**Pass criteria**: ≥3 R zones visited, Val PPL < 300

### Stage 5: Production (10,000 steps)
```bash
tmux new -s production
python3 kssm/train_kssm_v3.py --max-steps 10000 --output-dir results/final
# CTRL+B, D to detach
```
**Pass criteria**: All hypotheses validated, final benchmarks pass

---

## Decision Matrix

| Stage | Metric | Healthy | Warning | Failure |
|-------|--------|---------|---------|---------|
| **1 (100)** | Loss | Descending | Flat | Increasing/NaN |
| **2 (500)** | Val PPL | < 1000 | 1000-2000 | > 2000 |
| | u_val | [0.1, 10] | Near clamps | Hitting clamps |
| | R | > 0.01 | 0.001-0.01 | Locked at 0 |
| **3 (1500)** | Val PPL | < 500 | 500-1000 | > 1000 |
| | R zones | ≥ 2 | 1 | Locked |
| | Samples | Words/phrases | Fragments | Gibberish |
| **4 (5000)** | Val PPL | < 300 | 300-500 | > 500 |
| | R zones | ≥ 3 | 2 | < 2 |
| | u_val | Edge-surfing | Wandering | Locked |
| **5 (10K)** | Goldilocks | R ≥ 0.30 | 0.20-0.30 | < 0.20 |
| | Agency | Present | Fragments | None |

**Actions**:
- **Healthy**: Proceed to next stage
- **Warning**: Investigate, possibly adjust hyperparams and retry current stage
- **Failure**: Stop, debug, restart stage or revise architecture

---

## Checkpoints

| Stage | Checkpoint | Size | What to Save |
|-------|-----------|------|--------------|
| 1 | checkpoint_100.pt | ~180 MB | Quick sanity check |
| 2 | checkpoint_500.pt | ~180 MB | First quality gate |
| 3 | checkpoint_1500.pt | ~180 MB | Milestone baseline |
| 4 | checkpoint_5000.pt | ~180 MB | Multi-attractor confirmed |
| 5 | checkpoint_10000.pt | ~180 MB | Production model |

**Also save**: `best_model.pt` (lowest val loss across all stages)

**Total disk**: ~1 GB for full progression + history

---

## Common Failure Modes

### Stage 1 Failures

**Loss = NaN**:
```bash
# Reduce learning rate
--learning-rate 0.0001  # (default: 0.001)
```

**MPS crash**:
```bash
# Clear cache before run
python3 -c "import torch; torch.mps.empty_cache()"
```

### Stage 2 Failures

**Val PPL > 2000** (degrading like v2):
```bash
# Check data loading
python3 -c "
import numpy as np
tokens = np.load('data/cache_v3/tokens_train.npy', mmap_mode='r')
print(f'Tokens: {len(tokens):,}')
print(f'Sample: {tokens[:100]}')
"
```

**u_val hitting clamps**:
```bash
# Increase log barrier strength
--lambda-reg 0.1  # (default: 0.01)
```

**R locked at 0**:
```bash
# Check Kuramoto layer
# Verify multi-scale order parameter computation
# May indicate architecture bug
```

### Stage 3 Failures

**Val PPL not improving**:
- Reduce learning rate: `--learning-rate 0.0005`
- Increase batch size: `--batch-size 16` (if memory allows)
- Check gradient norms: Should be 1-10, not 0.001 or 1000

**Samples still gibberish**:
- Wait longer (quality often leaps suddenly around step 2000-3000)
- If no improvement by 2000, check tokenizer
- Verify corpus quality

### Stage 4 Failures

**R not exploring** (< 3 zones):
- Check if locked in single attractor
- May need to restart with different random seed
- Possible architecture issue if persistent

**Edge-surfing not observed**:
- System may prefer different u value
- Not necessarily a failure if quality is good
- Monitor if u_val is stable

---

## Time Budget

| Stage | Duration | Can pause? | Recommended method |
|-------|----------|------------|-------------------|
| 1 (100) | 2 min | No | Foreground |
| 2 (500) | 10 min | Maybe | Foreground or tmux |
| 3 (1500) | 30 min | Yes | tmux |
| 4 (5000) | 2 hours | Yes | tmux (background) |
| 5 (10K) | 4-6 hours | Yes | tmux (background) |

**Total time budget** (all stages): ~8 hours

**With failures/retries**: Plan for 12-16 hours across multiple days

---

## Monitoring Commands

```bash
# Watch live training
tail -f results/stageN/training.log

# Check if process alive
ps aux | grep train_kssm

# View last 50 lines
tail -50 results/stageN/training.log

# Monitor in tmux
tmux attach -t stage4

# Check memory
top -l 1 | grep Python
```

---

## Emergency Recovery

**Training crashed mid-stage**:
```bash
# Check for checkpoint
ls -lt results/stageN/checkpoint_*.pt

# Resume from last checkpoint
python3 kssm/train_kssm_v3.py --resume --max-steps [stage_target]
```

**Disk full**:
```bash
# Clean old checkpoints (keep best_model.pt)
rm results/stage*/checkpoint_[0-9]*.pt

# Keep only final checkpoints
find results -name "checkpoint_*.pt" | grep -v best | head -n -3 | xargs rm
```

**Lock file stuck**:
```bash
# Check lock
cat results/stageN/training.lock

# Remove if stale (no process running)
rm results/stageN/training.lock
```

---

## Final Validation Checklist

After Stage 5 completes, run full benchmark:

```bash
# 1. Agency evaluation
python3 kssm/eval_agency.py --model results/final/best_model.pt

# 2. R-confidence correlation
python3 kssm/eval_r_correlation.py --model results/final/best_model.pt

# 3. Antifragility test
python3 kssm/eval_robustness.py --model results/final/best_model.pt

# 4. Clamp ablation
python3 kssm/eval_clamp_sweep.py --model results/final/best_model.pt

# 5. Generate samples
python3 kssm/inference_suite.py --model results/final/best_model.pt \
    --prompts "The nature of consciousness" "I think therefore" "To be or not"

# 6. Comprehensive benchmark
python3 kssm/benchmark_final.py --model results/final/best_model.pt
```

**All tests must pass** before declaring training successful.

---

**Remember**: Incremental validation saves time in the long run. Catching issues at Stage 2 (10 min) is better than discovering them at Stage 5 (6 hours).

**The spiral progresses methodically.** 🌀

*Last Updated: 2026-01-30*
*Version: 1.0*
