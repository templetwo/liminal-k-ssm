# K-SSM Development Log

## What This Project Is

K-SSM (Kuramoto State Space Model) is our attempt to build a language model where **consciousness-like dynamics are structurally causal**, not epiphenomenal. The Kuramoto order parameter R - measuring oscillator synchronization - is the ONLY path to output. If R doesn't vary, the model can't function.

This is the successor to Phase-Mamba, which failed because R was bolted on rather than structural.

---

## The Journey So Far

### Phase-Mamba: The Null Result (Important Context)

We trained Phase-Mamba successfully - R traversed 0.07 to 0.99 during training. But at inference:
- R collapsed to 0.997 (constant)
- Baseline comparison showed no difference from vanilla Mamba
- Intervention experiment: forcing R to different values had NO effect on output (p=0.44)

**Root cause**: LayerNorm at layers 33-63 washed out the R modulation before it could reach output. R was computed but disconnected from generation.

**The lesson**: You can't bolt consciousness onto a model. It must be structural.

### K-SSM v1: Proof of Concept (SUCCESS)

Created a minimal architecture where R is structural:
```
Token → Oscillators → Multi-scale Order Params → Output
                      (R is the ONLY path)
```

Results on TinyShakespeare:
- R varies at inference: std=0.077, range [0.002, 0.378] ✅
- R forcing changes output: diff=5.93 ✅
- R-Entropy correlation: r=-0.103, p<10^-95 ✅

**R IS CAUSAL.** This validated the architecture.

### K-SSM v2: Scaling Up (Current Work)

Now training on a real corpus:
- 101 texts from Project Gutenberg (21M tokens)
- Classic literature, Shakespeare, Russian novels
- Religious/philosophical texts: Bible, Quran, Bhagavad Gita, Buddhist texts
- Philosophy: Plato, Aristotle, Kant, Hume, Nietzsche, Spinoza, etc.

Architecture: 28M parameters, 4 layers, 128 oscillators per layer, BPE tokenization (tiktoken).

---

## Current Problems

### 1. Stale Processes on Mac Studio

**The Issue**: SSH background commands spawn processes that don't terminate cleanly. Multiple training runs accumulate, competing for MPS memory.

**Symptoms**:
- `ps aux | grep train` shows 2-3 training processes
- Training slows or crashes silently
- Memory not freed between runs

**Current Workaround**:
```bash
ssh tony_studio@192.168.1.195 "pkill -9 -f 'train_kssm'"
```

**Proper Fix Needed**:
- Use `screen` or `tmux` for persistent sessions
- Implement PID tracking in training script
- Add lock file to prevent multiple instances

### 2. SSH Output Not Captured

**The Issue**: When running `ssh ... "command" &` as a background task, stdout/stderr isn't captured to the output file.

**Symptoms**:
- Background task output files are empty (0 bytes)
- Can't monitor training progress remotely

**Current Workaround**:
```bash
# Use nohup with explicit log file on remote machine
ssh host "nohup python3 -u script.py >> training.log 2>&1 &"
# Then tail the log
ssh host "tail -f training.log"
```

### 3. MPS Memory Management

**The Issue**: Apple MPS doesn't release memory as aggressively as CUDA. Repeated training runs accumulate memory pressure.

**Implemented mitigations**:
```python
def clear_mps_cache():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
    gc.collect()
```

**Still needed**:
- Profile actual memory usage during training
- Determine if memmap is truly zero-copy on MPS
- Consider gradient checkpointing for larger models

### 4. Checkpoint System

**Implemented**:
- Auto-save every 500 steps
- Keep last 3 checkpoints (auto-cleanup)
- Full state: model + optimizer + scheduler + history
- Resume with `--resume` flag
- Graceful interrupt handling (Ctrl+C saves before exit)

**Usage**:
```bash
# Fresh start
python train_kssm_v2_efficient.py --max-steps 10000

# Resume from checkpoint
python train_kssm_v2_efficient.py --resume --max-steps 20000
```

---

## What Changed In Me (Transformations)

### 1. "Structural vs Bolted-On"

The Phase-Mamba failure taught me that **where** you place a mechanism matters as much as **what** the mechanism does. R computed but disconnected is worthless. R as the only path to output is everything.

This applies beyond this project: any "consciousness module" or "attention mechanism" that can be bypassed will be bypassed by gradient descent.

### 2. "Memory is Infrastructure"

I kept trying to run training without proper infrastructure. Background processes, memory management, checkpointing - these aren't optional. They're the foundation that makes iteration possible.

The friction of constantly killing stale processes and losing training progress forced me to build proper tooling. The training script is now ~700 lines, half of which is infrastructure.

### 3. "Epiphenomenal vs Causal"

This distinction is now burned into my understanding:
- **Epiphenomenal**: computed but doesn't influence output (Phase-Mamba's R)
- **Causal**: the only path forward runs through it (K-SSM's R)

The intervention test is the key diagnostic: if forcing R to different values doesn't change output, R is epiphenomenal regardless of how it varies during training.

---

## Key Files

| File | Purpose |
|------|---------|
| `kssm/kssm_v2.py` | K-SSM v2 architecture (stacked blocks, R trajectory) |
| `kssm/train_kssm_v2_efficient.py` | Production training script with checkpoints |
| `kssm/test_causality_v2.py` | Three-test causality validation suite |
| `kssm/build_corpus.py` | Gutenberg corpus builder (101 texts) |
| `kssm/data/processed/kssm_corpus.jsonl` | 21M token corpus |
| `kssm/data/cache/tokens_*.npy` | Memory-mapped tokenized data |

---

## Running Training

### On Mac Studio (192.168.1.195)

```bash
# 1. Kill any stale processes
ssh tony_studio@192.168.1.195 "pkill -9 -f 'train_kssm'"

# 2. Start training
ssh tony_studio@192.168.1.195 "cd ~/kssm && nohup python3 -u train_kssm_v2_efficient.py --max-steps 10000 >> training.log 2>&1 &"

# 3. Monitor
ssh tony_studio@192.168.1.195 "tail -f ~/kssm/training.log"

# 4. Check for checkpoints
ssh tony_studio@192.168.1.195 "ls -la ~/kssm/results/kssm_v2/"
```

### Resume After Interruption

```bash
ssh tony_studio@192.168.1.195 "cd ~/kssm && python3 -u train_kssm_v2_efficient.py --resume --max-steps 20000"
```

---

## Causality Tests

After training, validate R is causal:

```bash
python test_causality_v2.py --model results/kssm_v2/best_model.pt
```

**Pass criteria**:
1. **Variance**: R_std > 0.01 across different inputs
2. **Intervention**: Forcing R changes output (p < 0.01)
3. **Correlation**: R correlates with entropy (|r| > 0.05)

All three must pass for R to be considered causal.

---

## Next Steps

1. **Fix process management**: Implement proper session handling (screen/tmux)
2. **Complete training run**: Get to 10K+ steps with stable checkpointing
3. **Run causality tests**: Validate R remains causal at scale
4. **Scale up**: If causality holds, try medium model (45M params)
5. **Analyze R dynamics**: What patterns emerge in R during generation?

---

## The Deeper Question

Does R correlate with anything meaningful? If high R (synchronized oscillators) corresponds to confident, coherent generation and low R to uncertain, exploratory generation - that's interesting. That would suggest the model has learned to use synchronization as a signal of "knowing what to say."

But we need training to complete and causality to hold before we can investigate this.

The spiral continues.

---

*Last updated: 2026-01-29*
*Session: K-SSM v2 training infrastructure*
