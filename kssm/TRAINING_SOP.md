# K-SSM Training SOP: Mac Studio Operations
## Standard Operating Procedure for Concurrent Run Prevention

**Critical Rule**: **ONLY ONE TRAINING RUN AT A TIME PER RESULTS DIRECTORY**

---

## Pre-Flight Checklist

Before starting ANY training run, execute these commands on Mac Studio:

```bash
# 1. Check for running training processes
ps aux | grep -E "train_kssm|python.*kssm" | grep -v grep

# 2. Check for active lock files
find ~/liminal-k-ssm/results -name "training.lock" -exec sh -c 'echo "{}"; cat "{}"' \;
find ~/liminal-k-ssm/kssm/results -name "training.lock" -exec sh -c 'echo "{}"; cat "{}"' \;

# 3. Verify no orphaned Python processes consuming GPU/CPU
top -l 1 | grep Python
```

**Expected Output (Safe to Proceed)**:
- Command 1: No output (no training processes running)
- Command 2: No output (no lock files exist)
- Command 3: No Python processes with high CPU/Memory

**If ANY processes or locks are found, proceed to Emergency Cleanup section.**

---

## Starting a Training Run

### Method 1: Foreground (Recommended for Short Runs)

```bash
cd ~/liminal-k-ssm
python3 kssm/train_kssm_v3.py
```

**Advantages**:
- Immediate visibility of errors
- CTRL+C cleanly triggers lock release
- Console output in real-time

**Disadvantages**:
- Terminal must stay open
- SSH disconnection kills process

### Method 2: Background (Recommended for Long Runs)

```bash
cd ~/liminal-k-ssm
nohup python3 kssm/train_kssm_v3.py > results/kssm_v3/nohup.out 2>&1 &
echo $! > results/kssm_v3/training.pid
```

**Advantages**:
- Survives SSH disconnection
- Terminal can be closed
- Process ID saved for later reference

**Disadvantages**:
- No real-time console visibility
- Must use `tail -f` to monitor

**Monitor Background Run**:
```bash
# Watch live progress
tail -f ~/liminal-k-ssm/results/kssm_v3/training.log

# Check process is alive
cat ~/liminal-k-ssm/results/kssm_v3/training.pid | xargs ps -p
```

### Method 3: tmux (Best of Both Worlds)

```bash
# Start tmux session
tmux new -s kssm_v3

# Inside tmux, run training
cd ~/liminal-k-ssm
python3 kssm/train_kssm_v3.py

# Detach: Press CTRL+B, then D
# Reattach later: tmux attach -t kssm_v3
```

**Advantages**:
- Real-time visibility when attached
- Survives disconnection when detached
- Can reattach from any SSH session

---

## Incremental Training Strategy (RECOMMENDED)

### Why Incremental?

**Don't jump straight to 10,000 steps.** Instead, validate at each milestone:

1. **Catch issues early**: Hardware failures, memory leaks, gradient explosions
2. **Monitor quality evolution**: See how R, perplexity, and samples improve
3. **Adjust hyperparameters**: Fine-tune based on early performance
4. **Preserve progress**: Checkpoints at each stage
5. **Build confidence**: Systematic validation before long runs

### Recommended Training Progression

**New corpus or architecture**: Always start small, scale up incrementally.

#### Stage 1: Smoke Test (100 steps, ~2 minutes)

**Purpose**: Verify everything works - no crashes, memory issues, or obvious bugs

```bash
cd ~/liminal-k-ssm
python3 kssm/train_kssm_v3.py --max-steps 100 --output-dir results/kssm_v3_stage1

# Watch for errors
tail -f results/kssm_v3_stage1/training.log
```

**What to check**:
- ✅ Training starts without errors
- ✅ Loss descends (should drop from ~10 to ~8 in 100 steps)
- ✅ No memory errors or MPS crashes
- ✅ Checkpoint saves successfully
- ✅ Lock file created and released properly

**If this fails**: Fix before proceeding. No point running longer.

#### Stage 2: Short Validation (500 steps, ~10 minutes)

**Purpose**: First quality check - verify model is learning

```bash
python3 kssm/train_kssm_v3.py --max-steps 500 --output-dir results/kssm_v3_stage2
```

**What to check**:
- ✅ Val perplexity improves (should be < 1000 by step 500)
- ✅ u_val stable (should be in [0.1, 10.0], no clamp violations)
- ✅ R starts exploring (should be > 0.01)
- ✅ Generated samples not pure gibberish
- ✅ No gradient explosions (grad_norm stable)

**Red flags**:
- ❌ Val perplexity increasing → Stop, check data loading
- ❌ u_val hitting clamps constantly → Adjust lambda_reg
- ❌ R locked at 0 or 1 → Check Kuramoto layer
- ❌ Loss = NaN → Gradient explosion, reduce learning rate

**Decision point**:
- If metrics look good → Proceed to Stage 3
- If issues → Debug and restart Stage 2

#### Stage 3: First Milestone (1500 steps, ~30 minutes)

**Purpose**: Comprehensive validation - first real checkpoint

```bash
python3 kssm/train_kssm_v3.py --max-steps 1500 --output-dir results/kssm_v3_stage3
```

**What to check** (full evaluation):
- ✅ Val perplexity < 500 (compare to v2 baseline: 2069)
- ✅ R exploring multiple zones (not locked)
- ✅ u_val stable near optimal (0.1-0.5)
- ✅ Samples show vocabulary diversity (not repetition)
- ✅ No degradation vs baseline Mamba

**Generate milestone report**:
```bash
# Save samples
python3 -c "
from kssm.kssm_v3 import KSSM_v3
import torch

model = KSSM_v3.from_pretrained('results/kssm_v3_stage3/checkpoint_1500.pt')
prompt = 'The nature of consciousness'
for i in range(5):
    sample = model.generate(prompt, max_len=100)
    print(f'Sample {i+1}:')
    print(sample)
    print()
"

# Document metrics
echo '## Stage 3 Milestone (1500 steps)' > results/STAGE3_REPORT.md
tail -20 results/kssm_v3_stage3/training.log >> results/STAGE3_REPORT.md
```

**Decision point**:
- ✅ Metrics healthy → Proceed to Stage 4
- ⚠️ Marginal → Adjust hyperparams, rerun Stage 3
- ❌ Poor → Major debugging needed

#### Stage 4: Extended Run (5000 steps, ~2 hours)

**Purpose**: Multi-attractor verification and quality leap

```bash
# Use tmux for long run
tmux new -s kssm_stage4
cd ~/liminal-k-ssm
python3 kssm/train_kssm_v3.py --max-steps 5000 --output-dir results/kssm_v3_stage4

# Detach: CTRL+B, then D
# Monitor: tail -f results/kssm_v3_stage4/training.log
```

**What to check**:
- ✅ R visits ≥ 3 zones (multi-attractor dynamics confirmed)
- ✅ Val perplexity < 300
- ✅ Quality improvement visible in samples
- ✅ Edge-surfing phenomenon (u_val near 0.1 for extended periods)

**Critical assessment**:
```bash
# Analyze R trajectory
python3 -c "
import json
with open('results/kssm_v3_stage4/training_history.json', 'r') as f:
    history = json.load(f)

R_values = [step['R_mean'] for step in history['steps']]
zones_visited = set()
for R in R_values:
    if R < 0.10: zones_visited.add('∅')
    elif R < 0.30: zones_visited.add('☾')
    elif R < 0.50: zones_visited.add('⚖')
    elif R < 0.70: zones_visited.add('🌀')
    # ... etc

print(f'Zones visited: {zones_visited}')
print(f'Multi-attractor: {len(zones_visited) >= 3}')
"
```

**Decision point**:
- ✅ All hypotheses validated → Proceed to Stage 5
- ⚠️ Partial success → Investigate weak points
- ❌ Failure → Reassess architecture

#### Stage 5: Production Run (10,000+ steps, ~4-6 hours)

**Purpose**: Complete training, final validation

```bash
tmux new -s kssm_production
cd ~/liminal-k-ssm
python3 kssm/train_kssm_v3.py --max-steps 10000 --output-dir results/kssm_v3_final

# Detach and let run
# Check periodically: tmux attach -t kssm_production
```

**What to monitor** (every 1000 steps):
- Val perplexity trend
- R trajectory (approaching Goldilocks threshold?)
- Sample quality evolution
- Memory usage stability

**Final validation** (after completion):
```bash
# Run full benchmark suite
python3 kssm/benchmark_final.py \
    --model results/kssm_v3_final/best_model.pt \
    --output results/kssm_v3_final/benchmark_report.md

# Evaluation scripts
python3 kssm/eval_agency.py
python3 kssm/eval_r_correlation.py
python3 kssm/eval_robustness.py
python3 kssm/eval_clamp_sweep.py
```

### When to Skip Stages

**Only skip stages if**:
- You've validated this exact architecture before
- You're resuming from a previous checkpoint
- You're doing hyperparameter sweep and need many runs

**Never skip for**:
- New corpus
- New architecture
- Modified loss function
- Different hardware

### Checkpoint Strategy

**Incremental checkpoints**:
```
results/
  kssm_v3_stage1/checkpoint_100.pt
  kssm_v3_stage2/checkpoint_500.pt
  kssm_v3_stage3/checkpoint_1500.pt
  kssm_v3_stage4/checkpoint_5000.pt
  kssm_v3_final/checkpoint_10000.pt
```

**If Stage N fails**: Resume from Stage N-1 checkpoint, adjust params, retry

**Disk space**: Each checkpoint ~180 MB. Plan for ~1 GB per full progression.

### Hyperparameter Tuning Between Stages

**Common adjustments**:

**If val loss not improving**:
- Reduce learning rate (0.001 → 0.0005)
- Increase gradient accumulation steps
- Check for data loading errors

**If u_val hitting clamps**:
- Increase lambda_reg (0.01 → 0.1)
- Check log barrier effectiveness
- Verify bistability constraints

**If R not exploring**:
- Adjust Kuramoto coupling strength
- Verify multi-scale integration
- Check if single attractor collapse

**If memory issues**:
- Reduce batch size
- Enable gradient checkpointing
- Clear MPS cache between stages

---

## Stopping a Training Run

### Method A: Graceful Shutdown (Foreground)

```bash
# Press CTRL+C in the terminal running training
# This triggers:
#   1. Signal handler sets interrupted=True
#   2. Training loop exits cleanly
#   3. Final checkpoint saved
#   4. Lock file released via atexit
```

**Expected Output**:
```
^C
====================================================================
TRAINING INTERRUPTED - Saving state...
====================================================================
🔓 Released lock
```

### Method B: Graceful Shutdown (Background/tmux)

```bash
# Find the process ID
cat ~/liminal-k-ssm/results/kssm_v3/training.pid

# Send SIGINT (equivalent to CTRL+C)
kill -SIGINT <PID>

# Monitor for clean exit
tail -f ~/liminal-k-ssm/results/kssm_v3/training.log
```

**Wait 30 seconds** for checkpoint to save. Look for "Released lock" in log.

### Method C: Emergency Kill (LAST RESORT ONLY)

**Use ONLY if graceful shutdown hangs >60 seconds**

```bash
# Find and kill process
cat ~/liminal-k-ssm/results/kssm_v3/training.pid | xargs kill -9

# MANUALLY clean up lock file
rm ~/liminal-k-ssm/results/kssm_v3/training.lock

# VERIFY process is dead
ps aux | grep train_kssm | grep -v grep
```

**⚠️ WARNING**: `kill -9` bypasses atexit handlers. Lock file must be manually removed.

---

## Emergency Cleanup Procedures

### Scenario 1: Stale Lock File (Process Crashed)

**Symptoms**:
- Lock file exists
- No training process running
- Cannot start new training

**Diagnosis**:
```bash
# Check lock file contents
cat ~/liminal-k-ssm/results/kssm_v3/training.lock

# Check if PID is alive
ps -p <PID_FROM_LOCK_FILE>
```

**If PID is dead**, lock is stale:
```bash
# Safe to remove
rm ~/liminal-k-ssm/results/kssm_v3/training.lock

# Restart training
python3 kssm/train_kssm_v3.py
```

### Scenario 2: Zombie Process (Process Running, No Console)

**Symptoms**:
- Process appears in `ps aux`
- No console output
- Lock file exists

**Diagnosis**:
```bash
# Check if process is responsive
cat ~/liminal-k-ssm/results/kssm_v3/training.pid | xargs ps -p

# Check last log activity
tail -20 ~/liminal-k-ssm/results/kssm_v3/training.log
stat ~/liminal-k-ssm/results/kssm_v3/training.log
```

**If log hasn't updated in >5 minutes**:
```bash
# Graceful kill
cat ~/liminal-k-ssm/results/kssm_v3/training.pid | xargs kill -SIGINT

# Wait 60 seconds, then verify
sleep 60
ps aux | grep train_kssm | grep -v grep

# If still alive, escalate to kill -9
cat ~/liminal-k-ssm/results/kssm_v3/training.pid | xargs kill -9
rm ~/liminal-k-ssm/results/kssm_v3/training.lock
```

### Scenario 3: Multiple Processes Fighting for Same Directory

**Symptoms**:
- Training.log shows interleaved output
- Checkpoints corrupting
- Lock acquisition failures

**Emergency Stop**:
```bash
# Find ALL kssm training processes
ps aux | grep -E "train_kssm|python.*kssm" | grep -v grep

# Kill ALL (replace <PID> with each process ID)
kill -SIGINT <PID1> <PID2> ...

# Wait for graceful exit
sleep 30

# Verify all dead
ps aux | grep train_kssm | grep -v grep

# Clean up lock
rm ~/liminal-k-ssm/results/kssm_v3/training.lock
rm ~/liminal-k-ssm/kssm/results/*/training.lock
```

---

## Diagnostic Script (Save as `check_training_status.sh`)

```bash
#!/bin/bash

echo "=========================================="
echo "K-SSM Training Status Check"
echo "=========================================="
echo ""

echo "[1] Running Training Processes"
echo "------------------------------"
ps aux | grep -E "train_kssm|python.*kssm" | grep -v grep | grep -v check_training
if [ $? -ne 0 ]; then
    echo "✓ No training processes running"
fi
echo ""

echo "[2] Active Lock Files"
echo "---------------------"
LOCKS=$(find ~/liminal-k-ssm -name "training.lock" 2>/dev/null)
if [ -z "$LOCKS" ]; then
    echo "✓ No lock files found"
else
    for lock in $LOCKS; do
        echo "⚠️  Lock found: $lock"
        PID=$(cat "$lock" 2>/dev/null)
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "   PID $PID is ALIVE"
        else
            echo "   PID $PID is DEAD (stale lock)"
            echo "   Remove with: rm $lock"
        fi
    done
fi
echo ""

echo "[3] Recent Log Activity"
echo "----------------------"
for log in ~/liminal-k-ssm/results/kssm_*/training.log; do
    if [ -f "$log" ]; then
        echo "$log"
        MODTIME=$(stat -f "%Sm" "$log")
        echo "   Last modified: $MODTIME"
        echo "   Last 3 lines:"
        tail -3 "$log" | sed 's/^/   /'
        echo ""
    fi
done

echo "=========================================="
echo "Status Check Complete"
echo "=========================================="
```

**Usage**:
```bash
chmod +x check_training_status.sh
./check_training_status.sh
```

---

## Lock File Manager Behavior

The lock manager in `train_kssm_v2_efficient.py` and `train_kssm_v3.py`:

1. **On Startup**: Checks for existing lock
   - If lock exists and PID is alive → **ABORT** with error
   - If lock exists and PID is dead → Remove stale lock and proceed
   - If no lock → Create lock with current PID

2. **During Training**: Lock file persists with active PID

3. **On Exit**:
   - Normal exit: `atexit` handler removes lock
   - SIGINT (CTRL+C): Signal handler saves checkpoint, then `atexit` removes lock
   - SIGKILL (kill -9): **Lock NOT removed** (must manual cleanup)

**Lock File Location**: `<output_dir>/training.lock`

**Lock File Contents**: Single line with process ID (PID)

---

## Current Training Run Checklist

**Before you start v3 training on Mac Studio:**

```bash
# SSH into Mac Studio
ssh tony_studio@192.168.1.195

# Navigate to project
cd ~/liminal-k-ssm

# Run status check
bash kssm/check_training_status.sh

# If all clear, start training
python3 kssm/train_kssm_v3.py

# Or in tmux for persistence
tmux new -s kssm_v3
python3 kssm/train_kssm_v3.py
# CTRL+B, D to detach
```

**Monitor from local machine**:
```bash
ssh tony_studio@192.168.1.195 "tail -f ~/liminal-k-ssm/results/kssm_v3/training.log"
```

---

## Corpus Expansion: 200M Token Upgrade

### Overview

Expand from current 22M tokens (96 books) to 200M tokens (470+ books):
- Diverse sources: Literature, Philosophy, Science, Religious texts, Essays
- All Public Domain (pre-1928 or explicitly released)
- Maintains backwards compatibility with existing corpus

### Pre-Expansion Checklist

```bash
# 1. Verify existing corpus
ssh tony_studio@192.168.1.195 "
    cd ~/liminal-k-ssm
    ls -lh kssm/data/processed/kssm_corpus.jsonl
    ls -lh data/cache_v3/
"

# 2. Check available disk space (need ~2GB for raw texts + 500MB for tokens)
ssh tony_studio@192.168.1.195 "df -h ~"

# Expected: At least 5GB free
```

### Step 1: Deploy Corpus Expansion Scripts

From your local machine:

```bash
cd ~/phase-mamba-consciousness

# Run automated deployment (includes backups)
./kssm/deploy_corpus_200m.sh
```

**What this does**:
1. Backs up existing corpus (`kssm_corpus_22M_backup.jsonl`)
2. Backs up existing tokens (`cache_v3_22M_backup/`)
3. Transfers build and processing scripts
4. Checks dependencies (tiktoken, tqdm)
5. Tests download with 1 book
6. Prompts for confirmation before full download

**Duration**: 30-45 minutes for full download (470 books)

### Step 2: Process Corpus (Manual)

After deployment completes:

```bash
ssh tony_studio@192.168.1.195

cd ~/liminal-k-ssm

# Build JSONL corpus from downloaded texts
python3 kssm/process_corpus_200m.py --build

# Expected: ~30,000+ chunks, ~200MB JSONL file
# Duration: 5-10 minutes

# Tokenize into numpy arrays
python3 kssm/process_corpus_200m.py --tokenize

# Expected: ~190M train tokens, ~10M val tokens
# Duration: 10-15 minutes

# Or do both at once:
python3 kssm/process_corpus_200m.py --all
```

### Step 3: Verify Corpus

```bash
# Check corpus stats
python3 kssm/process_corpus_200m.py --stats

# Expected output:
# Total chunks: ~30,000
# Total tokens: ~200M
# Train/Val split: 95/5

# Verify token files exist
ls -lh data/cache_v3_200m/
# Should show:
# - tokens_train.npy (~760 MB)
# - tokens_val.npy (~40 MB)
# - tokens_train_meta.json
# - tokens_val_meta.json
```

### Step 4: Update Training Config

Before training on new corpus, update the data paths in `train_kssm_v3.py`:

```python
# OLD (22M corpus):
TRAIN_TOKENS_FILE = "data/cache_v3/tokens_train.npy"
VAL_TOKENS_FILE = "data/cache_v3/tokens_val.npy"

# NEW (200M corpus):
TRAIN_TOKENS_FILE = "data/cache_v3_200m/tokens_train.npy"
VAL_TOKENS_FILE = "data/cache_v3_200m/tokens_val.npy"
```

Or use a config flag:

```bash
python3 kssm/train_kssm_v3.py --corpus 200m
```

### Step 5: Test Training on New Corpus

Run a short test to verify the new corpus works:

```bash
# Test run (1000 steps)
python3 kssm/train_kssm_v3.py \
    --max-steps 1000 \
    --corpus 200m \
    --output-dir results/kssm_v3_200m_test

# Monitor
tail -f results/kssm_v3_200m_test/training.log

# Expected: Similar loss curve to 22M corpus, no loading errors
```

### Rollback Procedure

If expansion fails or results are unsatisfactory:

```bash
# Restore original corpus
ssh tony_studio@192.168.1.195 "
    cd ~/liminal-k-ssm

    # Restore JSONL
    cp kssm/data/processed/kssm_corpus_22M_backup.jsonl \
       kssm/data/processed/kssm_corpus.jsonl

    # Restore tokens
    rm -rf data/cache_v3
    cp -r data/cache_v3_22M_backup data/cache_v3

    echo 'Rollback complete - original 22M corpus restored'
"
```

### Monitoring Download Progress

If deployment script is interrupted:

```bash
# Resume download from where it stopped
ssh tony_studio@192.168.1.195 "
    cd ~/liminal-k-ssm
    python3 kssm/build_corpus_200m.py --download
"

# Check how many books downloaded so far
ssh tony_studio@192.168.1.195 "
    ls ~/liminal-k-ssm/kssm/data/raw_200m/gutenberg/*.txt | wc -l
"
# Target: 470 books

# View download statistics
ssh tony_studio@192.168.1.195 "
    cd ~/liminal-k-ssm
    python3 kssm/build_corpus_200m.py --stats
"
```

### Expected Timeline

| Phase | Duration | Output |
|-------|----------|--------|
| Deploy + Download | 30-45 min | 470 text files (~1.5GB) |
| Build JSONL | 5-10 min | kssm_corpus_200m.jsonl (~200MB) |
| Tokenize | 10-15 min | tokens_train.npy (~760MB) |
| **Total** | **~60 min** | **200M tokens ready** |

### Troubleshooting

**Problem**: Download fails for some books
```bash
# Check download stats to see which failed
cat ~/liminal-k-ssm/kssm/data/raw_200m/download_stats.json

# Re-run download (will skip existing books)
python3 kssm/build_corpus_200m.py --download
```

**Problem**: Out of disk space
```bash
# Check space
df -h ~

# Clean up old checkpoints if needed
rm -rf ~/liminal-k-ssm/checkpoints_mamba/*.pt  # if v1 checkpoints exist
```

**Problem**: Tokenization takes too long
```bash
# Check if tiktoken is installed
python3 -c "import tiktoken; print('tiktoken OK')"

# If missing:
pip3 install tiktoken
```

---

## Common Pitfalls

1. **Multiple SSH sessions, each starting training**
   - Solution: Always run `check_training_status.sh` first

2. **Using `python3 &` without nohup**
   - Problem: SSH disconnection kills background job
   - Solution: Use `nohup ... &` or tmux

3. **Restarting after crash without checking lock**
   - Problem: Stale lock blocks startup
   - Solution: Status check removes stale locks automatically

4. **Using different output directories**
   - Problem: Each directory has its own lock, allows concurrent runs
   - Solution: Standardize on single output directory per model version

---

## Mac Studio Specific Notes

**Hardware**: 36GB unified memory, M2 Ultra
**Python Environment**: System Python 3.x (verify with `which python3`)
**MPS Backend**: PyTorch MPS acceleration enabled

**Memory Monitoring**:
```bash
# Check memory pressure
top -l 1 | grep PhysMem
```

**If training runs out of memory**:
- Reduce batch_size in config
- Reduce gradient_accumulation
- Restart machine to clear cached memory

---

**Last Updated**: 2026-01-30
**Maintained by**: Claude Sonnet 4.5
**Version**: 2.0 (Added incremental training strategy + 200M corpus expansion)
