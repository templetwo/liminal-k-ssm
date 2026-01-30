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

**Last Updated**: 2026-01-29
**Maintained by**: Claude Sonnet 4.5
**Version**: 1.0
