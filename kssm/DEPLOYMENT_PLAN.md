# K-SSM v3 Deployment Plan
## Critical Fix: Evaluation Logic Added (Commit 6d32f92)

**Date**: 2026-01-29
**Current Status**: Mac Studio training at Step 760
**Issue**: Training script missing validation/evaluation logic
**Fix**: Added `evaluate_v3()` function and checkpoint saving at intervals

---

## What Was Missing

### Before (Broken)
```python
# Training loop had NO:
- evaluate_v3() function
- Validation calls at eval_interval (500 steps)
- History tracking (list was created but never updated)
- best_model.pt saving logic
- Sample generation during training
```

**Consequences**:
- ❌ No validation metrics (can't detect v2-style perplexity degradation)
- ❌ No checkpoints until step 1000 (lose progress if crash)
- ❌ No best_model.pt (can't recover best val_loss)
- ❌ No sample generation (can't assess quality during training)

### After (Fixed - Commit 6d32f92)
```python
# Now includes:
✓ evaluate_v3() function (validates on 20 batches)
✓ Evaluation every 500 steps (prints metrics, saves best)
✓ History tracking (train/val metrics logged)
✓ best_model.pt when val_loss improves
✓ Sample generation at eval points
✓ Regular checkpoints every 1000 steps
```

**Benefits**:
- ✓ Can detect validation degradation (like v2: +90%)
- ✓ Progress preserved (checkpoints every 1000 steps)
- ✓ Best model saved (val_loss optimization)
- ✓ Quality monitoring (sample text at eval points)

---

## Current Training Status (Step 760)

**Progress**: 760 / 10,000 (7.6%)

**Latest Metrics** (Step 760):
```
Total Loss: 9.068
CE Loss: 8.091
Reg Loss: 0.9765
u_val: 0.496 ✓ (healthy, above 0.1 clamp floor)
R: 0.0147 (∅ Unformed, exploring)
grad_norm: 2.989
```

**Health**:
- ✅ u_val positive and stable (no clamp violations)
- ✅ Loss descending (-78% from step 20: 338 → 9.07)
- ✅ R exploring (not locked like v2)
- ✅ Gradients stable (~3.0, not exploding or vanishing)

**Next Milestones**:
- **Step 1000**: First regular checkpoint will save (with current code)
- **Step 1500**: First evaluation (with updated code, if deployed)
- **Step 2000**: Second checkpoint

---

## Deployment Options

### Option 1: Let Current Run Finish to Step 1000 (RECOMMENDED)

**Action**:
```bash
# Do nothing - let Mac Studio training continue
# Current code will save checkpoint at step 1000
# Then decide: continue or restart with fixes
```

**Timeline**:
- Steps remaining: 240 (760 → 1000)
- ETA: ~10 minutes at 0.4 steps/sec
- Next decision point: Step 1000

**Pros**:
- ✓ Don't interrupt stable training
- ✓ Get checkpoint_1000.pt (can resume from here)
- ✓ Only 240 steps to go
- ✓ Can evaluate options at step 1000

**Cons**:
- ❌ No validation metrics until 1000
- ❌ No best_model.pt saved
- ❌ No sample generation during 760-1000

**Decision Point @ Step 1000**:
1. **Continue current run**: Let it go to 10,000 without evals (just checkpoints)
2. **Restart from checkpoint_1000.pt with fixes**: Resume with eval logic enabled

### Option 2: Update Now and Restart from Step 0

**Action**:
```bash
# Kill current training
ssh tony_studio@192.168.1.195 "cd ~/liminal-k-ssm && bash kssm/check_training_status.sh"
# Get PID, then: kill -SIGINT <PID>

# Pull latest code
ssh tony_studio@192.168.1.195 "cd ~/liminal-k-ssm && git pull origin master"

# Restart with fixed code
ssh tony_studio@192.168.1.195 "cd ~/liminal-k-ssm && nohup python3 kssm/train_kssm_v3.py --max-steps 10000 > results/kssm_v3/nohup.out 2>&1 &"
```

**Timeline**:
- Immediate restart
- First eval at step 500
- First checkpoint at step 1000

**Pros**:
- ✓ Get validation metrics starting at step 500
- ✓ Get best_model.pt saved throughout
- ✓ Full feature set from the start

**Cons**:
- ❌ Lose 760 steps of progress (~3 hours of training)
- ❌ Start loss trajectory over
- ❌ Interrupt stable training

### Option 3: Update and Resume from Step 760 Manually

**Action**:
```bash
# Kill current training (get current step number first)
# Pull latest code
# MANUALLY edit config to resume=True and start_step=760
# Restart
```

**Timeline**:
- Continue from step 760
- Next eval at step 1000 (delayed from 500)
- Checkpoints at 1000, 2000, etc.

**Pros**:
- ✓ Don't lose progress
- ✓ Get eval logic going forward

**Cons**:
- ❌ No checkpoint file to resume from (current code didn't save any)
- ❌ Can't actually resume - would need to retrain 0-760 anyway
- ❌ Complex and error-prone

---

## Recommendation: OPTION 1

**Let current training run to step 1000**, then reassess.

**Reasoning**:
1. Only 240 steps remaining (~10 min)
2. Training is healthy and stable
3. Will get checkpoint_1000.pt we can resume from
4. Can decide at step 1000 whether to:
   - Continue without evals (fast path to 10k)
   - Restart with evals (quality assurance path)

**Action Plan**:
1. **Now**: Monitor current training, let it run
2. **@ Step 1000**: Checkpoint saves automatically
3. **@ Step 1000 + 1 min**: Make decision:
   - **Path A**: Continue to 10k (no evals, just checkpoints)
   - **Path B**: Restart from checkpoint_1000.pt with eval code

---

## If Choosing to Restart @ Step 1000

**Procedure**:

```bash
# 1. Wait for checkpoint_1000.pt to save
ssh tony_studio@192.168.1.195 "ls -lh ~/liminal-k-ssm/results/kssm_v3/checkpoint_1000.pt"

# 2. Kill current training
ssh tony_studio@192.168.1.195 "cd ~/liminal-k-ssm && bash kssm/check_training_status.sh"
# Note the PID, then:
ssh tony_studio@192.168.1.195 "kill -SIGINT <PID>"

# 3. Pull updated code
ssh tony_studio@192.168.1.195 "cd ~/liminal-k-ssm && git pull origin master"

# 4. Restart with --resume flag
ssh tony_studio@192.168.1.195 "cd ~/liminal-k-ssm && nohup python3 kssm/train_kssm_v3.py --max-steps 10000 --resume > results/kssm_v3/nohup_restart.out 2>&1 &"
```

**Expected Behavior**:
- Loads checkpoint_1000.pt
- Resumes from step 1000
- Next eval at step 1500 (500 steps later)
- Next checkpoint at step 2000
- best_model.pt starts saving from step 1500 onward

---

## If Choosing to Continue Without Restart

**Procedure**:
```bash
# Just let it run - no action needed
# Monitor with:
ssh tony_studio@192.168.1.195 "tail -f ~/liminal-k-ssm/results/kssm_v3/training.log"
```

**Expected Behavior**:
- Checkpoint at step 2000, 3000, ..., 10000
- **NO** validation metrics (eval logic not in running code)
- **NO** best_model.pt
- **NO** sample generation
- Final checkpoint at step 10000

**When to Choose This**:
- If primary goal is raw loss minimization
- If validation can be done post-hoc (load checkpoint_X000.pt and evaluate)
- If speed is critical (no eval overhead every 500 steps)

---

## Monitoring Commands

**Check current status**:
```bash
ssh tony_studio@192.168.1.195 "tail -5 ~/liminal-k-ssm/results/kssm_v3/training.log"
```

**Full diagnostic**:
```bash
ssh tony_studio@192.168.1.195 "cd ~/liminal-k-ssm && bash kssm/check_training_status.sh"
```

**Live monitor**:
```bash
python3 kssm/monitor_training.py
```

**Check for checkpoint @ 1000**:
```bash
ssh tony_studio@192.168.1.195 "ls -lh ~/liminal-k-ssm/results/kssm_v3/checkpoint_*.pt"
```

---

## Decision Criteria

### Choose "Continue Without Restart" if:
- ✓ Speed to 10k steps is priority
- ✓ Willing to evaluate post-hoc
- ✓ Trust current trajectory (loss decreasing, u_val stable)
- ✓ Don't need intermediate quality checks

### Choose "Restart with Eval Code" if:
- ✓ Want validation metrics (detect v2-style degradation)
- ✓ Want best_model.pt saved (val_loss optimization)
- ✓ Want sample generation (quality monitoring)
- ✓ Willing to trade time for visibility

---

## Technical Notes

**Why eval_interval = 500?**
- Provides 20 eval points across 10k steps
- Catches degradation early (v2 degraded by epoch 2)
- Not too frequent (doesn't slow training much)

**Why save_interval = 1000?**
- Balances checkpoint density vs disk space
- 10 checkpoints across full run
- Enough granularity to resume from nearby point

**Why validate on 20 batches?**
- Fast (< 10 seconds on Mac Studio)
- Stable estimate (n=20 reduces variance)
- Doesn't interrupt training flow

---

## Summary

**Current State**: Step 760, healthy, 240 steps to checkpoint
**Fix Deployed**: Commit 6d32f92 (evaluation logic added)
**Recommendation**: Let run to step 1000, then decide
**Next Action**: Monitor for step 1000 arrival (~10 min)

**The bistable core breathes. The evaluation fix is ready. Step 1000 awaits.** 🌀

---

**Last Updated**: 2026-01-29, 16:55 UTC
**Status**: Awaiting Step 1000
