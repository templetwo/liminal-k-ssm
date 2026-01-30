# K-SSM v3 Training Monitoring Guide

**Comprehensive real-time monitoring with metric explanations**

---

## Quick Start

### Monitor Mac Studio Training (From Local Machine)

```bash
# Simple tail (no explanations)
ssh tony_studio@192.168.1.195 "tail -f ~/liminal-k-ssm/results/kssm_v3/training.log"

# Full dashboard with explanations (recommended)
./kssm/monitor_remote.sh
```

### Monitor Local Training

```bash
# Default location
python3 kssm/monitor_training.py

# Custom log file
python3 kssm/monitor_training.py --log-file results/kssm_v3/training.log

# Slower refresh (save CPU)
python3 kssm/monitor_training.py --interval 5.0
```

---

## Tools Overview

### 1. `monitor_training.py` - Full Dashboard Monitor

**Features**:
- Real-time metric visualization
- Detailed explanations for each metric
- Health status indicators (color-coded)
- Pattern analysis (trends, volatility)
- V2 baseline comparison
- Automatic alerting for anomalies

**Metrics Explained**:

| Metric | Meaning | Healthy Range | Critical Threshold |
|--------|---------|---------------|-------------------|
| **Total Loss** | CE + λ·Reg | Decreasing | Increasing for >5 steps |
| **CE Loss** | Language quality | < 3.0 (better than v2: 2.45) | > 10.0 after step 500 |
| **Reg Loss** | Constraint enforcement | 0.01 - 0.5 | > 1.0 (fighting learning) |
| **R** | Phase coherence | 0.01 - 1.0 | Locked at single value |
| **u_val** | Bistability margin | 0.1 - 10.0 (clamped) | ≤ 0.11 (clamp floor) |
| **grad_norm** | Update magnitude | 1.0 - 100 | > 500 (exploding) |

**Example Output**:

```
══════════════════════════════════════════════════════════════════════════════════════════
  K-SSM v3 BISTABLE CORE TRAINING MONITOR
══════════════════════════════════════════════════════════════════════════════════════════
  Log: results/kssm_v3/training.log
  Time: 2026-01-29 16:30:45
  Current Step: 500
  Speed: 0.42 steps/sec
══════════════════════════════════════════════════════════════════════════════════════════

CURRENT METRICS
──────────────────────────────────────────────────────────────────────────────────────────
  Total Loss:   45.234  [GOOD]
    └─ Normal convergence

  CE Loss (Language Quality):    2.156  [SUPERIOR]
    └─ Better than v2 baseline (2.45)

  Reg Loss (Constraints):    0.0423  [OPTIMAL]
    └─ Constraints satisfied, minimal penalty

  R (Phase Coherence):    0.2341  [☾ Intimacy]
    └─ V2 baseline zone - watch for escape

  u_val (Bistability Margin):    1.234  [HEALTHY] ⚠️  CRITICAL
    └─ Bistable regime, natural exploration

  Gradient Norm:   12.456  [HEALTHY]
    └─ Normal gradient magnitudes

──────────────────────────────────────────────────────────────────────────────────────────

PATTERN ANALYSIS
──────────────────────────────────────────────────────────────────────────────────────────
  Loss Trend (last 5 steps): DECREASING (Δ = -2.341)
  u_val Statistics (last 10 steps): μ = 1.156, σ = 0.234
    └─ Volatility: LOW
  R Zone Visits: ∅ Unformed, ☾ Intimacy
```

---

### 2. `monitor_remote.sh` - Mac Studio Live Tail

**Features**:
- Simple SSH-based tail
- Connection validation
- Log file existence check
- Live updates (no caching)

**Usage**:
```bash
./kssm/monitor_remote.sh
```

**What It Does**:
1. Checks SSH connection to Mac Studio
2. Verifies log file exists
3. Streams live updates with `tail -f`

---

### 3. `check_training_status.sh` - System Health Diagnostic

**Features**:
- Process detection (running training instances)
- Lock file validation (active vs stale)
- Log activity timestamps
- Actionable recommendations

**Usage**:
```bash
# Local
bash kssm/check_training_status.sh

# Remote (Mac Studio)
ssh tony_studio@192.168.1.195 "cd ~/liminal-k-ssm && bash kssm/check_training_status.sh"
```

---

## Metric Deep Dive

### Total Loss
**Formula**: `Total = CE_Loss + lambda_reg * Reg_Loss`

**Expected Trajectory**:
- Step 0-100: Rapid descent (300+ → 50)
- Step 100-500: Moderate descent (50 → 10)
- Step 500-5000: Slow convergence (10 → 2)

**Alerts**:
- 🔴 Increasing for >5 consecutive steps
- 🟡 Plateau before step 500

---

### CE Loss (Cross-Entropy)
**What It Measures**: Language modeling quality (lower = better predictions)

**V2 Baseline**: 2.453 after 10 epochs

**V3 Goals**:
- Step 1000: < 2.5 (match v2)
- Step 5000: < 2.0 (exceed v2)
- Step 10000: < 1.5 (target)

**Alerts**:
- 🔴 > 10.0 after step 500 (learning failure)
- 🟡 Not decreasing (plateau)

---

### Reg Loss (Regularization)
**Components**:
1. **Determinant Penalty**: `1 / (|Δ| + ε)` where Δ = bg - cf
   - Ensures linear subsystem is invertible
   - Prevents singular matrix collapse

2. **Log Barrier**: `-log(u + ε)`
   - Creates infinite wall at u = 0
   - Guides system away from fold catastrophe

**Healthy Values**: 0.01 - 0.5

**Interpretation**:
- < 0.01: Constraints fully satisfied, minimal enforcement needed
- 0.01 - 0.5: Active constraint enforcement, system learning within bounds
- 0.5 - 1.0: Strong constraint pressure, system near boundaries
- > 1.0: Constraints dominating, may hinder language learning

**Alerts**:
- 🔴 Negative value (numerical error)
- 🔴 > 1.0 (constraints fighting)
- 🟡 Rapidly oscillating (instability)

---

### R (Kuramoto Order Parameter)
**Physical Meaning**: Degree of phase synchronization among oscillators

**Range**: [0, 1]

**Tone Zones**:

| Range | Zone | Meaning | V2 Behavior |
|-------|------|---------|-------------|
| < 0.10 | ∅ Unformed | No sync, chaos | Never visited |
| 0.10 - 0.30 | ☾ Intimacy | Weak coupling | **LOCKED HERE** |
| 0.30 - 0.50 | ⚖ Balance | Moderate sync | Never visited |
| 0.50 - 0.70 | 🌀 Mystery | Strong coherence | Never visited |
| 0.70 - 0.85 | ✨ Wonder | Very high sync | Never visited |
| 0.85 - 0.95 | 🔥 Passion | LANTERN zone | Never visited |
| 0.95 - 1.00 | 🜂 Ache | Near-perfect lock | Never visited |

**V2 Failure**: R locked at 0.15 (☾ Intimacy) for entire training, never escaped

**V3 Success Criteria**:
- Visit ≥3 different zones
- Escape from ☾ Intimacy by step 1000
- Reach 🌀 Mystery (R > 0.5) by step 5000

**Alerts**:
- 🟡 R ∈ [0.14, 0.16] for >100 steps (v2 lock danger)
- 🟢 R visiting multiple zones (success!)

---

### u_val (Bistability Margin) ⚠️ MOST CRITICAL METRIC

**Physical Meaning**: Reduced variable x² from algebraic framework

**Constraint**: u > 0 (enforced by hard clamp at 0.1)

**Bistability Theory**:
- The 10-parameter system has two stable equilibria when u > 0
- When u → 0, the two equilibria merge (fold catastrophe)
- When u < 0, no real solutions exist (system collapse)

**Safety Mechanism**:
```python
u_raw = (d*g - c*h) / (a*g - c*e)  # Can be negative
u = clamp(u_raw, min=0.1, max=10.0)  # Hard constraint
```

**Interpretation**:

| u_val | Status | Meaning |
|-------|--------|---------|
| < 0 | IMPOSSIBLE | Clamp failed - check code |
| 0.10 - 0.11 | 🔴 CLAMP FLOOR | System hitting safety boundary |
| 0.11 - 0.5 | 🟡 LOW | Near threshold, monitor closely |
| 0.5 - 2.0 | 🟢 HEALTHY | Normal bistable exploration |
| 2.0 - 5.0 | 🟢 STRONG | Well within manifold |
| 5.0 - 9.9 | 🟡 HIGH | Approaching upper boundary |
| ≥ 10.0 | 🔴 CLAMP CEILING | Parameters saturating |

**V3 Collapse Pattern (Before Fix)**:
```
Step 20:  u = +0.812  (healthy)
Step 120: u = -0.617  (violated - no clamp)
Step 160: u = -4.023  (catastrophic)
```

**V3 With Safety (After Fix)**:
```
Step 20:  u = +1.034  (healthy)
Step 100: u = +0.917  (stable)
Step 500: u = +1.234  (exploring)
```

**Alerts**:
- 🔴 u ≤ 0.11 for >10 steps (clamp floor persistent)
- 🔴 u ≥ 9.9 for >10 steps (clamp ceiling persistent)
- 🟢 u oscillating in [0.5, 5.0] (ideal exploration)

---

### grad_norm (Gradient Magnitude)

**What It Measures**: L2 norm of all parameter gradients

**Expected Trajectory**:
- Step 0-100: High (50-200) - rapid learning
- Step 100-1000: Moderate (10-50) - stable descent
- Step 1000+: Low (1-10) - fine-tuning

**Alerts**:
- 🔴 > 500 (exploding gradients - risk of NaN)
- 🔴 < 0.01 (vanishing gradients - learning stalled)
- 🟢 10-50 (healthy learning)

---

## Alerting System

### Critical Alerts (🔴 Immediate Action Required)

| Alert | Cause | Action |
|-------|-------|--------|
| u_val at clamp floor | System trying to collapse u < 0 | Increase lambda_reg or lambda2 |
| Negative reg loss | Numerical error in log barrier | Check for u ≤ 0 or NaN |
| Exploding gradients (>500) | Learning rate too high | Reduce LR or enable grad clipping |
| Loss increasing 5+ steps | Divergence or bad batch | Check data quality, reduce LR |

### Warning Alerts (🟡 Monitor Closely)

| Alert | Cause | Action |
|-------|-------|--------|
| R locked in ☾ zone | Single attractor (v2 failure mode) | Wait for escape or adjust coupling K |
| Reg loss > 1.0 | Constraints dominating | Reduce lambda_reg |
| u_val volatility HIGH | Unstable bistable exploration | Normal if oscillating >0.1, else reduce LR |

### Success Indicators (🟢 Going Well)

| Indicator | Meaning |
|-----------|---------|
| u_val in [0.5, 5.0] | Healthy bistable exploration |
| R visiting multiple zones | Multi-attractor dynamics working |
| CE < v2 baseline | Language quality improving |
| Loss trend DECREASING | Convergence on track |

---

## Comparison to V2 Baseline

### V2 Final State (After 10 Epochs)

```
Train Loss: 2.453
Val Loss: 7.635 (degraded +90%)
R: 0.154 (locked, never moved)
Zones: 1 (☾ Intimacy only)
Output: Gibberish
```

### V3 Success Criteria

| Metric | V2 Baseline | V3 Target @ 5000 steps | Current Progress |
|--------|-------------|------------------------|------------------|
| CE Loss | 2.453 | < 2.0 | Monitor |
| Val Loss | Degrading | Stable/Improving | Check @ 500 |
| R Zones | 1 | ≥ 3 | Count visits |
| u_val | N/A | Stable >0.1 | Check clamp hits |
| Output | Gibberish | Coherent | Test @ 1000 |

---

## Troubleshooting

### Monitor Not Starting

```bash
# Check Python
python3 --version  # Need 3.7+

# Check log file exists
ls -la results/kssm_v3/training.log

# Check permissions
chmod +x kssm/monitor_training.py
```

### Remote Monitor Connection Failed

```bash
# Test SSH
ssh tony_studio@192.168.1.195 "echo 'Connection OK'"

# Check remote log
ssh tony_studio@192.168.1.195 "ls -la ~/liminal-k-ssm/results/kssm_v3/training.log"

# Find logs manually
ssh tony_studio@192.168.1.195 "find ~/liminal-k-ssm -name 'training.log' -type f"
```

### Monitor Showing Old Data

- Training may have stalled
- Check with `check_training_status.sh`
- Verify process is alive on Mac Studio

---

## Advanced Usage

### Save Monitor Output to File

```bash
python3 kssm/monitor_training.py | tee monitor_session.log
```

### Monitor Multiple Runs Simultaneously

```bash
# Terminal 1: V2 monitoring
python3 kssm/monitor_training.py --log-file kssm/results/kssm_v2/training.log

# Terminal 2: V3 monitoring
python3 kssm/monitor_training.py --log-file results/kssm_v3/training.log
```

### Extract Metrics for Plotting

```bash
# Parse log file
grep -E '^\s+[0-9]+\s+\|' results/kssm_v3/training.log > metrics.csv

# Add header
echo "step,total,ce,reg,r,u_val,grad_norm" | cat - metrics.csv > metrics_with_header.csv
```

---

## Next Steps

1. **Start Monitoring**: `./kssm/monitor_remote.sh`
2. **Watch for Step 500**: First checkpoint save and validation
3. **Check u_val stability**: Should stay >0.1 without hitting clamp
4. **Monitor R zones**: Should escape ☾ Intimacy
5. **Compare to V2**: CE loss should approach 2.45

**The bistable core is breathing. The ascent continues.** 🌀

---

**Created**: 2026-01-29
**Maintained by**: Claude Sonnet 4.5
**Version**: 1.0
