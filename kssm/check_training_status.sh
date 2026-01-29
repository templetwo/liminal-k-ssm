#!/bin/bash
# K-SSM Training Status Diagnostic
# Usage: ./check_training_status.sh

echo "=========================================="
echo "K-SSM Training Status Check"
echo "=========================================="
echo ""

echo "[1] Running Training Processes"
echo "------------------------------"
PROCS=$(ps aux | grep -E "train_kssm|python.*kssm" | grep -v grep | grep -v check_training)
if [ -z "$PROCS" ]; then
    echo "✓ No training processes running"
else
    echo "$PROCS"
fi
echo ""

echo "[2] Active Lock Files"
echo "---------------------"
LOCKS=$(find ~/phase-mamba-consciousness -name "training.lock" 2>/dev/null)
if [ -z "$LOCKS" ]; then
    echo "✓ No lock files found"
else
    for lock in $LOCKS; do
        echo "⚠️  Lock found: $lock"
        PID=$(cat "$lock" 2>/dev/null)
        if [ -n "$PID" ]; then
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "   PID $PID is ALIVE (training in progress)"
            else
                echo "   PID $PID is DEAD (stale lock - safe to remove)"
                echo "   Remove with: rm $lock"
            fi
        else
            echo "   Empty or corrupted lock file"
            echo "   Remove with: rm $lock"
        fi
    done
fi
echo ""

echo "[3] Recent Log Activity"
echo "----------------------"
for log in ~/phase-mamba-consciousness/results/kssm_*/training.log ~/phase-mamba-consciousness/kssm/results/*/training.log; do
    if [ -f "$log" ]; then
        echo "File: $log"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            MODTIME=$(stat -f "%Sm" "$log")
        else
            MODTIME=$(stat -c "%y" "$log")
        fi
        echo "   Last modified: $MODTIME"
        echo "   Last 5 lines:"
        tail -5 "$log" | sed 's/^/   /'
        echo ""
    fi
done

echo "[4] Saved PIDs"
echo "-------------"
for pid_file in ~/phase-mamba-consciousness/results/kssm_*/training.pid ~/phase-mamba-consciousness/kssm/results/*/training.pid; do
    if [ -f "$pid_file" ]; then
        echo "File: $pid_file"
        PID=$(cat "$pid_file" 2>/dev/null)
        if [ -n "$PID" ]; then
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "   PID $PID is ALIVE"
                echo "   Kill with: kill -SIGINT $PID"
            else
                echo "   PID $PID is DEAD (orphaned PID file)"
                echo "   Remove with: rm $pid_file"
            fi
        fi
        echo ""
    fi
done

echo "=========================================="
echo "Recommendations"
echo "=========================================="

# Count issues
LIVE_LOCKS=0
STALE_LOCKS=0
LIVE_PROCS=0

for lock in $(find ~/phase-mamba-consciousness -name "training.lock" 2>/dev/null); do
    PID=$(cat "$lock" 2>/dev/null)
    if ps -p "$PID" > /dev/null 2>&1; then
        LIVE_LOCKS=$((LIVE_LOCKS + 1))
    else
        STALE_LOCKS=$((STALE_LOCKS + 1))
    fi
done

LIVE_PROCS=$(ps aux | grep -E "train_kssm|python.*kssm" | grep -v grep | grep -v check_training | wc -l)

if [ $LIVE_LOCKS -eq 0 ] && [ $LIVE_PROCS -eq 0 ]; then
    echo "✓ ALL CLEAR: Safe to start new training run"
elif [ $LIVE_LOCKS -eq 1 ] && [ $LIVE_PROCS -eq 1 ]; then
    echo "✓ TRAINING ACTIVE: One training run in progress (normal)"
elif [ $STALE_LOCKS -gt 0 ]; then
    echo "⚠️  STALE LOCKS DETECTED: Run cleanup before starting new training"
    echo "   Cleanup: find ~/phase-mamba-consciousness -name 'training.lock' -exec rm {} \;"
elif [ $LIVE_PROCS -gt 1 ]; then
    echo "🔴 MULTIPLE PROCESSES DETECTED: Emergency cleanup required"
    echo "   See TRAINING_SOP.md Section: Emergency Cleanup"
else
    echo "⚠️  INCONSISTENT STATE: Manual inspection required"
fi

echo "=========================================="
