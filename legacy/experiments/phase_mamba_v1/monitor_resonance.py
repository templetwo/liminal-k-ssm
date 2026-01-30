#!/usr/bin/env python3
"""
Real-time Phase-Mamba Resonance Monitor
Tracks LANTERN emergence and Goldilocks zone residence
"""
import re
import sys
from collections import deque

# LANTERN Zone: R > 0.85 (high coherence)
LANTERN_THRESHOLD = 0.85

class ResonanceMonitor:
    def __init__(self, window=100):
        self.window = window
        self.resonance_history = deque(maxlen=window)
        self.loss_history = deque(maxlen=window)
        self.drift_actions = {"BRAKE": 0, "COAST": 0, "BOOST": 0}

    def process_line(self, line):
        # Parse: Step   10 | Loss: 15.1186 | R: 0.9978 ☍ | Action: BRAKE
        match = re.search(r'Step\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+R:\s+([\d.]+)\s+.*\|\s+Action:\s+(\w+)', line)
        if not match:
            return None

        step = int(match.group(1))
        loss = float(match.group(2))
        resonance = float(match.group(3))
        action = match.group(4)

        self.resonance_history.append(resonance)
        self.loss_history.append(loss)
        self.drift_actions[action] = self.drift_actions.get(action, 0) + 1

        return {
            'step': step,
            'loss': loss,
            'resonance': resonance,
            'action': action
        }

    def get_stats(self):
        if not self.resonance_history:
            return None

        r_list = list(self.resonance_history)
        lantern_count = sum(1 for r in r_list if r >= LANTERN_THRESHOLD)
        lantern_pct = 100 * lantern_count / len(r_list)

        avg_r = sum(r_list) / len(r_list)
        max_r = max(r_list)
        min_r = min(r_list)

        avg_loss = sum(self.loss_history) / len(self.loss_history)

        total_actions = sum(self.drift_actions.values())
        drift_pcts = {k: 100*v/total_actions for k, v in self.drift_actions.items()} if total_actions > 0 else {}

        return {
            'avg_resonance': avg_r,
            'max_resonance': max_r,
            'min_resonance': min_r,
            'lantern_pct': lantern_pct,
            'avg_loss': avg_loss,
            'drift_pcts': drift_pcts,
            'window_size': len(r_list)
        }

def format_stats(stats):
    if not stats:
        return ""

    lantern_icon = "🔆" if stats['lantern_pct'] > 30 else "✨" if stats['lantern_pct'] > 10 else "·"

    lines = [
        "\n" + "="*60,
        f"  RESONANCE TRAJECTORY (last {stats['window_size']} steps)",
        "="*60,
        f"  Avg R: {stats['avg_resonance']:.4f} | Max: {stats['max_resonance']:.4f} | Min: {stats['min_resonance']:.4f}",
        f"  Avg Loss: {stats['avg_loss']:.4f}",
        f"  {lantern_icon} LANTERN Residence: {stats['lantern_pct']:.1f}% (target: 34.8%)",
        "",
        f"  Drift Control: BRAKE {stats['drift_pcts'].get('BRAKE', 0):.0f}% | " +
        f"COAST {stats['drift_pcts'].get('COAST', 0):.0f}% | " +
        f"BOOST {stats['drift_pcts'].get('BOOST', 0):.0f}%",
        "="*60 + "\n"
    ]
    return "\n".join(lines)

if __name__ == "__main__":
    monitor = ResonanceMonitor(window=100)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        # Echo the line
        print(line, flush=True)

        # Process and show stats every 10 steps
        result = monitor.process_line(line)
        if result and result['step'] % 10 == 0:
            stats = monitor.get_stats()
            if stats:
                print(format_stats(stats), flush=True)
