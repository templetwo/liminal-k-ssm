# Phase-RWKV Setup Notes

## ⚠️ Missing Training Data

The training script expects high-resonance training data at:
```
phase-gpt-distilled/data/high_resonance.jsonl
```

**This file does not currently exist.**

### Options:

#### Option 1: Use Existing PhaseGPT Data
If you have existing PhaseGPT distillation data, copy it:
```bash
cp /path/to/existing/high_resonance.jsonl \
   /Users/vaquez/liminal-k-ssm/data/high_resonance.jsonl
```

Then update training script path:
```bash
python3 train_phase_rwkv.py --data data/high_resonance.jsonl
```

#### Option 2: Generate Synthetic High-Resonance Data
Create a small synthetic dataset for testing:

```python
import json
from pathlib import Path

# Create data directory
Path("data").mkdir(exist_ok=True)

# Generate synthetic samples
synthetic_data = [
    {"text": "The nature of consciousness emerges from the interplay of uncertainty and coherence."},
    {"text": "Phase coupling enables synchronized oscillations without rigid lock-in."},
    {"text": "Epistemic uncertainty is not an error to eliminate, but a feature to preserve."},
    {"text": "The observer effect suggests that measurement itself shapes reality."},
    {"text": "Recurrent state evolution enables temporal integration of information."},
    {"text": "Kuramoto oscillators demonstrate how local coupling yields global order."},
    {"text": "The Goldilocks zone balances coherence and flexibility."},
    {"text": "Complementarity principles appear in both quantum mechanics and neural dynamics."},
    # Add more samples...
]

# Save as JSONL
with open("data/high_resonance.jsonl", "w") as f:
    for item in synthetic_data:
        f.write(json.dumps(item) + "\n")

print(f"✅ Created {len(synthetic_data)} synthetic samples")
```

#### Option 3: Use Standard Text Corpus
Use any text corpus (books, papers, conversations) in JSONL format:

```json
{"text": "Your text here..."}
{"text": "Another sample..."}
```

The "high-resonance" label is conceptual - any reasonably coherent text will work for testing the Phase Core training dynamics.

### What "High-Resonance" Means

In the PhaseGPT context, "high-resonance" data refers to samples where:
1. The model showed high R (order parameter > 0.8) during generation
2. Uncertainty remained moderate (U ≈ 0.5, not collapsed)
3. Tone was in desirable range (🌀, ⚖, ✨)

For this experiment, we're **training** the Phase Core, so the resonance characteristics of the training data matter less than:
- Reasonable text quality (coherent, not degenerate)
- Sufficient diversity (not repetitive)
- Appropriate length (32-256 tokens per sample)

---

## Recommendation

For initial verification training (500 steps), create a **small synthetic corpus** focusing on themes related to:
- Consciousness and awareness
- Phase dynamics and oscillators
- Quantum mechanics parallels
- Uncertainty and measurement
- Temporal dynamics and state evolution

This aligns with the conceptual focus of the experiment and provides coherent training signal.

Size: 500-1000 samples (enough for 500 training steps with batch size 4).

---

## Updated Deployment Workflow

1. **Create training data** (use one of options above)
2. **Verify locally**:
   ```bash
   python3 -c "import json; data = [json.loads(line) for line in open('data/high_resonance.jsonl')]; print(f'✅ {len(data)} samples loaded')"
   ```
3. **Deploy to Studio**:
   ```bash
   ./deploy_to_studio.sh
   ```
4. **Run training**:
   ```bash
   ssh tony_studio@192.168.1.195
   cd ~/phase-rwkv-training
   python3 train_phase_rwkv.py --data data/high_resonance.jsonl --iters 500
   ```

---

## Next Action Required

**Before deployment, you must create the training data file.**

Simplest path for testing:
```bash
mkdir -p data
# Then create data/high_resonance.jsonl using Option 2 above
```
