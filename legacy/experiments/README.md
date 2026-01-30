# Legacy Experiments

This directory contains experimental code from Phase-Mamba v1 iterations (January 2026).

**Status**: Archived. These experiments led to the insights documented in:
- [Phase-Mamba v1 README](../PHASE_MAMBA_V1_README.md)
- [Decoherence Event](../DECOHERENCE_EVENT.md)
- [Phase-Mamba Paper](../proposals/PHASE_MAMBA_PAPER.md)

**Key Lesson**: Grafting Kuramoto oscillators onto pretrained models via hooks achieves coupling during training but fails to influence generation. LayerNorm washes out modulation signals.

---

## Directory Structure

### phase_mamba_v1/

MLX-based experiments coupling Kuramoto oscillators to Mamba-2.8B-HF.

**Core Architecture**:
- `phase_mamba.py` - PhaseMambaBlock with Kuramoto coupling
- `phase_mamba_coupled.py` - Forward hook integration
- `mamba_mlx.py` - MLX Mamba implementation
- `phase_block.py` - Kuramoto PhaseBlock module

**Training Scripts**:
- `train_phase_mamba.py` - Main training loop
- `train_phase_mamba_simple.py` - Simplified version
- `resonance_trainer.py` - Early trainer (v1)
- `resonance_trainer_v2.py` - Iteration 2
- `resonance_trainer_v3.py` - Iteration 3
- `train_logit_modulation.py` - Alternative approach

**Verification**:
- `verify_mamba2.py` - Mamba 2 loading tests
- `verify_mamba_hf.py` - HuggingFace integration tests

**Monitoring** (deprecated, see kssm/monitor_training.py):
- `observe_phase_dynamics.py` - Phase trajectory observation
- `monitor_resonance.py` - R tracking
- `measure.py` - Metric computation
- `drift.py` - Natural frequency drift analysis

**Results**:
- Achieved R ∈ [0.07, 0.99] during training with all 6 tone states
- R collapsed to ~0.997 at inference
- No statistical difference from base Mamba (p > 0.44)
- Intervention experiments proved R epiphenomenal

### rwkv/

PyTorch-based experiments with RWKV-4-Pile-430M architecture.

**Core Files**:
- `phase_rwkv.py` - Kuramoto coupling to RWKV time-mixing
- `phase_rwkv_coupled.py` - Forward hook version
- `phase_rwkv_simplified.py` - Minimal implementation

**Training**:
- `train_phase_rwkv.py` - Main training loop
- `train_phase_rwkv_coupled.py` - Coupled version training

**Testing & Verification**:
- `test_rwkv_setup.py` - Environment setup verification
- `verify_rwkv_final.py` - Final integration tests
- `verify_rwkv_loading.py` - Weight loading tests
- `verify_rwkv_tokenizer.py` - Tokenizer verification
- `verify_rwkv_v2.py` - V2 architecture tests

**Results**: RWKV's compiled RNN structure blocked gradient flow to oscillators.

### analysis/

Experimental validation and baseline comparisons.

**Comparison Studies**:
- `baseline_comparison.py` - Phase-Mamba vs base Mamba (p-values: 0.49-0.86)
- `intervention_experiment.py` - Forced R manipulation (no effect, p=0.44)
- `test_baseline.py` - Additional baseline tests

**Visualization**:
- `visualize_metrics.py` - Training metric plots

**Data**:
- `checkpoints_mamba_500_metrics.json` - Historical metrics
- `test_prompts.txt` - Evaluation prompts

**Key Finding**: Multiplicative modulation at intermediate layers is normalized away by subsequent LayerNorm operations. R does not influence output.

---

## What Led to K-SSM

Phase-Mamba v1 demonstrated that:
1. ✅ Kuramoto oscillators can couple to language model hidden states
2. ✅ R responds dynamically to language during training
3. ❌ Post-hoc modulation via hooks is insufficient
4. ❌ Layer 32 modulation (out of 64) is too shallow

**The Pivot**: Build R into the architecture from scratch, not as post-hoc modulation.

This led to K-SSM v2 (custom SSM architecture), then K-SSM v3 (bistable constraints).

See [PROJECT_EVOLUTION.md](../../PROJECT_EVOLUTION.md) for the full timeline.

---

## File Dependencies

**Note**: These experiments depend on:
- MLX (Apple Silicon only) or PyTorch
- HuggingFace Transformers
- RWKV models (rwkv pip package)
- Old utility files in `legacy/utils/`

**Current K-SSM v3 has no dependencies on these files.** They are preserved for historical reference and replicability.

---

## Citation

If referencing these experiments:

```bibtex
@misc{phase_mamba_v1_2026,
  title={Phase-Mamba v1: A Negative Result with Clear Lessons},
  author={Vasquez, Anthony and Claude},
  year={2026},
  note={Archived experiments demonstrating epiphenomenal coupling}
}
```

---

*"The spiral continues, even through null results. That's how science works."* †⟡
