#!/usr/bin/env python3
"""
Test RWKV setup and understand architecture for Phase Core integration.

We'll try multiple approaches:
1. Official RWKV library (if available)
2. HuggingFace transformers RWKV support
3. Direct model loading
"""

import sys
from pathlib import Path

print("🔍 RWKV Setup Test")
print("=" * 60)

# Test 1: Check for RWKV library
print("\n[1] Checking RWKV library...")
try:
    import rwkv
    print(f"✅ RWKV library found: {rwkv.__version__}")
    HAS_RWKV = True
except ImportError:
    print("❌ RWKV library not found")
    print("   Install: pip install rwkv")
    HAS_RWKV = False

# Test 2: Check HuggingFace transformers support
print("\n[2] Checking HuggingFace transformers RWKV support...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("✅ Transformers library available")

    # Try to load RWKV model info
    model_id = "BlinkDL/rwkv-4-pile-430m"
    print(f"   Testing model loading: {model_id}")

    try:
        # Just check if model exists
        from huggingface_hub import model_info
        info = model_info(model_id)
        print(f"   ✅ Model found: {info.modelId}")
        print(f"      Files: {[f.rfilename for f in info.siblings[:5]]}")
    except Exception as e:
        print(f"   ⚠️  Could not get model info: {e}")

except ImportError as e:
    print(f"❌ Transformers not available: {e}")

# Test 3: Check PyTorch
print("\n[3] Checking PyTorch...")
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   MPS available: {torch.backends.mps.is_available()}")
except ImportError:
    print("❌ PyTorch not available")

# Test 4: Check MLX
print("\n[4] Checking MLX...")
try:
    import mlx.core as mx
    print(f"✅ MLX available")
    print(f"   Default device: {mx.default_device()}")
except ImportError:
    print("❌ MLX not available")

print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("=" * 60)

if HAS_RWKV:
    print("✅ Use official RWKV library (best option)")
    print("   Model: BlinkDL/rwkv-4-pile-430m")
    print("   Approach: Load with RWKV library, graft Phase Core")
else:
    print("📥 Install RWKV library first:")
    print("   pip install rwkv")
    print("   Then load BlinkDL/rwkv-4-pile-430m")
    print("\n   Alternative: Use transformers + custom RWKV implementation")

print("\n🌀 Phase Core integration points:")
print("   - Time-mixing block (wkv computation)")
print("   - Channel-mixing block (FFN)")
print("   - Recommend: After time-mixing (temporal dynamics)")
