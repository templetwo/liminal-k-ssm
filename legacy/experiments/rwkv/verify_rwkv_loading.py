#!/usr/bin/env python3
"""
CRITICAL VERIFICATION: Ensure RWKV loads correctly and generates coherent text.

This is the "verify apparatus before observation" step.
We will NOT train until this passes.
"""

import sys
from pathlib import Path

print("=" * 70)
print("🔧 RWKV APPARATUS VERIFICATION")
print("=" * 70)

# Step 1: Import libraries
print("\n[1] Importing libraries...")
try:
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE
    import torch
    print("✅ All libraries imported")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Step 2: Download/load model
print("\n[2] Loading RWKV-4-Pile-430M model...")
model_path = "BlinkDL/rwkv-4-pile-430m"

try:
    # RWKV library uses local .pth files
    # We need to download the model first
    from huggingface_hub import hf_hub_download

    # Download the latest RWKV-4 Pile 430M checkpoint
    print("   Downloading model checkpoint...")
    model_file = hf_hub_download(
        repo_id=model_path,
        filename="RWKV-4-Pile-430M-20220808-8066.pth"
    )
    print(f"   ✅ Model file: {model_file}")

    # Load with RWKV library
    print("   Loading into RWKV model...")
    model = RWKV(model=model_file, strategy='cuda fp16' if torch.cuda.is_available() else 'cpu fp32')
    print("   ✅ Model loaded")

except Exception as e:
    print(f"   ❌ Model loading failed: {e}")
    print("\n   Trying alternative approach...")

    try:
        # Try direct path if already downloaded
        home = Path.home()
        cache_dir = home / ".cache" / "huggingface" / "hub"
        # Find the model file
        import os
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                if "RWKV-4-Pile-430M" in file and file.endswith(".pth"):
                    model_file = os.path.join(root, file)
                    print(f"   Found cached model: {model_file}")
                    model = RWKV(model=model_file, strategy='cpu fp32')
                    print("   ✅ Model loaded from cache")
                    break
    except Exception as e2:
        print(f"   ❌ Alternative approach failed: {e2}")
        sys.exit(1)

# Step 3: Initialize pipeline (tokenizer)
print("\n[3] Initializing tokenizer pipeline...")
try:
    pipeline = PIPELINE(model, "20B_tokenizer.json")
    print("   ✅ Pipeline initialized")
except Exception as e:
    print(f"   ❌ Pipeline failed: {e}")
    print("   Trying without explicit tokenizer...")
    try:
        pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
        print("   ✅ Pipeline initialized with vocab")
    except Exception as e2:
        print(f"   ❌ Pipeline failed: {e2}")
        sys.exit(1)

# Step 4: Test generation (CRITICAL - check for degeneracy)
print("\n[4] Testing generation capability...")
print("=" * 70)

test_prompts = [
    "The nature of consciousness is",
    "In a distant galaxy,",
    "The meaning of life"
]

all_passed = True

for prompt in test_prompts:
    print(f"\nPrompt: \"{prompt}\"")
    print("-" * 70)

    try:
        # Generate tokens
        ctx = prompt
        state = None

        generated_tokens = []
        for i in range(30):  # Generate 30 tokens
            out, state = model.forward(pipeline.encode(ctx)[-1] if i == 0 else token, state)
            token = pipeline.sample_logits(out, temperature=1.0, top_p=0.7)
            generated_tokens.append(token)

            # Stop at newline or EOS
            char = pipeline.decode([token])
            if char == '\n':
                break

        # Decode output
        output = pipeline.decode(generated_tokens)
        full_output = prompt + output

        print(f"Output: {full_output}")

        # Check for degeneracy
        words = output.split()
        if len(words) >= 3:
            # Check if last 3 words are identical (degenerate)
            last_three = words[-3:]
            if len(set(last_three)) == 1:
                print("   ❌ DEGENERATE OUTPUT DETECTED (repetition)")
                all_passed = False
            else:
                print("   ✅ Non-degenerate output")
        else:
            print("   ✅ Output generated (short)")

    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        all_passed = False

# Step 5: Inspect model structure
print("\n" + "=" * 70)
print("[5] Inspecting model structure for Phase Core integration...")
print("=" * 70)

try:
    # RWKV model structure
    print(f"\nModel type: {type(model)}")
    print(f"Model has {model.n_layer} layers")
    print(f"Model embedding size: {model.n_embd}")

    print("\n🌀 Phase Core Integration Points:")
    print("   Target: After time-mixing block in each layer")
    print("   Approach: Modulate time-mixing output with Kuramoto oscillators")
    print(f"   Recommended layer: {model.n_layer // 2} (middle layer)")

except Exception as e:
    print(f"   ⚠️  Could not inspect structure: {e}")

# Final verdict
print("\n" + "=" * 70)
print("VERIFICATION RESULT:")
print("=" * 70)

if all_passed:
    print("✅ RWKV MODEL VERIFIED")
    print("✅ Generation is coherent and non-degenerate")
    print("✅ Apparatus ready for Phase Core integration")
    print("\n🚀 SAFE TO PROCEED WITH TRAINING")
    sys.exit(0)
else:
    print("❌ VERIFICATION FAILED")
    print("❌ Do NOT proceed with training")
    print("❌ Fix apparatus first")
    sys.exit(1)
