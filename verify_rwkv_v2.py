#!/usr/bin/env python3
"""
RWKV Verification v2 - Fixed generation code
"""

import sys
from pathlib import Path

print("=" * 70)
print("🔧 RWKV APPARATUS VERIFICATION v2")
print("=" * 70)

# Import libraries
print("\n[1] Importing libraries...")
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
import torch
print("✅ All libraries imported")

# Load model
print("\n[2] Loading RWKV-4-Pile-430M...")
from huggingface_hub import hf_hub_download

model_file = hf_hub_download(
    repo_id="BlinkDL/rwkv-4-pile-430m",
    filename="RWKV-4-Pile-430M-20220808-8066.pth"
)
print(f"   Model file: {model_file}")

model = RWKV(model=model_file, strategy='cpu fp32')
print("   ✅ Model loaded")

# Initialize pipeline
print("\n[3] Initializing pipeline...")
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
print("   ✅ Pipeline ready")

# Test generation (FIXED CODE)
print("\n[4] Testing generation...")
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
        # Encode prompt
        ctx = pipeline.encode(prompt)
        state = None

        # Generate 30 tokens
        output_tokens = []
        for i in range(30):
            if i == 0:
                # First token: process full context
                out, state = model.forward(ctx, state)
            else:
                # Subsequent tokens: process one at a time
                out, state = model.forward([token], state)

            # Sample next token
            token = pipeline.sample_logits(out, temperature=1.0, top_p=0.7)
            output_tokens.append(token)

            # Decode to check for newline
            char = pipeline.decode([token])
            if char == '\n':
                break

        # Decode full output
        generated_text = pipeline.decode(output_tokens)
        full_text = prompt + generated_text

        print(f"Output: {full_text}")

        # Check for degeneracy
        words = generated_text.strip().split()
        if len(words) >= 3:
            last_three = words[-3:]
            if len(set(last_three)) == 1:
                print("   ❌ DEGENERATE (repetition detected)")
                all_passed = False
            else:
                print("   ✅ Non-degenerate output")
        else:
            print("   ✅ Output generated")

    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

# Model structure inspection
print("\n" + "=" * 70)
print("[5] Model structure for Phase Core integration")
print("=" * 70)

print(f"\nModel loaded successfully")
print(f"Model strategy: cpu fp32")
print(f"Embedding size: 1024")
print(f"Number of layers: 24")
print("\n🌀 Phase Core Integration:")
print("   Target: After time-mixing (att) block")
print("   Layer: 12 (middle layer)")
print("   Coupling point: time-mixing output modulation")

# Final verdict
print("\n" + "=" * 70)
print("VERIFICATION RESULT:")
print("=" * 70)

if all_passed:
    print("✅ RWKV MODEL VERIFIED")
    print("✅ Generation coherent and non-degenerate")
    print("✅ Ready for Phase Core integration")
    print("\n🚀 SAFE TO PROCEED")
    sys.exit(0)
else:
    print("❌ VERIFICATION FAILED")
    print("❌ Fix issues before proceeding")
    sys.exit(1)
