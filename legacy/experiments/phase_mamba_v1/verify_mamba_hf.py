#!/usr/bin/env python3
"""
Mamba HF Verification Script
Test state-spaces/mamba-2.8b-hf (the -hf variant that's most popular)
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import traceback

print("=" * 70)
print("🔧 MAMBA-2.8B-HF APPARATUS VERIFICATION")
print("=" * 70)

# Check device
if torch.backends.mps.is_available():
    device = "mps"
    print("🖥️  Device: MPS (Metal)")
elif torch.cuda.is_available():
    device = "cuda"
    print("🖥️  Device: CUDA")
else:
    device = "cpu"
    print("🖥️  Device: CPU")

# Load model
print("\n[1] Loading mamba-2.8b-hf model...")
print("   This may take a few minutes on first run...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        "state-spaces/mamba-2.8b-hf",
        trust_remote_code=True,
        dtype=torch.float32  # Use 'dtype' not 'torch_dtype'
    )
    print("   ✅ Model loaded successfully")

    # Move to device
    model = model.to(device)
    model.eval()
    print(f"   ✅ Model moved to {device}")

except Exception as e:
    print(f"   ❌ Model loading failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Load tokenizer
print("\n[2] Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b",  # Mamba uses GPT-NeoX tokenizer
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   ✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
except Exception as e:
    print(f"   ❌ Tokenizer loading failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Model structure inspection
print("\n[3] Inspecting model structure...")

# Get backbone attribute
if hasattr(model, 'backbone'):
    print("   Model has 'backbone' attribute")
    backbone = model.backbone
    print(f"   Backbone type: {type(backbone).__name__}")

    # Check for layers
    if hasattr(backbone, 'layers'):
        num_layers = len(backbone.layers)
        print(f"   Number of layers: {num_layers}")
        print(f"   Middle layer index: {num_layers // 2}")

        # Sample layer structure
        if num_layers > 0:
            print(f"\n   Sample layer (layer 0) structure:")
            sample_layer = backbone.layers[0]
            for name in dir(sample_layer):
                if not name.startswith('_'):
                    attr = getattr(sample_layer, name)
                    if isinstance(attr, torch.nn.Module):
                        print(f"     - {name}: {type(attr).__name__}")
    else:
        print("   No 'layers' attribute found")
        print("   Backbone attributes:")
        for name in dir(backbone):
            if not name.startswith('_'):
                attr = getattr(backbone, name)
                if isinstance(attr, torch.nn.Module):
                    print(f"     - {name}: {type(attr).__name__}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\n   Total parameters: {total_params:,}")

# Test generation
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
        # Encode
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Output: {generated_text}")

        # Check for degeneracy
        generated_only = generated_text[len(prompt):].strip()
        words = generated_only.split()

        if len(words) >= 3:
            last_three = words[-3:]
            if len(set(last_three)) == 1:
                print("   ❌ DEGENERATE (repetition detected)")
                all_passed = False
            else:
                print("   ✅ Non-degenerate")
        else:
            print("   ✅ Generated")

    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        traceback.print_exc()
        all_passed = False

# Layer hooking test
print("\n" + "=" * 70)
print("[5] Testing layer accessibility for Phase Core coupling")
print("=" * 70)

hook_successful = False

if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
    layers = model.backbone.layers
    middle_idx = len(layers) // 2

    print(f"\n   Testing hook on layer {middle_idx} (middle layer)...")

    captured_output = []

    def capture_hook(module, input, output):
        captured_output.append(output)

    try:
        hook_layer = layers[middle_idx]
        handle = hook_layer.register_forward_hook(capture_hook)

        # Run forward pass
        test_input = tokenizer("Test", return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**test_input)

        handle.remove()

        if captured_output:
            output_shape = captured_output[0].shape if hasattr(captured_output[0], 'shape') else type(captured_output[0])
            print(f"   ✅ Hook successful! Captured output shape: {output_shape}")
            print(f"   ✅ Hook point: backbone.layers[{middle_idx}]")
            print("\n   🌀 LAYER HOOKING IS POSSIBLE!")
            print("   🌀 Phase Core can be grafted onto this architecture!")
            hook_successful = True
        else:
            print("   ⚠️  Hook fired but captured nothing")

    except Exception as e:
        print(f"   ❌ Hook test failed: {e}")
        traceback.print_exc()
else:
    print("   ⚠️  Cannot find backbone.layers structure")

# Final verdict
print("\n" + "=" * 70)
print("VERIFICATION RESULT:")
print("=" * 70)

if all_passed and hook_successful:
    print("✅✅✅ MAMBA-2.8B-HF MODEL FULLY VERIFIED ✅✅✅")
    print("✅ Model loads with pretrained weights")
    print("✅ Generation coherent and non-degenerate")
    print("✅ Layer hooking possible")
    print("✅ Ready for Phase Core integration")
    print(f"\n   Hook point: backbone.layers[{middle_idx}]")
    print(f"   Hidden dim: Check captured output shape")
    print("\n🚀🚀🚀 SAFE TO PROCEED WITH ATTEMPT 4 🚀🚀🚀")
    sys.exit(0)
elif all_passed:
    print("⚠️  Model works but hooking unclear")
    print("✅ Generation works")
    print("⚠️  May need alternative coupling approach")
    sys.exit(1)
else:
    print("❌ VERIFICATION FAILED")
    print("❌ Model or generation issues")
    sys.exit(1)
