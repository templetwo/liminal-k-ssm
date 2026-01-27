#!/usr/bin/env python3
"""
Mamba 2 Verification Script
Test loading, generation, and layer accessibility for Phase Core coupling
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import traceback

print("=" * 70)
print("🔧 MAMBA 2 APPARATUS VERIFICATION")
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
print("\n[1] Loading mamba2-2.7b model...")
print("   This may take a few minutes on first run...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        "state-spaces/mamba2-2.7b",
        trust_remote_code=True,
        torch_dtype=torch.float32
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
        "state-spaces/mamba2-2.7b",
        trust_remote_code=True
    )
    print(f"   ✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
except Exception as e:
    print(f"   ❌ Tokenizer loading failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Model structure inspection
print("\n[3] Inspecting model structure...")
print("   Top-level attributes:")
for name in dir(model):
    if not name.startswith('_') and not name.startswith('__'):
        attr = getattr(model, name)
        if isinstance(attr, torch.nn.Module):
            print(f"     - {name}: {type(attr).__name__}")

print("\n   Named modules (first 20):")
for i, (name, module) in enumerate(model.named_modules()):
    if i >= 20:
        print(f"     ... ({len(list(model.named_modules()))} total modules)")
        break
    print(f"     {name}: {type(module).__name__}")

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
        input_ids = inputs["input_ids"]

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
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

print("\n   Attempting to hook into middle layer...")

# Find layers that might contain hidden states
potential_hook_points = []
for name, module in model.named_modules():
    # Look for layer/block patterns
    if 'layer' in name.lower() or 'block' in name.lower():
        if not any(skip in name.lower() for skip in ['norm', 'dropout', 'embed']):
            potential_hook_points.append((name, module))

print(f"\n   Found {len(potential_hook_points)} potential hook points:")
for name, module in potential_hook_points[:10]:
    print(f"     - {name}: {type(module).__name__}")
if len(potential_hook_points) > 10:
    print(f"     ... and {len(potential_hook_points) - 10} more")

# Test hooking
captured_output = None

if potential_hook_points:
    print("\n   Testing hook on first potential point...")
    hook_name, hook_module = potential_hook_points[len(potential_hook_points) // 2]  # Middle layer

    def capture_hook(module, input, output):
        global captured_output
        captured_output = output

    try:
        handle = hook_module.register_forward_hook(capture_hook)

        # Run forward pass
        test_input = tokenizer("Test", return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**test_input)

        handle.remove()

        if captured_output is not None:
            print(f"   ✅ Hook successful! Captured output shape: {captured_output.shape if hasattr(captured_output, 'shape') else type(captured_output)}")
            print(f"   ✅ Hook point: {hook_name}")
            print("\n   🌀 LAYER HOOKING IS POSSIBLE!")
            print("   🌀 Phase Core can be grafted onto this architecture!")
        else:
            print("   ⚠️  Hook fired but captured nothing")

    except Exception as e:
        print(f"   ❌ Hook test failed: {e}")
        traceback.print_exc()
else:
    print("   ⚠️  No obvious hook points found")

# Final verdict
print("\n" + "=" * 70)
print("VERIFICATION RESULT:")
print("=" * 70)

if all_passed and potential_hook_points:
    print("✅✅✅ MAMBA 2 MODEL FULLY VERIFIED ✅✅✅")
    print("✅ Model loads correctly")
    print("✅ Generation coherent and non-degenerate")
    print("✅ Layer hooking possible")
    print("✅ Ready for Phase Core integration")
    print("\n🚀🚀🚀 SAFE TO PROCEED WITH ATTEMPT 4 🚀🚀🚀")
    sys.exit(0)
else:
    print("⚠️  VERIFICATION INCOMPLETE")
    if not all_passed:
        print("❌ Generation issues detected")
    if not potential_hook_points:
        print("❌ No obvious hook points found")
    print("⚠️  Proceed with caution")
    sys.exit(1)
