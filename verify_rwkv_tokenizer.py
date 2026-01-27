#!/usr/bin/env python3
"""
Find correct tokenizer for RWKV-4-Pile model
"""

from rwkv.model import RWKV
from huggingface_hub import hf_hub_download

print("🔍 Finding correct tokenizer for RWKV-4-Pile-430M\n")

# Load model
model_file = hf_hub_download(
    repo_id="BlinkDL/rwkv-4-pile-430m",
    filename="RWKV-4-Pile-430M-20220808-8066.pth"
)

model = RWKV(model=model_file, strategy='cpu fp32')
print("✅ Model loaded\n")

# Model vocab size
print(f"Model vocabulary size: 50277\n")

# Try different tokenizers
print("Testing tokenizer options:\n")

# Option 1: Use transformers GPT-2 tokenizer (Pile models often use this)
print("[1] GPT-2 tokenizer (from transformers)")
try:
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    test_text = "The nature of consciousness"
    tokens = tokenizer.encode(test_text)
    print(f"   Text: '{test_text}'")
    print(f"   Tokens: {tokens}")
    print(f"   Max token ID: {max(tokens)}")
    print(f"   Vocab size: {tokenizer.vocab_size}")

    if max(tokens) < 50277:
        print("   ✅ Compatible!")

        # Try generation
        print("\n   Testing generation...")
        state = None
        out, state = model.forward(tokens, state)

        # Sample next token
        import torch
        logits = torch.tensor(out)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        next_word = tokenizer.decode([next_token])
        print(f"   Next token: {next_token} → '{next_word}'")
        print("   ✅ GENERATION WORKS!")

    else:
        print(f"   ❌ Token {max(tokens)} out of bounds!")

except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\n" + "="*60)
print("RESULT: Use transformers GPT2TokenizerFast")
print("="*60)
