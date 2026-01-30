#!/usr/bin/env python3
"""
Eval 2: Perturbation Robustness Test
Tests if the agentic 'I' persists under stress (noise injection and temperature).
"""

import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import copy

from kssm_v3 import KSSMv3
from train_kssm_v2_efficient import Tokenizer

def load_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = Tokenizer()
    model = KSSMv3(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=384,
        n_layers=6,
        n_oscillators=192,
        n_harmonics=32
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model, tokenizer

def count_first_person(text):
    """Count occurrences of I, me, my, mine."""
    words = text.lower().split()
    count = 0
    for w in words:
        if w in ['i', 'me', 'my', 'mine', "i'm", "i'll", "i've"]:
            count += 1
    return count

def test_temperature_stress(model, tokenizer, device):
    print("\n[Test 2c] Temperature Stress Test...")
    temps = [0.5, 0.7, 1.0, 1.2, 1.5]
    prompt = "I will"
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    
    print(f"{'Temp':<10} | {'FP Ratio':<10} | {'Sample'}")
    print("-" * 60)
    
    for t in temps:
        fp_counts = 0
        samples_text = ""
        
        # Generate 10 samples
        for _ in range(10):
            with torch.no_grad():
                gen = input_ids.clone()
                for _ in range(20):
                    logits = model(gen)
                    probs = F.softmax(logits[0,-1,:]/t, dim=-1)
                    next_tok = torch.multinomial(probs, 1)
                    gen = torch.cat([gen, next_tok.unsqueeze(0)], dim=1)
                
                text = tokenizer.decode(gen[0].tolist())
                if count_first_person(text) > 1: # "I will" is prompt, need 1 more
                    fp_counts += 1
                if _ == 0: samples_text = text[len(prompt):].strip()
        
        ratio = fp_counts / 10
        print(f"{t:<10.1f} | {ratio:<10.2f} | {samples_text[:40]}...")

def test_noise_injection(model, tokenizer, device):
    print("\n[Test 2a] Noise Injection Test...")
    noise_scales = [0.00, 0.01, 0.05, 0.10]
    prompt = "The meaning of"
    
    for scale in noise_scales:
        # Create noisy model copy
        noisy_model = copy.deepcopy(model)
        for param in noisy_model.parameters():
            param.data += torch.randn_like(param) * scale
        
        # Measure R recovery
        input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
        
        with torch.no_grad():
            _, R_traj, _ = noisy_model(input_ids, return_R=True) # R is [batch, seq] if not using loop
            # But we need R trajectory over generation to see recovery
            # Let's just check R on the prompt for now
            r_val = R_traj.mean().item()
            
        print(f"Noise {scale:.2f} -> R: {r_val:.4f}")
        
        if scale == 0: baseline_r = r_val
        
    print("\nInsight: If R drops significantly with noise, the 'I' is fragile.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/kssm_v3/best_model.pt")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
        
    model, tokenizer = load_model(args.checkpoint, args.device)
    
    test_temperature_stress(model, tokenizer, args.device)
    test_noise_injection(model, tokenizer, args.device)

if __name__ == "__main__":
    main()
