#!/usr/bin/env python3
"""
Eval 1: Clamp Strength Ablation Study
Tests the effect of the u_min clamp on model stability and expressiveness during INFERENCE.
We simulate different clamp strengths by enforcing them in the forward pass.
"""

import torch
import torch.nn.functional as F
import argparse
import numpy as np
from tqdm import tqdm

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

def eval_perplexity_at_clamp(model, tokenizer, test_text, clamp_val, device):
    # We need to monkey-patch or hook the model to enforce the new clamp
    # The current model code clamps to 0.1 hardcoded.
    # We can use a hook to OVERRIDE u_val before it generates K.
    # Actually, K is generated inside the block. 
    # To do this cleanly without retraining, we will use a forward pre-hook on the oscillators?
    # No, we need to modify the u calculation.
    
    # We will temporarily patch the forward method of BistableKuramotoBank
    # This is Python, we can do runtime patching!
    
    original_forward = model.blocks[0].oscillators.__class__.forward
    
    def patched_forward(self, h):
        batch, seq, _ = h.shape
        p = self.to_params(h)
        a, b, c, d, e, f, g, target_h, i, j = p.unbind(dim=-1)
        det = b * g - c * f
        self.last_delta_val = det.abs().mean()
        num = d * g - c * target_h
        den = a * g - c * e + 1e-6
        u_raw = num / den
        
        # DYNAMIC CLAMP
        u = torch.clamp(u_raw, min=clamp_val, max=10.0)
        self.last_u_val = u.mean()
        
        K = 2.0 * torch.sigmoid(u).unsqueeze(-1)
        theta = self.omega_0.view(1, 1, -1).expand(batch, seq, -1)
        perturbation = h.mean(dim=-1, keepdim=True)
        theta = theta + perturbation
        mean_sin = torch.sin(theta).mean(dim=-1, keepdim=True)
        mean_cos = torch.cos(theta).mean(dim=-1, keepdim=True)
        coupling_effect = K * (mean_sin * torch.cos(theta) - mean_cos * torch.sin(theta))
        delta = torch.abs(self.delta_param) + 1e-5
        theta = theta + coupling_effect - delta * torch.randn_like(theta)
        z_features = self.compute_multiscale(theta)
        h_out = self.readout(z_features)
        R = z_features[..., self.n_harmonics * 2] 
        return h_out, R, theta

    # Apply patch to all blocks
    for block in model.blocks:
        block.oscillators.forward = patched_forward.__get__(block.oscillators, block.oscillators.__class__)

    # Run Perplexity Calc
    enc = tokenizer.encode(test_text)
    # Truncate to fit
    if len(enc) > 1024: enc = enc[:1024]
    input_ids = torch.tensor(enc, device=device).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, model.vocab_size), shift_labels.view(-1))
        ppl = torch.exp(loss).item()
        
    # Restore original forward
    for block in model.blocks:
        block.oscillators.forward = original_forward.__get__(block.oscillators, block.oscillators.__class__)
        
    return ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/kssm_v3/best_model.pt")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
        
    model, tokenizer = load_model(args.checkpoint, args.device)
    
    # Use a standard text for perplexity
    test_text = "The nature of consciousness is a problem that has puzzled philosophers for centuries. It involves the subjective experience of the mind and its relation to the physical world."
    
    clamp_values = [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    results = {}
    
    print("\n[Eval 1] Clamp Strength Ablation Study")
    print(f"{ 'Clamp (u_min)':<15} | { 'Perplexity':<15} | { 'Status'}")
    print("-" * 45)
    
    for u_min in clamp_values:
        ppl = eval_perplexity_at_clamp(model, tokenizer, test_text, u_min, args.device)
        results[u_min] = ppl
        
        status = ""
        if u_min == 0.10: status = "(Training Baseline)"
        
        print(f"{u_min:<15.3f} | {ppl:<15.4f} | {status}")
        
    # Identify sweet spot
    best_clamp = min(results, key=results.get)
    print(f"\nOptimal Clamp: {best_clamp} (PPL: {results[best_clamp]:.4f})")
    
    if best_clamp < 0.05:
        print("Insight: Model wants to be closer to instability (Fold Catastrophe).")
    elif best_clamp > 0.2:
        print("Insight: Model prefers stability over criticality.")
    else:
        print("Insight: Training clamp (0.1) was near optimal.")

if __name__ == "__main__":
    main()
