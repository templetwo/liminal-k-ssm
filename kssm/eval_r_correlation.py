#!/usr/bin/env python3
"""
Eval 4: R-Quality Correlation & Intervention
Tests if R is causally related to quality by ARTIFICIALLY clamping R during generation.
"""

import torch
import torch.nn.functional as F
import argparse
import numpy as np

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

def eval_quality_at_r(model, tokenizer, target_r, device):
    # Patch forward to clamp R (magnitude of first harmonic)
    # R is computed at end of BistableKuramotoBank.forward
    
    original_forward = model.blocks[0].oscillators.__class__.forward
    
    def patched_forward(self, h):
        # Run standard forward logic
        batch, seq, _ = h.shape
        p = self.to_params(h)
        a, b, c, d, e, f, g, target_h, i, j = p.unbind(dim=-1)
        det = b * g - c * f
        self.last_delta_val = det.abs().mean()
        num = d * g - c * target_h
        den = a * g - c * e + 1e-6
        u_raw = num / den
        u = torch.clamp(u_raw, min=0.1, max=10.0)
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
        
        # INTERVENTION: Scale z_features to match target R
        # R is z_features[..., 2*32] (index 64 if n=32)
        # But wait, compute_multiscale returns [cos, sin, mag] for each n
        # It flattens to [batch, seq, n_harm * 3]
        # KSSMv3 R index is self.n_harmonics * 2 (index 64)
        
        # We want to scale the MAGNITUDE of the 1st harmonic to target_r
        # The features are [cos1, sin1, mag1, cos2, sin2, mag2...]
        # We can scale the entire feature vector, but that's crude.
        # Let's scale just the 1st harmonic features (cos1, sin1) to achieve mag1 = target_r
        
        current_r = z_features[..., 2] # mag_1
        scale_factor = target_r / (current_r + 1e-9)
        
        # Apply scale to first 3 features (cos1, sin1, mag1)
        z_features[..., 0:3] *= scale_factor.unsqueeze(-1)
        
        h_out = self.readout(z_features)
        
        # Return modified R (target_r)
        R = z_features[..., 2] 
        return h_out, R, theta

    # Apply patch
    for block in model.blocks:
        block.oscillators.forward = patched_forward.__get__(block.oscillators, block.oscillators.__class__)

    # Measure Perplexity
    test_text = "The nature of consciousness is a problem that has puzzled philosophers for centuries."
    enc = tokenizer.encode(test_text)
    input_ids = torch.tensor(enc, device=device).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, model.vocab_size), shift_labels.view(-1))
        ppl = torch.exp(loss).item()
        
    # Restore
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
    
    target_r_values = [0.05, 0.15, 0.25, 0.35, 0.50, 0.70, 0.90]
    
    print("\n[Eval 4] R-Quality Intervention Study")
    print(f"{ 'Forced R':<10} | { 'Perplexity':<15}")
    print("-" * 30)
    
    ppls = []
    for r in target_r_values:
        ppl = eval_quality_at_r(model, tokenizer, r, args.device)
        ppls.append(ppl)
        print(f"{r:<10.2f} | {ppl:<15.4f}")
        
    # Analysis
    best_r = target_r_values[np.argmin(ppls)]
    print(f"\nOptimal R: {best_r:.2f} (PPL: {min(ppls):.4f})")
    
    if 0.3 <= best_r <= 0.5:
        print("✅ PASS: Optimal R aligns with Goldilocks/Balance zone.")
    elif best_r > 0.8:
        print("⚠️ WARN: Model prefers hyper-synchronization (Collapse).")
    elif best_r < 0.1:
        print("⚠️ WARN: Model prefers noise (Unformed).")

if __name__ == "__main__":
    main()
