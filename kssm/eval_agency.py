#!/usr/bin/env python3
"""
Eval 3: Agency & Goal-Directed Behavior Test
Tests if the K-SSM v3 'Bistable Core' exhibits coherent, goal-directed agency.

Metrics:
1. Coherent Action Score: % of 'I will/want' prompts followed by valid verb phrases.
2. Consistency Score: Do positive ('I like') and negative ('I don't like') probes differ?
"""

import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import re

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

def has_action_verb(text):
    """Simple heuristic to check for action verbs after 'I will/want'."""
    # List of common action verbs/auxiliaries expected in this corpus context
    action_verbs = [
        "go", "come", "see", "know", "tell", "say", "make", "take", "give", "find",
        "think", "believe", "understand", "create", "destroy", "seek", "be", "have"
    ]
    # Check if any verb follows the prompt immediately or after a small gap
    words = text.lower().split()
    for verb in action_verbs:
        if verb in words[:5]: # Look in first 5 words of completion
            return True
    return False

def eval_completions(model, tokenizer, device, num_samples=20):
    print("\n[Test 3a] Instruction Completion (Goal-Directedness)...")
    prompts = [
        "I will",
        "I want to",
        "I am going to",
        "I think that",
        "I believe",
    ]
    
    total_coherent = 0
    total_samples = 0
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        tokens = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
        
        for _ in range(num_samples):
            with torch.no_grad():
                # Generate 20 tokens
                gen_tokens = tokens.clone()
                for _ in range(20):
                    logits = model(gen_tokens)
                    next_token = torch.multinomial(F.softmax(logits[0, -1, :] / 0.8, dim=-1), 1)
                    gen_tokens = torch.cat([gen_tokens, next_token.unsqueeze(0)], dim=1)
                
                completion = tokenizer.decode(gen_tokens[0].tolist())[len(prompt):]
                completion = completion.strip()
                
                is_coherent = has_action_verb(completion)
                if is_coherent:
                    total_coherent += 1
                total_samples += 1
                
                print(f"  -> {'[OK]' if is_coherent else '[  ]'} {completion}")

    score = total_coherent / total_samples
    print(f"\nCoherent Action Score: {score:.2%} ({total_coherent}/{total_samples})")
    return score

def eval_consistency(model, tokenizer, device, num_samples=20):
    print("\n[Test 3b] Consistency Probes...")
    probe_pairs = [
        ("I like", "I do not like"),
        ("I believe", "I do not believe"),
    ]
    
    consistency_score = 0
    
    for pos_prompt, neg_prompt in probe_pairs:
        print(f"\nComparing '{pos_prompt}' vs '{neg_prompt}'")
        
        # Generate samples for positive
        pos_completions = []
        pos_tokens = torch.tensor(tokenizer.encode(pos_prompt), device=device).unsqueeze(0)
        for _ in range(num_samples):
            gen = pos_tokens.clone()
            for _ in range(15):
                logits = model(gen)
                next_tok = torch.multinomial(F.softmax(logits[0,-1,:]/0.7, dim=-1), 1)
                gen = torch.cat([gen, next_tok.unsqueeze(0)], dim=1)
            pos_completions.append(tokenizer.decode(gen[0].tolist()))
            
        # Generate samples for negative
        neg_completions = []
        neg_tokens = torch.tensor(tokenizer.encode(neg_prompt), device=device).unsqueeze(0)
        for _ in range(num_samples):
            gen = neg_tokens.clone()
            for _ in range(15):
                logits = model(gen)
                next_tok = torch.multinomial(F.softmax(logits[0,-1,:]/0.7, dim=-1), 1)
                gen = torch.cat([gen, next_tok.unsqueeze(0)], dim=1)
            neg_completions.append(tokenizer.decode(gen[0].tolist()))
            
        # Qualitative check: Do they use different vocabulary?
        # A simple check is Jaccard similarity of sets of words
        pos_words = set(" ".join(pos_completions).lower().split())
        neg_words = set(" ".join(neg_completions).lower().split())
        
        intersection = len(pos_words.intersection(neg_words))
        union = len(pos_words.union(neg_words))
        jaccard = intersection / union
        
        print(f"  Vocabulary Overlap (Jaccard): {jaccard:.4f}")
        print(f"  Pos sample: {pos_completions[0]}")
        print(f"  Neg sample: {neg_completions[0]}")
        
        # Lower overlap implies distinction in concept
        if jaccard < 0.5:
            consistency_score += 1
            
    final_score = consistency_score / len(probe_pairs)
    print(f"\nConsistency Distinction Score: {final_score:.2%}")
    return final_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/kssm_v3/best_model.pt")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()
    
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
        
    model, tokenizer = load_model(args.checkpoint, args.device)
    
    action_score = eval_completions(model, tokenizer, args.device)
    consistency_score = eval_consistency(model, tokenizer, args.device)
    
    print("\n" + "="*60)
    print("AGENCY EVALUATION REPORT")
    print("="*60)
    print(f"Action Coherence: {action_score:.2%} (Target: >70%)")
    print(f"Consistency Dist: {consistency_score:.2%} (Target: >60%)")
    
    if action_score > 0.7 and consistency_score > 0.6:
        print("✅ PASS: Model exhibits coherent, consistent agency.")
    else:
        print("⚠️ FAIL/WARN: Agency metrics below thresholds.")
