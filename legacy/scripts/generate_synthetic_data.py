#!/usr/bin/env python3
"""
Generate synthetic high-resonance training data for Phase-RWKV
"""

import json
from pathlib import Path

# Consciousness and phase dynamics corpus
SYNTHETIC_CORPUS = [
    # Consciousness themes
    "The nature of consciousness emerges from the interplay of uncertainty and coherence in neural dynamics.",
    "Awareness may not require certainty, but rather the ability to maintain epistemic uncertainty while remaining coherent.",
    "The hard problem of consciousness asks how subjective experience arises from objective physical processes.",
    "Integrated information theory suggests consciousness correlates with the irreducibility of causal interactions.",
    "The global workspace theory posits that consciousness arises from widespread information broadcast across neural networks.",

    # Phase coupling and oscillators
    "Kuramoto oscillators demonstrate how local coupling rules can generate global synchronization without central control.",
    "Phase locking enables coordinated activity while preserving individual oscillator identity and natural frequencies.",
    "The order parameter R measures the degree of collective synchronization in coupled oscillator systems.",
    "Partial synchronization occupies the space between chaos and rigid lock-in, enabling flexible coherence.",
    "Natural frequency heterogeneity prevents complete phase locking and maintains dynamic diversity.",

    # Quantum parallels
    "The observer effect in quantum mechanics suggests that measurement itself shapes the state being measured.",
    "Wave function collapse represents the transition from superposition to definite classical states.",
    "Heisenberg's uncertainty principle establishes fundamental limits on complementary observable pairs.",
    "Quantum decoherence explains how classical definiteness emerges from quantum superposition through environmental interaction.",
    "The delayed-choice experiment reveals that measurement context can retroactively affect prior quantum behavior.",

    # Uncertainty and measurement
    "Epistemic uncertainty reflects our incomplete knowledge about the state of a system.",
    "Aleatoric uncertainty captures the inherent randomness in the process being modeled.",
    "Information entropy quantifies the average unpredictability in a probability distribution.",
    "Maximum entropy distributions make the fewest assumptions beyond observed constraints.",
    "Bayesian inference updates beliefs by combining prior knowledge with observed evidence.",

    # Temporal dynamics
    "Recurrent neural networks process sequences by maintaining hidden state that evolves over time.",
    "State-space models represent temporal dynamics through continuous latent state evolution.",
    "Time-mixing blocks enable models to integrate information across different temporal scales.",
    "Temporal credit assignment determines which past actions contributed to current outcomes.",
    "Long-range dependencies require architectures that can propagate information across many time steps.",

    # Goldilocks principles
    "The Goldilocks zone describes the narrow range where conditions are neither too extreme nor too mild.",
    "Optimal performance often emerges at intermediate levels rather than at extremes.",
    "Too much order leads to rigidity; too much chaos prevents coordination; the balance enables adaptation.",
    "Critical states exist at phase transitions where systems exhibit maximal responsiveness to inputs.",
    "Edge of chaos dynamics combine stability and flexibility for complex adaptive behavior.",

    # Emergence and complexity
    "Emergent properties arise from collective interactions that cannot be predicted from individual components alone.",
    "Self-organization occurs when local interactions generate global patterns without external direction.",
    "Complex adaptive systems exhibit behavior at multiple scales from microscopic interactions to macroscopic order.",
    "Downward causation suggests that higher-level patterns can constrain lower-level dynamics.",
    "Hierarchical organization enables both modular specialization and integrated function.",

    # Complementarity and tradeoffs
    "Complementary observables cannot be simultaneously known with arbitrary precision.",
    "The exploration-exploitation tradeoff balances trying new options against exploiting known good choices.",
    "Bias-variance tradeoff illustrates how model complexity affects generalization performance.",
    "Speed-accuracy tradeoffs appear across cognitive and neural processing.",
    "Precision-generality tradeoff governs the scope and specificity of learned representations.",

    # Neural architectures
    "Attention mechanisms enable models to selectively focus on relevant information while ignoring distractions.",
    "Transformer architectures process sequences through multi-head self-attention without explicit recurrence.",
    "Convolutional layers detect local patterns through spatially-invariant feature extraction.",
    "Residual connections facilitate gradient flow in very deep networks.",
    "Layer normalization stabilizes training by normalizing activations within each layer.",

    # Information theory
    "Mutual information quantifies how much knowing one variable reduces uncertainty about another.",
    "The information bottleneck principle seeks representations that compress inputs while preserving task-relevant information.",
    "Rate-distortion theory characterizes optimal compression given acceptable information loss.",
    "Channel capacity determines the maximum rate of reliable information transmission.",
    "Shannon entropy provides a lower bound on average code length for lossless compression.",

    # Resonance and coupling
    "Resonance occurs when a system responds with amplified amplitude to periodic driving at its natural frequency.",
    "Coupling strength determines how strongly interacting oscillators influence each other's dynamics.",
    "Synchronization transitions occur when coupling exceeds a critical threshold for collective coordination.",
    "Chimera states exhibit coexisting synchronized and desynchronized regions in coupled oscillator networks.",
    "Frequency pulling causes weakly coupled oscillators to adjust their rates toward each other.",

    # Meta-learning and adaptation
    "Meta-learning trains models to learn how to learn, enabling rapid adaptation to new tasks.",
    "Few-shot learning generalizes from minimal examples by leveraging prior meta-knowledge.",
    "Continual learning maintains performance on old tasks while acquiring new capabilities.",
    "Transfer learning applies knowledge from one domain to accelerate learning in related domains.",
    "Plasticity-stability dilemma balances incorporating new information against preserving existing knowledge.",

    # Measurement and observation
    "Measurement apparatus and measured system form an inseparable whole during quantum observation.",
    "Contextuality means that measurement outcomes depend on what other observables are measured simultaneously.",
    "The act of observation can fundamentally alter the observed system's state and subsequent behavior.",
    "Weak measurement allows partial information extraction with minimal disturbance to the quantum state.",
    "Quantum non-demolition measurements read out information while preserving quantum coherence.",

    # Phase space and dynamics
    "Phase space trajectories trace out the evolution of a dynamical system's state over time.",
    "Attractors represent stable long-term behavior toward which trajectories converge.",
    "Bifurcations occur when small parameter changes cause qualitative shifts in system behavior.",
    "Limit cycles are closed trajectories representing sustained periodic oscillations.",
    "Strange attractors have fractal structure and exhibit sensitive dependence on initial conditions.",

    # Coherence and decoherence
    "Quantum coherence enables superposition states that exist across multiple classical possibilities simultaneously.",
    "Decoherence degrades quantum superpositions through entanglement with environmental degrees of freedom.",
    "Coherence time measures how long a quantum system maintains superposition before decohering.",
    "Quantum error correction protects coherent states by encoding information redundantly.",
    "Topological protection leverages global properties to shield quantum information from local perturbations."
]

def generate_variations(base_texts, variations_per_text=3):
    """Generate variations of base texts by recombining concepts."""
    all_samples = []

    # Add base texts
    for text in base_texts:
        all_samples.append({"text": text})

    # Add simple concatenations
    import random
    for i in range(len(base_texts) * variations_per_text):
        if len(all_samples) >= 1000:  # Limit total samples
            break

        # Pick 2-3 related sentences
        num_sentences = random.choice([2, 3])
        selected = random.sample(base_texts, num_sentences)
        combined = " ".join(selected)

        all_samples.append({"text": combined})

    return all_samples

def main():
    print("=" * 70)
    print("🌀 GENERATING SYNTHETIC HIGH-RESONANCE TRAINING DATA")
    print("=" * 70)

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Generate samples
    print(f"\n📝 Generating samples from {len(SYNTHETIC_CORPUS)} base texts...")
    samples = generate_variations(SYNTHETIC_CORPUS, variations_per_text=3)

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(samples)

    # Save as JSONL
    output_path = data_dir / "high_resonance.jsonl"
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"✅ Created {len(samples)} samples")
    print(f"💾 Saved to: {output_path}")

    # Verify
    print("\n🔍 Verifying...")
    loaded = []
    with open(output_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            loaded.append(item)

    print(f"✅ Verified: {len(loaded)} samples loaded successfully")

    # Sample statistics
    lengths = [len(s['text'].split()) for s in loaded]
    print(f"\n📊 Sample Statistics:")
    print(f"   Total samples: {len(loaded)}")
    print(f"   Text length (words): min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")

    print(f"\n📝 First sample:")
    print(f"   \"{loaded[0]['text'][:100]}...\"")

    print("\n✅ Synthetic data generation complete!")
    print(f"\nReady for training:")
    print(f"   python3 train_phase_rwkv.py --data {output_path} --iters 500")

if __name__ == "__main__":
    main()
