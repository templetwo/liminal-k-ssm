# Phase-Mamba: Kuramoto Phase Coupling for Selective State Space Models
# Adapted from Apple Inc. MLX implementation

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import mlx.core as mx
import mlx.nn as nn

from mamba_mlx import ModelArgs, MambaBlock, ResidualBlock, Mamba, Model
from phase_block import PhaseBlock

class PhaseMambaBlock(MambaBlock):
    """
    Mamba Block with Kuramoto Phase Coupling injected into the SSM output.
    """
    def __init__(self, args: ModelArgs, phase_block: PhaseBlock):
        super().__init__(args)
        self.phase_block = phase_block

    def _process_sequence(self, x, conv_cache, state_cache):
        # 1. Standard Mamba sequence processing
        B, T, D = x.shape
        xz = self.in_proj(x)
        x, z = xz.split(indices_or_sections=2, axis=-1)
        K = self.conv_kernel_size
        if conv_cache is not None:
            x_full = mx.concatenate([conv_cache, x], axis=1)
        else:
            x_full = mx.pad(x, [(0, 0), (K - 1, 0), (0, 0)])
        conv_out = self.conv1d(x_full)
        new_conv_cache = x_full[:, -(K - 1) :, :]
        x = nn.silu(conv_out)
        A = -mx.exp(self.A_log)
        current_state = state_cache
        y = []
        for t in range(T):
            y_t, current_state = self.ssm_step(x[:, t], A, current_state)
            y.append(y_t)
        y = mx.stack(y, axis=1)
        
        # 2. Phase Core Handshake
        # y is the output of the SSM [B, T, intermediate_size]
        # We apply phase coupling here before the final output projection
        y_phase = self.phase_block(y)
        
        # 3. Final projection
        z = self.out_proj(nn.silu(z) * y_phase)
        return z, (new_conv_cache, current_state)

class PhaseResidualBlock(ResidualBlock):
    """
    Residual Block containing a PhaseMambaBlock.
    """
    def __init__(self, args: ModelArgs, phase_block: PhaseBlock):
        super().__init__(args)
        # Replace the standard mixer with our Phase-aware version
        self.mixer = PhaseMambaBlock(args, phase_block)

class PhaseMamba(Mamba):
    """
    Mamba model with Phase Coupling at a specific layer.
    """
    def __init__(self, args: ModelArgs, phase_layer: int = 24):
        super().__init__(args)
        self.phase_layer_idx = phase_layer
        
        # Inject Phase Core at phase_layer
        # Mamba-2.8B has 56 layers
        # d_inner = d_model * 2 = 2560 * 2 = 5120
        d_inner = args.intermediate_size
        
        self.phase_block = PhaseBlock(
            d_model=d_inner,
            rank=32, # Higher rank for larger model
            steps=2,
            dt=0.08,
            k_scale=0.5
        )
        
        # Replace the target layer
        if phase_layer < len(self.layers):
            original = self.layers[phase_layer]
            self.layers[phase_layer] = PhaseResidualBlock(args, self.phase_block)
            print(f"🌀 Phase Core grafted onto Mamba Layer {phase_layer}")

class PhaseMambaModel(Model):
    def __init__(self, args: ModelArgs, phase_layer: int = 24):
        super().__init__(args)
        # Replace the backbone with PhaseMamba
        self.backbone = PhaseMamba(args, phase_layer)

    def compute_order_parameter(self):
        return self.backbone.phase_block.current_R
        
    def get_tonal_state(self):
        return self.backbone.phase_block.current_tone
