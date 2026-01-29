"""
MPS-Compatible DIT Model for MDLM

Replaces flash_attn calls with standard PyTorch attention for Mac Studio.
Based on kuleshov-group/mdlm dit.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# Flags for JIT fusion (work on MPS)
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)


def bias_dropout_add_scale(x, bias, scale, residual, prob, training):
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)
    return _bias_dropout_add


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@torch.jit.script
def modulate_fused(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        # MPS-compatible layer norm (no autocast needed)
        x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class Rotary(nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.cos_cached[:, :, 2, :, :].fill_(1.)
            self.sin_cached[:, :, 2, :, :].fill_(0.)
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_mps(q, k, cos, sin):
    """MPS-compatible rotary position embedding"""
    # cos, sin: [1, seq, 1, 1, head_dim]
    cos = cos[0, :, 0, 0, :]  # [seq, head_dim]
    sin = sin[0, :, 0, 0, :]

    # q, k: [batch, seq, heads, head_dim]
    q_embed = (q * cos.unsqueeze(0).unsqueeze(2)) + (rotate_half(q) * sin.unsqueeze(0).unsqueeze(2))
    k_embed = (k * cos.unsqueeze(0).unsqueeze(2)) + (rotate_half(k) * sin.unsqueeze(0).unsqueeze(2))

    return q_embed, k_embed


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DDiTBlockMPS(nn.Module):
    """DiT block with MPS-compatible attention (no flash_attn)"""

    def __init__(self, hidden_size, n_heads, cond_dim, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.dropout = dropout

        self.norm1 = LayerNorm(hidden_size)
        self.attn_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)

        self.norm2 = LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * hidden_size, hidden_size, bias=True))

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = get_bias_dropout_add_scale(self.training)

        # Get modulation parameters
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # Attention
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        # QKV projection
        qkv = self.attn_qkv(x)  # [batch, seq, 3*hidden]
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d',
                        three=3, h=self.n_heads)  # [batch, seq, 3, heads, head_dim]

        # Split Q, K, V
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # Each: [batch, seq, heads, head_dim]

        # Apply rotary embeddings (MPS-compatible)
        cos, sin = rotary_cos_sin
        q, k = apply_rotary_pos_emb_mps(q, k, cos, sin)

        # Standard scaled dot-product attention (MPS-compatible)
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        x = torch.matmul(attn, v)

        # Reshape back
        x = x.transpose(1, 2)  # [batch, seq, heads, head_dim]
        x = rearrange(x, 'b s h d -> b s (h d)')

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # MLP
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout)

        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DITMPS(nn.Module):
    """MPS-Compatible DIT for MDLM"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Handle both dict and OmegaConf
        if hasattr(config, 'model'):
            hidden_size = config.model.hidden_size
            n_heads = config.model.n_heads
            n_blocks = config.model.n_blocks
            cond_dim = config.model.cond_dim
            dropout = config.model.dropout
            vocab_size = config.model.vocab_size
            self.scale_by_sigma = config.model.scale_by_sigma
        else:
            hidden_size = config['hidden_size']
            n_heads = config['n_heads']
            n_blocks = config['n_blocks']
            cond_dim = config['cond_dim']
            dropout = config.get('dropout', 0.1)
            vocab_size = config['vocab_size']
            self.scale_by_sigma = config.get('scale_by_sigma', True)

        self.vocab_size = vocab_size
        self.vocab_embed = EmbeddingLayer(hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(hidden_size // n_heads)

        self.blocks = nn.ModuleList([
            DDiTBlockMPS(hidden_size, n_heads, cond_dim, dropout=dropout)
            for _ in range(n_blocks)
        ])

        self.output_layer = DDitFinalLayer(hidden_size, vocab_size, cond_dim)

        # For Phase Core access
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks

    def forward(self, indices, sigma, return_hidden=False):
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        hidden_states = []
        for i, block in enumerate(self.blocks):
            x = block(x, rotary_cos_sin, c, seqlens=None)
            if return_hidden:
                hidden_states.append(x.clone())

        logits = self.output_layer(x, c)

        if return_hidden:
            return logits, hidden_states
        return logits


def create_mdlm_config(
    hidden_size=768,
    n_heads=12,
    n_blocks=12,
    cond_dim=768,
    vocab_size=50257,  # GPT-2 vocab
    dropout=0.1
):
    """Create config for MDLM model"""
    return {
        'hidden_size': hidden_size,
        'n_heads': n_heads,
        'n_blocks': n_blocks,
        'cond_dim': cond_dim,
        'vocab_size': vocab_size,
        'dropout': dropout,
        'scale_by_sigma': True
    }


# Test
if __name__ == "__main__":
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Create small test model
    config = create_mdlm_config(
        hidden_size=256,
        n_heads=4,
        n_blocks=4,
        cond_dim=256,
        vocab_size=1000
    )

    model = DITMPS(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size, seq_len = 2, 64
    indices = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    sigma = torch.rand(batch_size, device=device)

    logits, hidden = model(indices, sigma, return_hidden=True)
    print(f"Logits shape: {logits.shape}")
    print(f"Hidden states: {len(hidden)} layers, shape {hidden[0].shape}")

    print("✅ MPS-compatible DIT works!")
