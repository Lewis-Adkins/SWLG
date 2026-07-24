import torch
import torch.nn as nn
import numpy as np
 
import torch

class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)     # (seq_len, head_dim/2)
            emb = torch.cat((freqs, freqs), dim=-1)                # (seq_len, head_dim)
            cos = emb.cos()   # (seq_len, head_dim)
            sin = emb.sin()   # (seq_len, head_dim)
            # explicitly reshape to (1, seq_len, 1, head_dim) for batch-first broadcast
            self.cos_cached = cos.view(1, seq_len, 1, -1)
            self.sin_cached = sin.view(1, seq_len, 1, -1)
        return self.cos_cached, self.sin_cached

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RoPEAttention(nn.Module):
    def __init__(self, dim_val, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim_val // n_heads
        self.q_proj = nn.Linear(dim_val, dim_val)
        self.k_proj = nn.Linear(dim_val, dim_val)
        self.v_proj = nn.Linear(dim_val, dim_val)
        self.out_proj = nn.Linear(dim_val, dim_val)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        batch, seq_len, dim_val = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)

        cos, sin = self.rotary(q, seq_dim=1)   # batch-first: seq is dim 1
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).reshape(batch, seq_len, dim_val)
        return self.out_proj(out)


import torch
from torch import nn


class RoPEAttnLayer(nn.Module):
    """
    A version of an Attention-Encoder Layer with Rotary Positional Encodings
    (RoPE) as described in [1].

    [1] https://arxiv.org/pdf/2104.09864.pdf - RoPE Paper
    [2] https://github.com/karpathy/nanoGPT/blob/master/model.py - Guidance for
            loop-free implementation of multi-head architecture.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_pos_enc_len: int,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.inv_sqrt_d_head = 1.0 / torch.sqrt(torch.tensor(self.d_head))

        self.multi_head_in_projection = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.multi_head_out_projection = nn.Linear(d_model, d_model, bias=bias)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.n_heads = n_heads
        self.d_model = d_model
        self._construct_rope_matrices(max_pos_enc_len)

    def _construct_rope_matrices(self, max_pos_enc_len):
        """Constructs rotary embedding matrices for additive version
        [1, p. 7, eq. (34)]. Configured for x beeing of shape
        (batch_size, seqlen, d_model).
        """
        assert self.d_head % 2 == 0
        # [t1, t1, t2, t2, t3, t3, ...]
        thetas = 1000 ** (
            -2.0 * torch.arange(1, self.d_head / 2 + 1) / self.d_head
        ).repeat_interleave(2)
        positions = torch.arange(1, max_pos_enc_len + 1).float()
        # [ [1t1, 1t1, 1t2, 1t2, ...],
        #   [2t1, 2t1, 2t2, 2t2, ...],
        #   [3t1, 3t1, 3t2, 3t2, ...],
        #   ...                       ]
        args = positions.reshape(-1, 1) @ thetas.reshape(1, -1)
        self.register_buffer("rope_sin", torch.sin(args))
        self.register_buffer("rope_cos", torch.cos(args))

    def _reorder_for_rope_sin(self, x):
        """Reorders the inputs according to [1, p. 7, eq. (34)] for the
        multiplication with the sinus-part of the RoPE. Configured for x beeing
        having d_head as last dimension, should be of shape
        (batch_size, n_heads, seqlen, d_head).
        """
        # [x1, x3, x5, ...]
        x_odd = x[..., ::2]
        # [x2, x4, x6, ...]
        x_even = x[..., 1::2]
        # [[-x2, x1], [-x4, x3], [-x6, x5], ...]
        x_stacked = torch.stack([-x_even, x_odd], dim=-1)
        # [-x2, x1, -x4, x3, ...]
        return x_stacked.flatten(start_dim=-2)

    def _apply_rope(self, x):
        """Applies RoPE the inputs according to [1, p. 7, eq. (34)].
        Configured for x being of shape (batch_size, n_heads, seqlen, d_head).
        """
        T = x.shape[2]
        x_sin = self._reorder_for_rope_sin(x)
        x_rope = x * self.rope_cos[:T, :] + x_sin * self.rope_sin[:T, :]
        return x_rope

    def forward(self, x):
        B, T, C = x.size()  # batch_size, seqlen, d_model

        # apply key, query, value projections
        q, k, v = self.multi_head_in_projection(x).split(self.d_model, dim=2)

        # separate heads (batch_size, n_heads, seqlen, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # apply RoPE transformation [1, p. 7]
        q_rope = self._apply_rope(q)
        k_rope = self._apply_rope(k)

        # RoPE self attention:
        #   (batch_size, n_heads, seqlen, d_head) x
        #   (batch_size, n_heads, d_head, seqlen)
        #       -> (batch_size, n_heads, seqlen, seqlen)
        #  This is the place, where the rotations get "inserted" into the
        #  attention mechanism as presented in [1, p. 6, eq. 19]. I stick to
        #  the basic `exp` as non-negativities.
        att_numerator = torch.exp(
            (q_rope @ k_rope.transpose(-2, -1)) * self.inv_sqrt_d_head
        )
        att_denominator = torch.exp((q @ k.transpose(-2, -1)) * self.inv_sqrt_d_head)
        att_denominator = torch.sum(att_denominator, dim=-1, keepdim=True)
        att = att_numerator / att_denominator
        # (batch_size, n_heads, seqlen, seqlen) x
        #   (batch_size, n_heads, seqlen, d_head)
        # -> (batch_size, n_heads, seqlen, d_head)
        y = att @ v
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.multi_head_out_projection(y)

        # skip-connection and regularization
        y = self.layer_norm(y + x)
        y = self.dropout(y)
        return y


class RoPEEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, max_pos_enc_len, dim_feedforward, dropout=0.1, bias=True):
        super().__init__()
        self.attn_layer = RoPEAttnLayer(d_model, n_heads, max_pos_enc_len, dropout, bias)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.attn_layer(x)                          # attention + residual + norm (built in)
        x = x + self.dropout(self.ff(self.norm2(x)))     # feedforward + residual + norm (added)
        return x
    
class M1TransformerRoPE(nn.Module):
    def __init__(self, input_size=3, d_model=16, n_heads=2, n_encoder_layers=1,
                 dim_feedforward=64, dropout=0.1, window_size=25):
        super().__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(d_model, n_heads, window_size, dim_feedforward, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = x[:, -1, :]
        x = self.output_layer(x)
        return x