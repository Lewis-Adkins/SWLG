import math

import torch
import torch.nn as nn


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """T5-style relative position bucketing [1, Appendix].

    Maps an integer offset (key_pos - query_pos) to a bucket id in
    [0, num_buckets). Small offsets get their own exact bucket; offsets
    beyond `max_exact` are compressed logarithmically into the remaining
    buckets, up to `max_distance`, beyond which they all share the last
    bucket. This lets a fixed, small number of learned biases cover an
    arbitrarily long relative distance -- the mechanism T5 uses so a handful
    of buckets can represent both "the previous timestep" and "very far away"
    without needing one embedding per possible offset.
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


class RelativePositionBias(nn.Module):
    """Learned (bucket, head) -> scalar bias table [1, Section 2.1].

    Computed once per forward pass from the sequence length alone (no
    dependence on q/k content) and shared across every encoder layer, matching
    how the reference T5 implementation computes the bias in its first layer
    and reuses it for the rest of the stack.
    """

    def __init__(self, n_heads, num_buckets=32, max_distance=25, bidirectional=True):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(num_buckets, n_heads)

    def forward(self, seq_len, device):
        query_pos = torch.arange(seq_len, device=device)[:, None]
        key_pos = torch.arange(seq_len, device=device)[None, :]
        relative_position = key_pos - query_pos  # (T, T)
        rp_bucket = relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.embedding(rp_bucket)  # (T, T, n_heads)
        return values.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, T, T), broadcasts over batch


class T5LayerNorm(nn.Module):
    """T5's RMSNorm [1, Section 2.1]: rescale-only, no mean-subtraction or bias."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class T5AttnLayer(nn.Module):
    """Multi-head self-attention with relative position bias added to the
    logits [1, eq. in Section 2.1], instead of injecting position information
    into the input embeddings (contrast with the additive sin/learned
    encodings, or the rotation-based RoPE encoding, used elsewhere in this repo).
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.inv_sqrt_d_head = 1.0 / (self.d_head ** 0.5)

        # T5 omits bias terms on every dense layer
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, position_bias):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att_scores = (q @ k.transpose(-2, -1)) * self.inv_sqrt_d_head + position_bias
        att = torch.softmax(att_scores, dim=-1)
        att = self.dropout(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class T5EncoderLayer(nn.Module):
    """Pre-norm encoder block [1, Section 2.1]: T5 normalizes *before* each
    sublayer and only re-normalizes at the very end of the stack, rather than
    normalizing after every residual add.
    """

    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.norm1 = T5LayerNorm(d_model)
        self.attn = T5AttnLayer(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = T5LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=False),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model, bias=False),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, position_bias):
        x = x + self.dropout1(self.attn(self.norm1(x), position_bias))
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x


class M1TransformerT5(nn.Module):
    """
    [1] https://arxiv.org/abs/1910.10683 - "Exploring the Limits of Transfer
        Learning with a Unified Text-to-Text Transformer" (T5), Raffel et al.
        2020 - source of the relative-position-bucket attention bias.
    [2] https://par.nsf.gov/servlets/purl/10191796 - "Traffic Transformer:
        Capturing the continuity and periodicity of time series for traffic
        forecasting", Cai et al. 2020 - motivates relative (vs. absolute)
        position indexing for fixed-window time series forecasting, and
        injecting position information directly into the attention weights
        rather than only additively into the input embedding.

    Unlike M1Transformer (learned) / M1TransformerSin (sinusoidal) /
    M1TransformerRoPE, no positional information is added to the input
    embeddings at all -- position only enters through the per-head
    (query, key) relative-distance bias added to the attention logits.
    See transformer/m1transformerT5.md for the full write-up and comparison.
    """

    def __init__(self, input_size=3, d_model=16, n_heads=2, n_encoder_layers=1,
                 dim_feedforward=64, dropout=0.1, window_size=25,
                 num_buckets=32, max_distance=None):
        super().__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        self.relative_position_bias = RelativePositionBias(
            n_heads,
            num_buckets=num_buckets,
            max_distance=max_distance or window_size,
            bidirectional=True,
        )
        self.layers = nn.ModuleList([
            T5EncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.final_norm = T5LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.input_layer(x)
        # computed once and shared across all encoder layers, matching [1]
        position_bias = self.relative_position_bias(T, x.device)
        for layer in self.layers:
            x = layer(x, position_bias)
        x = self.final_norm(x)
        x = x[:, -1, :]
        x = self.output_layer(x)
        return x
