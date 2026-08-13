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
        """Constructs rotary embedding matrices, "rotate-half" layout
        (equivalent to the adjacent-pair rotation in [1, p. 7, eq. (34)] up to
        a fixed permutation of head dims — harmless since q/k projections are
        learned from scratch). Avoids strided ::2 indexing in _apply_rope.
        Configured for x beeing of shape (batch_size, seqlen, d_model).
        """
        assert self.d_head % 2 == 0
        thetas = 10000 ** (-2.0 * torch.arange(0, self.d_head // 2) / self.d_head)
        positions = torch.arange(0, max_pos_enc_len).float()
        freqs = positions.reshape(-1, 1) @ thetas.reshape(1, -1)   # (seqlen, d_head/2)
        emb = torch.cat((freqs, freqs), dim=-1)                     # (seqlen, d_head)
        self.register_buffer("rope_sin", torch.sin(emb))
        self.register_buffer("rope_cos", torch.cos(emb))

    def _rotate_half(self, x):
        """Splits x into contiguous halves along d_head and swaps them with a
        sign flip, i.e. [x1..x_{d/2}, x_{d/2+1}..x_d] -> [-x_{d/2+1}..-x_d, x1..x_{d/2}].
        Configured for x being of shape (batch_size, n_heads, seqlen, d_head).
        """
        x1, x2 = x[..., : self.d_head // 2], x[..., self.d_head // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, x):
        """Applies RoPE the inputs according to [1, p. 7, eq. (34)].
        Configured for x being of shape (batch_size, n_heads, seqlen, d_head).
        """
        T = x.shape[2]
        x_rope = x * self.rope_cos[:T, :] + self._rotate_half(x) * self.rope_sin[:T, :]
        return x_rope

    def forward(self, x):
        B, T, C = x.size()  # batch_size, seqlen, d_model

        # apply key, query, value projections
        q, k, v = self.multi_head_in_projection(self.layer_norm(x)).split(self.d_model, dim=2)

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
        #  Standard scaled dot-product softmax attention [1, eq. 16], using the
        #  rotated q/k so the dot product depends only on relative position.
        att_scores = (q_rope @ k_rope.transpose(-2, -1)) * self.inv_sqrt_d_head
        att = torch.softmax(att_scores, dim=-1)
        # (batch_size, n_heads, seqlen, seqlen) x
        #   (batch_size, n_heads, seqlen, d_head)
        # -> (batch_size, n_heads, seqlen, d_head)
        y = att @ v
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.multi_head_out_projection(y)

        # dropout on the sublayer's own output, before it joins the residual stream
        y = self.dropout(y)
        return x + y



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