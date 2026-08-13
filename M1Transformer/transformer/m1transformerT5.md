# M1TransformerT5: relative position bias

`transformer/m1transformerT5.py` adds a fourth positional-encoding variant to
the M1 model family, alongside `M1Transformer` (learned), `M1TransformerSin`
(sinusoidal), and `M1TransformerRoPE` (rotary). It's based on:

1. **T5** — Raffel et al. 2020, ["Exploring the Limits of Transfer Learning
   with a Unified Text-to-Text Transformer"](https://arxiv.org/abs/1910.10683),
   Section 2.1 — source of the relative-position-bucket attention bias
   mechanism implemented here.
2. **Traffic Transformer** — Cai et al. 2020, ["Capturing the continuity and
   periodicity of time series for traffic forecasting"](https://par.nsf.gov/servlets/purl/10191796),
   *Transactions in GIS* — a forecasting-specific paper (not actually about
   T5) that argues for *relative* rather than absolute position indexing in
   fixed windows, and for letting position influence the attention weights
   directly rather than only the input embedding. Its ideas motivate *why*
   a T5-style bias is worth trying here, even though its own mechanism
   differs from T5's (see [Traffic Transformer's mechanism](#traffic-transformers-actual-mechanism) below).

## How T5's relative position bias works

T5 departs from the original Transformer in two ways that matter for us:

**1. Position is never added to the token embedding.** `M1Transformer`,
`M1TransformerSin`, and `M1TransformerRoPE` all inject position by modifying
`q`/`k` (or `x`) before the dot product — an added vector, or a rotation.
T5 does neither. The token embedding carries *no* position information at
all. Instead, a learned scalar bias is added directly to the attention
logits:

```
attn_scores[h, i, j] = (q_i · k_j) / sqrt(d_head) + bias[h, bucket(j - i)]
softmax over j
```

`bias` depends only on the *relative* offset `j - i` (key position minus
query position) and the head `h` — never on absolute position, and never on
the content of `q`/`k`. It's a single small lookup table
(`num_buckets × n_heads`), computed once and reused by every layer in the
stack (real T5 computes it in layer 0 and passes it down; this file computes
it once in `M1TransformerT5.forward` for the same effect).

**2. Bucketing compresses arbitrary distance into a fixed table.** A window
of length `L` has `2L-1` possible relative offsets. Rather than one embedding
per offset, `relative_position_bucket()` gives small offsets (the most
behaviorally important — "1 step ago" vs "2 steps ago") their own exact
bucket, then compresses larger offsets logarithmically into the remaining
buckets, saturating at `max_distance`. For our fixed 25-step window
(`max_distance=window_size`), offsets 0–7 (`max_exact = num_buckets/2/2`)
get exact buckets, and offsets 8–24 share the rest — appropriate since flux
history 20 vs. 24 steps back is far less distinguishable than 1 vs. 2 steps
back.

**3. Pre-norm + RMSNorm, no linear biases.** `T5LayerNorm` is rescale-only
(no mean subtraction, no bias term), and every dense layer in
`T5AttnLayer`/`T5EncoderLayer` has `bias=False`, matching T5's actual
parameterization. Normalization happens *before* each sublayer
(`norm_first`-style), not after the residual add — this is also what
`M1Transformer`/`M1TransformerSin` do via `nn.TransformerEncoderLayer(...,
norm_first=True)`, and what the post-norm `RoPEAttnLayer` currently does
*not* do (a candidate factor in the RoPE model's training instability,
see below).

### Traffic Transformer's actual mechanism

Worth being precise about what Cai et al. propose, since the file is named
after T5 but the second paper isn't a T5 paper. Their relevant contributions:

- **Relative Position Encoding (RPE)**: index timesteps within a
  source/target window starting from 0, rather than by absolute time —
  i.e. exactly the "position 0..24 within this window" framing this repo
  already uses for `M1Transformer`/`M1TransformerSin`, justified for the
  same reason we'd want it: the model shouldn't care *when* a window
  occurred, only its internal structure.
- **Periodic position encodings** (daily/weekly) — not applicable here,
  since `data/data.csv` has no calendar features and solar particle flux
  events don't have a weekly/daily period the way road traffic does.
- **Similarity-based combination** (their eq. 8–12): instead of adding
  position embeddings to the input, they multiply the attention score
  `e_ij` by a decay factor `b_ij = softmax_k(pos_embedding_i ·
  pos_embedding_j)` derived from position-embedding similarity — a
  *different* mechanism from T5's learned-bucket bias, but the same
  underlying idea: **inject position into the attention weights, not the
  embeddings**. That convergence, from an unrelated forecasting paper, is
  the actual reason to expect a T5-style bias might do better here than
  the additive/rotary approaches already in this repo.

## Comparison to the other three M1 models

| | `M1Transformer` | `M1TransformerSin` | `M1TransformerRoPE` | `M1TransformerT5` |
|---|---|---|---|---|
| Position signal | learned vector, added to input | fixed sinusoidal vector, added to input | rotation applied to q/k per layer | learned scalar bias added to attention logits |
| Where position enters | input embedding (once) | input embedding (once) | attention (every layer, multiplicative) | attention (every layer, additive) |
| Depends on absolute or relative position | absolute (position index 0–24) | absolute (position index 0–24) | relative (rotation difference cancels to `m - n`) | relative (`bucket(j - i)` directly) |
| Learned params for position | `window_size × d_model` (zero-init) | 0 (fixed) | 0 (fixed frequencies) | `num_buckets × n_heads` (tiny — 32×2=64 by default) |
| Extrapolates beyond `window_size`? | no (fixed-size table) | yes, but arguably not meaningful for a 25-step forecast window | yes | yes (buckets saturate gracefully) |
| Norm placement | pre-norm (`norm_first=True`) | pre-norm (`norm_first=True`) | post-norm | pre-norm |
| Linear layer biases | yes (`nn.TransformerEncoderLayer` default) | yes | yes | no (T5 convention) |

The survey (arXiv:2502.12370) argued learned/absolute encodings are the
natural fit for short, fixed-length windows since there's no extrapolation
need — that argument favors `M1Transformer` over `M1TransformerRoPE`, but
doesn't address the *attention-bias-vs-embedding-addition* axis, which is
the real difference this T5 variant tests. If flux forecasting genuinely
benefits from directly weighting "how many steps back" rather than encoding
"which position" into the value each timestep carries, `M1TransformerT5`
should show it — with far fewer position-specific parameters than the
learned encoding (64 vs. 400) and no risk of the sign/rotation bugs that hit
the RoPE implementation, since the bias is a plain learned lookup table
added before softmax.

## Not yet wired in

`M1TransformerT5` isn't imported or instantiated in `main.py` yet — it's a
standalone model file, matching the constructor signature of
`M1TransformerRoPE` (`input_size, d_model, n_heads, n_encoder_layers,
dim_feedforward, dropout, window_size`, plus optional `num_buckets` /
`max_distance`) so it can be dropped into `set_up_model_train_test` the same
way once the current negative-PE regression in the RoPE path is resolved.
