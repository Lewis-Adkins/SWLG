import torch
import torch.nn as nn


class SigmoidGRUCell(nn.Module):
    """A GRU cell with sigmoid (not tanh) candidate-state activation, matching
    Keras's GRU(activation='sigmoid') -- Keras's gate activation is already
    sigmoid by default, so activation='sigmoid' makes every nonlinearity in
    the cell a sigmoid, unlike PyTorch's built-in nn.GRU/nn.GRUCell, which
    hardcode tanh for the candidate state and offer no way to change it.
    Standard GRU equations otherwise (reset gate r, update gate z, candidate
    n, h_t = (1-z)*n + z*h_{t-1})."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size, 3 * hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, x, h):     # x: (batch, input_size), h: (batch, hidden_size)
        gi = self.input_to_hidden(x)
        gh = self.hidden_to_hidden(h)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.sigmoid(i_n + r * h_n)
        return (1 - z) * n + z * h


class M1RNN(nn.Module):
    """Recurrent baseline matching Torres's Keras 'rnn' model: a single
    30-unit GRU layer with sigmoid activation over the (window, features)
    input, keeping only the final hidden state, then Dense(1). Looped by
    hand over the window (25 steps) rather than using cuDNN's fused GRU --
    that kernel doesn't support the sigmoid candidate activation and doesn't
    work under the vmap'd multi-seed training loop in utils/training.py, but
    a plain per-step loop is small enough (25 steps) to compile/vmap fine."""

    def __init__(self, input_size=6, window_size=25, hidden_size=30):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = SigmoidGRUCell(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):        # x: (batch, window, features)
        batch = x.shape[0]
        h = x.new_zeros(batch, self.hidden_size)
        for t in range(x.shape[1]):
            h = self.cell(x[:, t, :], h)
        return self.output_layer(h)   # (batch, 1)
