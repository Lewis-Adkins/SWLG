import torch.nn as nn


class M1NN(nn.Module):
    """Feedforward baseline matching Torres's Keras 'regular' model:
    flatten the (window, features) input and run it through one
    sigmoid-activated hidden layer before projecting to a single scalar.
    Dense(30, sigmoid) -> Dense(1), trained with MSE/Adam (see
    utils/training.py's train_models -- same vmap'd loop as the
    transformers, not a separate training path)."""

    def __init__(self, input_size=6, window_size=25, hidden_size=30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * window_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):        # x: (batch, window, features)
        return self.net(x)       # (batch, 1)
