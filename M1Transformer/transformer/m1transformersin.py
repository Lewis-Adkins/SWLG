import torch
import torch.nn as nn
import math

def sinusoidal_positional_encoding(window_size, dim_val):
    pe = torch.zeros(window_size, dim_val)
    position = torch.arange(0, window_size, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim_val, 2).float() * (-math.log(10000.0) / dim_val))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, window_size, dim_val)


class M1TransformerSin(nn.Module):
    def __init__(self, input_size=3, dim_val=32, n_heads=4,
                 n_encoder_layers=2, dropout=0.1, window_size=25):
        super().__init__()
        self.input_layer = nn.Linear(input_size, dim_val)
        self.register_buffer('pos_encoding', sinusoidal_positional_encoding(window_size, dim_val))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_val * 4,
            dropout=dropout,
            batch_first=True,
            norm_first= True,
            
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers, enable_nested_tensor=False)
        self.output_layer = nn.Linear(dim_val, 1)

    def forward(self, x):
        x = self.input_layer(x)      # (batch, 25, 3) -> (batch, 25, dim_val): project raw features up
        x = x + self.pos_encoding    # <-- THIS is where positional info gets injected
        x = self.encoder(x)          # the encoder stack: attention + feedforward, repeated n_encoder_layers times
        x = x[:, -1, :]               # take the last position's output vector
        x = self.output_layer(x)     # project down to a single scalar
        return x