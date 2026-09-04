import torch
import torch.nn as nn

class M1TransformerZero(nn.Module):
    def __init__(self, input_size=3, dim_val=32, n_heads=4,
                 n_encoder_layers=2, dropout=0.1, window_size=25):
        super().__init__()
        self.input_layer = nn.Linear(input_size, dim_val)
        self.pos_encoding = nn.Parameter(torch.zeros(1, window_size, dim_val))
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