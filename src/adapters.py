# src/adapters.py
import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, d_model: int, d_adapter: int = 64, non_linearity=nn.GELU):
        super().__init__()
        self.down = nn.Linear(d_model, d_adapter)
        self.act = non_linearity()
        self.up = nn.Linear(d_adapter, d_model)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        z = self.down(x)
        z = self.act(z)
        z = self.up(z)
        return x + self.scale * z


def insert_adapters_wav2vec2(model, adapter_dim=64):
    # Attach adapters as `adapter` attribute on transformer layers
    try:
        layers = model.wav2vec2.encoder.layers
    except Exception as e:
        raise RuntimeError('Model does not expose wav2vec2.encoder.layers: ' + str(e))

    for layer in layers:
        dim = model.config.hidden_size
        layer.adapter = Adapter(d_model=dim, d_adapter=adapter_dim)
    return model