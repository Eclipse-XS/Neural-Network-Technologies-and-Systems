import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        positions = torch.arange(max_len).float().unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2).float()
                                * (-math.log(10000.0) / d_model))
        values = torch.zeros(max_len, d_model)
        values[:, 0::2] = torch.sin(positions * frequencies)
        values[:, 1::2] = torch.cos(positions * frequencies[:d_model // 2])
        self.register_buffer("values", values.unsqueeze(0))

    def forward(self, embeddings):
        if embeddings.size(1) > self.values.size(1):
            raise ValueError("Sequence exceeds positional encoding capacity")
        return embeddings + self.values[:, :embeddings.size(1)].to(embeddings.dtype)


def validate_config(d_model, nhead, num_layers, maximum_layers):
    if d_model < 1 or nhead < 1 or d_model % nhead:
        raise ValueError("d_model must be positive and divisible by nhead")
    if not 1 <= num_layers <= maximum_layers:
        raise ValueError(f"num_layers must be between 1 and {maximum_layers}")
