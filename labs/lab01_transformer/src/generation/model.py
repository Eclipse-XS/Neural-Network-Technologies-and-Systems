import math
import torch
from torch import nn
from ..common.text import PAD
from ..common.positional_encoding import PositionalEncoding, validate_config


def causal_mask(length, device=None):
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)


class CausalLanguageModel(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=64, nhead=2,
                 num_layers=1, dim_feedforward=256, dropout=.1):
        super().__init__()
        validate_config(d_model, nhead, num_layers, 5)
        self.config = dict(vocab_size=vocab_size, max_len=max_len, d_model=d_model,
                           nhead=nhead, num_layers=num_layers,
                           dim_feedforward=dim_feedforward, dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.position = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                           dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids) * math.sqrt(self.config["d_model"])
        hidden = self.position(embeddings)
        # One constant memory vector, independent of ALL input tokens. Cross-attention
        # cannot expose a future token; its biases may learn a constant contribution.
        memory = hidden.new_zeros(hidden.size(0), 1, hidden.size(2))
        decoded = self.decoder(hidden, memory,
                               tgt_mask=causal_mask(hidden.size(1), hidden.device),
                               tgt_key_padding_mask=input_ids.eq(PAD))
        return self.projection(decoded)
