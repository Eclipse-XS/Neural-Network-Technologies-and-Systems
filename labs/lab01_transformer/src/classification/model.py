import math
from torch import nn
from ..common.text import PAD
from ..common.positional_encoding import PositionalEncoding, validate_config


class SpamClassifier(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=64, nhead=2,
                 num_layers=1, dim_feedforward=256, dropout=.1):
        super().__init__()
        validate_config(d_model, nhead, num_layers, 2)
        self.config = dict(vocab_size=vocab_size, max_len=max_len, d_model=d_model,
                           nhead=nhead, num_layers=num_layers,
                           dim_feedforward=dim_feedforward, dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.position = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                           dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 2)

    @staticmethod
    def masked_mean(encoded, padding_mask):
        valid = (~padding_mask).unsqueeze(-1)
        return (encoded * valid).sum(1) / valid.sum(1).clamp_min(1)

    def forward(self, input_ids):
        padding_mask = input_ids.eq(PAD)  # [batch, sequence], True means ignore.
        if padding_mask.all(1).any():
            raise ValueError("Each sequence must contain a non-PAD token")
        embeddings = self.embedding(input_ids) * math.sqrt(self.config["d_model"])
        encoded = self.encoder(self.position(embeddings), src_key_padding_mask=padding_mask)
        pooled = self.masked_mean(encoded, padding_mask)
        return self.classifier(self.dropout(pooled))
