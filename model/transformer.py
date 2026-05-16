import torch
import torch.nn as nn
import math

from model.layers import EncoderLayer, PositionalEncoding


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask=None):
        # x: (batch_size, seq_len) - тупо пока idшники слов
        embeds = self.embedding(x)
        embeds = embeds * math.sqrt(self.d_model)
        x_encoded = self.dropout(self.pos_encoder(embeds))

        res = x_encoded
        for layer in self.layers:
            res = layer(res, mask)
        return res
