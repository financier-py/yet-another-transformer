import torch
import torch.nn as nn
import math

from model.layers import EncoderLayer, DecoderLayer, PositionalEncoding


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


class Decoder(nn.Module):
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
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.d_model = d_model

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, tgt_mask=None, src_mask=None
    ):
        """
        x: (batch_size, tgt_len) - индексы целевых слов
        memory: (batch_size, src_len, d_model) - выход энкодера
        """
        embeds = self.embedding(x) * math.sqrt(self.d_model)
        embeds = self.dropout(self.pos_encoder(embeds))

        res = embeds
        for layer in self.layers:
            res = layer(res, memory, tgt_mask, src_mask)
        return res


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size,
        max_seq_len: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size, max_seq_len, d_model, num_layers, num_heads, d_ff, dropout
        )
        self.decoder = Decoder(
            tgt_vocab_size, max_seq_len, d_model, num_layers, num_heads, d_ff, dropout
        )

        self.projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, src_mask=None, tgt_mask=None
    ):
        """
        src: (batch_size, src_len)
        tgt: (batch_size, tgt_len)
        """
        memory = self.encoder(src, src_mask)
        out = self.decoder(tgt, memory, tgt_mask, src_mask)
        logits = self.projection(out)
        return logits
