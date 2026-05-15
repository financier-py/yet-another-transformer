import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.size()
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        x = x.transpose(1, 2)
        return x

    def scaled_dot_prod_atten(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k)

        if mask:
            scores = scores.masked_fill(mask == 0, -1e9)

        atten_w = F.softmax(scores, dim=-1)
        output = atten_w @ V
        return output, atten_w

    def forward(self, q, k, v, mask=None):
        Q = self.split_heads(self.W_q(q))
        K = self.split_heads(self.W_k(k))
        V = self.split_heads(self.W_v(v))

        x, atten_w = self.scaled_dot_prod_atten(Q, K, V, mask)

        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.W_o(x)  # (batch_size, seq_len, d_model)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        res = F.gelu(self.fc1(x))
        res = self.fc2(self.dropout(res))
        return res


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        atten_out, _ = self.mha(norm_x, norm_x, norm_x, mask)

        x = x + self.dropout1(atten_out)

        norm_x2 = self.norm2(x)
        ffn_out = self.ffn(norm_x2)

        x = x + self.dropout2(ffn_out)

        return x


class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]
