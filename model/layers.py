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
        return self.W_o(x) # (batch_size, seq_len, d_model)
