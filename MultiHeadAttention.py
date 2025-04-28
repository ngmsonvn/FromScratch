import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Transform input vector by linear layer and reshape output to separate heads
class PrepareMHA(nn.Module):
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        head_shape = x.shape[:-1]  # (seq_len, batch_size)
        x = self.linear(x)         # (seq_len, batch_size, heads * d_k)
        x = x.view(*head_shape, self.heads, self.d_k)  # (seq_len, batch_size, heads, d_k)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout: float = 0.2, bias: bool = True):
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads

        self.query = PrepareMHA(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareMHA(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareMHA(d_model, heads, self.d_k, bias=bias)

        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(self.d_k)

        self.attn = None

    def forward(self, *, query, key, value, mask=None):
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        
        # Transform input
        query = self.query(query)  # (seq_len, batch_size, heads, d_k)
        key = self.key(key)        # (seq_len, batch_size, heads, d_k)
        value = self.value(value)  # (seq_len, batch_size, heads, d_k)

        # Calculate attention score
        scores = self.get_scores(query, key)  # (seq_len, seq_len, batch_size, heads)
        scores *= self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=1)
        attn = self.dropout(attn)

        
        x = torch.einsum('ijbh,jbhd->ibhd', attn, value)

        self.attn = attn.detach()

        x = x.reshape(seq_len, batch_size, -1)

        return self.output(x)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        mask = mask.unsqueeze(-1)  
        return mask

mha = MultiHeadAttention(heads=8, d_model=512)

query = torch.randn(10, 32, 512) 
key = torch.randn(10, 32, 512)
value = torch.randn(10, 32, 512)
mask = torch.ones(10, 10, 32)      

output = mha(query=query, key=key, value=value, mask=mask)
print(output.shape)
