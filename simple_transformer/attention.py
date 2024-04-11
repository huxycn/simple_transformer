"""
This code is mainly copied from https://github.com/hyunwoongko/transformer/tree/master/models/layers
    - scale_dot_product_attention.py
    - multi_head_attention.py
    - layer_norm.py
"""

import math

import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn  import Module
from torch.nn import functional as F


# def _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p):

#     B, Nt, E = q.shape

#     # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
#     attn = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(E)

#     if attn_mask is not None:
#         attn = attn.masked_fill(attn_mask.squeeze(1), -1e9)

#     attn = F.softmax(attn, dim=-1)
#     if dropout_p > 0.0:
#         attn = F.dropout(attn, p=dropout_p)

#     # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
#     output = torch.bmm(attn, v)

#     return output, attn


def _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, e=1e-12):
    # input is 4 dimension tensor
    # [batch_size, head, length, d_tensor]
    batch_size, head, length, d_tensor = k.size()

    # 1. dot product Query with Key^T to compute similarity
    k_t = k.transpose(2, 3)  # transpose
    attn = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

    # 2. apply masking (opt)
    if attn_mask is not None:
        attn = attn.masked_fill(attn_mask != 0, -10000)

    # 3. pass them softmax to make [0, 1] range
    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    # 4. multiply with Value
    output = attn @ v

    return output, attn


class MultiheadAttention(Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):

        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, q, k, v, attn_mask):

        bsz, tgt_len, embed_dim = q.shape
        _, src_len, _ = k.shape
        head_dim = embed_dim // self.num_heads

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(bsz, tgt_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, head_dim).transpose(1, 2)

        # (bsz, nhead, seq_len, head_dim)

        # q = q.contiguous().view(bsz * self.num_heads, tgt_len, head_dim)
        # k = k.contiguous().view(bsz * self.num_heads, src_len, head_dim)
        # v = v.contiguous().view(bsz * self.num_heads, src_len, head_dim)

        out, attention = _scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout)

        out = out.view(bsz, self.num_heads, tgt_len, head_dim)

        # 4. concat and pass to linear layer
        out = out.transpose(1, 2).contiguous().view(bsz, out.size(2), embed_dim)

        # out = self.concat(out)
        out = self.out_proj(out)

        return out
