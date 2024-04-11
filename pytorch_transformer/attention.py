import math

import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn  import Module
from torch.nn import functional as F


def _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p):

    B, Nt, E = q.shape

    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(E)

    if attn_mask is not None:
        attn = attn.masked_fill(attn_mask, float('-inf'))

    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)

    return output, attn


class MultiheadAttention(Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):

        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, q, k, v, key_padding_mask, attn_mask):

        tgt_len, bsz, embed_dim = q.shape
        src_len, _, _ = k.shape
        head_dim = embed_dim // self.num_heads

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, head_dim).transpose(0, 1)

        # (bsz * nhead, seq_len, head_dim)

        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
        attn_mask = attn_mask.unsqueeze(0)
        attn_mask = attn_mask.logical_or(key_padding_mask)

        attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.dropout)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        # attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        return attn_output, None
