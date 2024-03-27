import math

import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn  import Module
from torch.nn import functional as F


def _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p):

    B, Nt, E = q.shape
    # q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(E)

    if attn_mask is not None:
        # attn += attn_mask
        attn = attn.masked_fill(attn_mask, float('-inf'))

    # if attn_mask is not None:
    #     attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
    # else:
    #     attn = torch.bmm(q, k.transpose(-2, -1))

    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,

    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],

    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,

    attn_mask: Optional[Tensor] = None,

) -> Tuple[Tensor, Optional[Tensor]]:

    is_batched = True

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    head_dim = embed_dim // num_heads
    q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    # prep attention mask
    # if attn_mask is not None:
    attn_mask = attn_mask.unsqueeze(0)
    # else:
    #     print('attn_mask is None')

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    
    k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    
    v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    src_len = k.size(1)

    # merge key padding and attention masks
    # if key_padding_mask is not None:
        # assert key_padding_mask.shape == (bsz, src_len), \
        #     f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
    key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
        expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
    # if attn_mask is None:
    #     attn_mask = key_padding_mask
    # elif attn_mask.dtype == torch.bool:
    attn_mask = attn_mask.logical_or(key_padding_mask)
    # else:
    #     attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))
    # else:
    #     print('key_padding_mask is None')

    # convert mask to float
    # if attn_mask is not None and attn_mask.dtype == torch.bool:
    #     new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
    #     new_attn_mask.masked_fill_(attn_mask, float("-inf"))
    #     attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    return attn_output, None



class MultiheadAttention(Module):

    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # if self._qkv_same_embed_dim is False:
        #     print('...')
        #     self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        #     self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
        #     self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
        #     self.register_parameter('in_proj_weight', None)
        # else:
        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        # if bias:
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        # else:
        #     print('...')
        #     self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        # if add_bias_kv:
        #     print('...')
        #     self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        #     self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        # else:
        self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        # self._reset_parameters()

    def _reset_parameters(self):
        # if self._qkv_same_embed_dim:
        xavier_uniform_(self.in_proj_weight)
        # else:
        #     xavier_uniform_(self.q_proj_weight)
        #     xavier_uniform_(self.k_proj_weight)
        #     xavier_uniform_(self.v_proj_weight)

        # if self.in_proj_bias is not None:
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)
        # if self.bias_k is not None:
        #     xavier_normal_(self.bias_k)
        # if self.bias_v is not None:
        #     xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:

        attn_output, attn_output_weights = multi_head_attention_forward(
            query, key, value, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,

            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, 
            attn_mask=attn_mask)

        return attn_output, attn_output_weights