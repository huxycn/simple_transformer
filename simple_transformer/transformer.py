"""
This code interfaces refer to pytorch transformer source code
but some details are aligned with the self-implemented attention in attention.py
you can compare this code with pytorch_transformer/transformer.py to see the difference
"""

import copy
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from .attention import MultiheadAttention
from torch.nn import ModuleList
from torch.nn import Dropout
from torch.nn import Linear
# from .attention import LayerNorm
from torch.nn import LayerNorm


class Transformer(Module):

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 ):
        super(Transformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, src_mask)
        return output


class TransformerEncoder(Module):

    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, src, src_mask):

        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask)

        return output


class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, tgt, memory, tgt_mask, memory_mask):

        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        return output


class TransformerEncoderLayer(Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)

        self.linear1 = Linear(d_model, dim_feedforward)
        
        self.activation = F.relu
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src, src_mask):
        x = src

        x = self.norm1(x + self._sa_block(x, src_mask))
        x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, src_mask):
        x = self.self_attn(x, x, x, mask=src_mask)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.multihead_attn = MultiheadAttention(d_model, nhead)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.activation = F.relu
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        x = tgt

        x = self.norm1(x + self._sa_block(x, tgt_mask))
        x = self.norm2(x + self._mha_block(x, memory, memory_mask))
        x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, tgt_mask):
        x = self.self_attn(x, x, x, mask=tgt_mask)

        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x, mem, mem_mask):
        x = self.multihead_attn(x, mem, mem, mask=mem_mask)
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])