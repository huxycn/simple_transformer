import torch

from torch.nn import Module, Parameter


class LayerNorm(Module):
    def __init__(self, embed_dim, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = Parameter(torch.ones(embed_dim))
        self.beta = Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
