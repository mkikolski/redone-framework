import torch
from torch import nn, Tensor
import numpy as np


class SinEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.sz = dim

    def forward(self, x: Tensor) -> Tensor:
        embedding = np.log(10000) / (self.dim // 2 - 1)
        embedding = torch.exp(torch.arange(self.dim // 2, device=x.device) * -embedding)
        embedding = x[:, None] * embedding[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)
