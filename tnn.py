import torch
from torch import nn, Tensor

from embedding import SinEmbedding


class DecoderOnlyBlock(nn.Module):
    def __init__(self, hidden_size: int = 256, nheads: int = 6):
        super(DecoderOnlyBlock, self).__init__()

        self.input_norm = nn.LayerNorm(hidden_size)
        self.mha = nn.MultiheadAttention(hidden_size, num_heads=nheads, batch_first=True, dropout=0.162)
        self.output_norm = nn.LayerNorm(hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        batch_size, sz, _ = x.shape
        ss_mask = torch.triu(torch.ones(sz, sz, device=x.device), sz).bool()

        norm = self.input_norm(x)
        x = self.mha(norm, norm, norm, attn_mask=mask, key_padding_mask=ss_mask)[0] + x
        norm = self.output_norm(x)

        x = self.mlp(norm) + x
        return x


class TNN(nn.Module):
    def __init__(self, n_embeddings: int, hidden_size: int = 256, n_layers: int = 6, nheads: int = 6):
        super(TNN, self).__init__()

        self.embedding = nn.Embedding(n_embeddings, hidden_size)
        self.positional_embedding = SinEmbedding(hidden_size)

        self.layers = nn.ModuleList([
            DecoderOnlyBlock(hidden_size, nheads) for _ in range(n_layers)
        ])

        self.fc = nn.Linear(hidden_size, n_embeddings)

    def forward(self, x: Tensor) -> Tensor:
        padding_mask = x == 0
        input_embeddings = self.embedding(x)
        batch_size, sz, h = input_embeddings.shape

        idxs = torch.arange(sz, device=x.device)
        positional = self.positional_embedding(idxs).reshape(1, sz, h).expand(batch_size, sz, h)
        embeddings = input_embeddings + positional

        for layer in self.layers:
            embeddings = layer(embeddings, padding_mask=padding_mask)

        return self.fc(embeddings)
