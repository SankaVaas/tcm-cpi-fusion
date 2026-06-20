"""Encoder building blocks for DualStreamCPI."""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoModel


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int = 6, hidden: int = 128, out_dim: int = 256):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.act   = nn.ReLU()
        self.drop  = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        x = self.act(self.conv1(x, edge_index))
        x = self.drop(x)
        x = self.act(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)   # [B, out_dim]


class ChemBERTaEncoder(nn.Module):
    CHECKPOINT = "seyonec/ChemBERTa-zinc-base-v1"

    def __init__(self, freeze: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(self.CHECKPOINT)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output   # [B, 768]


class ProteinEncoder(nn.Module):
    """Project pre-computed ESM-2 mean-pool embeddings to a shared latent space."""
    def __init__(self, esm_dim: int = 320, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(esm_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        return self.proj(x)
