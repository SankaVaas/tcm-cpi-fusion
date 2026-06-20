"""
DualStreamCPI — the main model.

Architecture:
    compound SMILES → ChemBERTa [768-d]  ─┐
    compound graph  → GCN       [256-d]  ─┼─ concat [1280-d] → MLP head → score
    protein seq     → ESM-2 proj[256-d]  ─┘
"""
import torch
import torch.nn as nn
from .components import GCNEncoder, ChemBERTaEncoder, ProteinEncoder


class DualStreamCPI(nn.Module):
    def __init__(
        self,
        gcn_out:  int   = 256,
        prot_out: int   = 256,
        bert_dim: int   = 768,
        dropout:  float = 0.3,
    ):
        super().__init__()
        self.chem_bert = ChemBERTaEncoder(freeze=False)
        self.gcn       = GCNEncoder(out_dim=gcn_out)
        self.prot_enc  = ProteinEncoder(out_dim=prot_out)

        fused_dim = bert_dim + gcn_out + prot_out   # 1280
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, smiles_tokens: dict, graph_batch, prot_emb: torch.Tensor):
        bert_emb = self.chem_bert(
            input_ids=smiles_tokens["input_ids"],
            attention_mask=smiles_tokens["attention_mask"],
        )                                                      # [B, 768]
        gcn_emb  = self.gcn(graph_batch.x,
                            graph_batch.edge_index,
                            graph_batch.batch)                 # [B, 256]
        prot_out = self.prot_enc(prot_emb)                    # [B, 256]
        fused    = torch.cat([bert_emb, gcn_emb, prot_out], dim=-1)  # [B, 1280]
        return self.head(fused).squeeze(-1)                   # [B]
