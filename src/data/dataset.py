"""PyTorch Dataset wrapping CPI pairs for training."""
import torch
from torch.utils.data import Dataset


class CPIDataset(Dataset):
    def __init__(self, records, smiles_tokenizer, mol_graphs: dict, prot_embeddings: dict,
                 max_smiles_len: int = 128):
        self.records = records.reset_index(drop=True)
        self.tokenizer = smiles_tokenizer
        self.graphs = mol_graphs          # {smiles: PyG Data}
        self.prot_emb = prot_embeddings   # {uniprot_id: tensor}
        self.max_len = max_smiles_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        tokens = self.tokenizer(
            row.smiles, return_tensors="pt",
            padding="max_length", max_length=self.max_len, truncation=True
        )
        # Remove batch dim added by return_tensors="pt"
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        graph = self.graphs.get(row.smiles)
        prot  = self.prot_emb.get(row.uniprot_id, torch.zeros(320))
        label = torch.tensor(row.label, dtype=torch.float)
        return tokens, graph, prot, label
