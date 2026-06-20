#!/usr/bin/env python
"""
Step 5: Train DualStreamCPI (ChemBERTa + GCN + ESM-2).
Recommended: run on Colab T4 GPU (~2 hrs for 50 epochs).
Local CPU run is possible but slow (~24 hrs).
"""
import sys
sys.path.insert(0, ".")

import pickle, yaml, torch, pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.models.dual_stream import DualStreamCPI
from src.features.mol_graph import batch_smiles_to_graphs
from src.data.dataset import CPIDataset
from src.training.trainer import CPITrainer
from src.evaluation.metrics import compute_metrics, enrichment_factor
from src.utils.seed import set_seed
from src.utils.io import save_checkpoint, save_json

CFG    = yaml.safe_load(open("experiments/configs/default.yaml"))
DEVICE = CFG["device"] if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")
set_seed(CFG["seed"])

if __name__ == "__main__":
    train = pd.read_csv("data/processed/splits/train.csv")
    val   = pd.read_csv("data/processed/splits/val.csv")
    test  = pd.read_csv("data/processed/splits/test.csv")

    with open("data/processed/embeddings/esm2_8m.pkl", "rb") as f:
        prot_emb = pickle.load(f)

    tok     = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    all_smi = pd.concat([train, val, test]).smiles.unique().tolist()
    graphs  = batch_smiles_to_graphs(all_smi)

    train_ds = CPIDataset(train, tok, graphs, prot_emb)
    val_ds   = CPIDataset(val,   tok, graphs, prot_emb)
    test_ds  = CPIDataset(test,  tok, graphs, prot_emb)

    bs = CFG["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=2)

    model = DualStreamCPI(
        gcn_out=CFG["model"]["gcn_out"],
        prot_out=CFG["model"]["prot_out"],
        bert_dim=CFG["model"]["bert_dim"],
        dropout=CFG["model"]["dropout"],
    )
    opt   = AdamW(model.parameters(),
                  lr=CFG["training"]["lr"],
                  weight_decay=CFG["training"]["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=CFG["training"]["epochs"])

    trainer = CPITrainer(model, opt, sched, device=DEVICE, use_wandb=False)
    trainer.fit(train_loader, val_loader, epochs=CFG["training"]["epochs"])

    ckpt_path = Path("experiments/checkpoints/dual_stream_best.pt")
    save_checkpoint(model, opt, CFG["training"]["epochs"], {}, ckpt_path)

    print("Evaluating on test set...")
    metrics = trainer.evaluate(test_loader)
    save_json(metrics, Path("experiments/results/dual_stream_test.json"))
    print(f"Test metrics: {metrics}")
