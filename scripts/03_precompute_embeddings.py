#!/usr/bin/env python
"""
Step 3: Fetch UniProt sequences + compute ESM-2 embeddings.
Tip: run on Colab T4 for speed; esm2_t6_8M works on local CPU (~2h).
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
from pathlib import Path

from src.features.protein_embed import embed_proteins_esm2
from src.utils.uniprot import fetch_sequence_uniprot

if __name__ == "__main__":
    uniprots = set()
    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"data/processed/splits/{split}.csv")
        uniprots.update(df.uniprot_id.dropna().unique())
    print(f"Fetching sequences for {len(uniprots)} unique proteins...")

    seqs = {}
    for i, uid in enumerate(uniprots):
        seq = fetch_sequence_uniprot(uid)
        if seq:
            seqs[uid] = seq
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(uniprots)} fetched")

    print(f"Computing ESM-2 embeddings for {len(seqs)} sequences...")
    embed_proteins_esm2(
        seqs,
        save_path=Path("data/processed/embeddings/esm2_8m.pkl"),
        model_name="esm2_t6_8M_UR50D",
    )
    print("Done.")
