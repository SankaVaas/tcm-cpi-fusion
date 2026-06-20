#!/usr/bin/env python
"""
Step 2: Parse TCMSP + BindingDB → merged CPI dataframe → train/val/test CSV splits.
Runtime: ~15 min on local CPU with 500k BindingDB rows.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.tcmsp_parser import (load_herb_compounds, adme_filter,
                                    load_compound_targets, build_cpi_triples)
from src.data.bindingdb_parser import load_bindingdb, affinity_label
from src.utils.seed import set_seed

set_seed(42)

if __name__ == "__main__":
    print("Loading TCMSP...")
    compounds = adme_filter(load_herb_compounds())
    targets   = load_compound_targets()
    tcm_pairs = build_cpi_triples(compounds, targets)
    print(f"  TCM CPI pairs (post ADME filter): {len(tcm_pairs)}")

    print("Loading BindingDB (500k rows)...")
    bdb_raw = load_bindingdb(
        Path("data/raw/bindingdb/BindingDB_All_2D_v2.tsv"), nrows=500_000
    )
    bdb = affinity_label(bdb_raw)
    print(f"  BindingDB actives: {bdb.label.sum()} / {len(bdb)}")

    combined = pd.concat(
        [tcm_pairs[["smiles","uniprot_id","label"]],
         bdb[["smiles","uniprot_id","label"]]],
        ignore_index=True,
    ).dropna().drop_duplicates(subset=["smiles","uniprot_id"])

    train, test = train_test_split(combined, test_size=0.10,
                                   stratify=combined.label, random_state=42)
    train, val  = train_test_split(train,    test_size=0.10,
                                   stratify=train.label,    random_state=42)

    out = Path("data/processed/splits")
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(  out / "val.csv",   index=False)
    test.to_csv( out / "test.csv",  index=False)
    print(f"Splits saved — train:{len(train)}  val:{len(val)}  test:{len(test)}")
