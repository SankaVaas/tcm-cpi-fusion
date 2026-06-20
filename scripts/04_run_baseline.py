#!/usr/bin/env python
"""
Step 4: Train Random Forest baseline (Morgan FP).
Runs in ~10 min on local CPU. Establishes the comparison bar.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd, numpy as np, yaml
from pathlib import Path

from src.features.fingerprints import morgan_fp
from src.models.baseline import BaselineCPI
from src.evaluation.metrics import compute_metrics, enrichment_factor
from src.utils.io import save_json
from src.utils.seed import set_seed

CFG = yaml.safe_load(open("experiments/configs/baseline.yaml"))
set_seed(42)

def featurise(df: pd.DataFrame, nbits: int = 2048):
    X = np.vstack([morgan_fp(s, nbits=nbits) for s in df.smiles])
    y = df.label.values
    return X, y

if __name__ == "__main__":
    train = pd.read_csv("data/processed/splits/train.csv")
    val   = pd.read_csv("data/processed/splits/val.csv")
    test  = pd.read_csv("data/processed/splits/test.csv")

    nbits = CFG["baseline"]["morgan_nbits"]
    X_tr, y_tr = featurise(train, nbits)
    X_v,  y_v  = featurise(val,   nbits)
    X_te, y_te = featurise(test,  nbits)

    clf = BaselineCPI(n_estimators=CFG["baseline"]["n_estimators"],
                      max_depth=CFG["baseline"]["max_depth"])
    clf.fit(X_tr, y_tr)

    for split_name, X, y in [("val", X_v, y_v), ("test", X_te, y_te)]:
        proba  = clf.predict_proba(X)
        m      = compute_metrics(y, proba)
        m["ef_1pct"] = enrichment_factor(y, proba, fraction=0.01)
        print(f"[baseline/{split_name}] {m}")
        save_json(m, Path(f"experiments/results/baseline_{split_name}.json"))

    clf.save("experiments/checkpoints/baseline_rf.pkl")
    print("Baseline complete.")
