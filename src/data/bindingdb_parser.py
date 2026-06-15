"""Parse BindingDB TSV for measured Kd / Ki / IC50 values."""
import pandas as pd
from pathlib import Path

COLS = {
    "Ligand SMILES": "smiles",
    "Target Name": "target_name",
    "UniProt (SwissProt) Primary ID of Target Chain": "uniprot_id",
    "Ki (nM)": "ki_nm",
    "Kd (nM)": "kd_nm",
    "IC50 (nM)": "ic50_nm",
}


def load_bindingdb(path: Path, nrows: int = None) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", usecols=list(COLS.keys()),
        nrows=nrows, low_memory=False
    )
    return df.rename(columns=COLS)


def affinity_label(df: pd.DataFrame, threshold_nm: float = 1000.0) -> pd.DataFrame:
    """
    Binary label: 1 = active (best affinity <= threshold nM), 0 = inactive.
    Uses minimum of Ki, Kd, IC50 for each compound-target pair.
    """
    numeric = df[["ki_nm", "kd_nm", "ic50_nm"]].apply(pd.to_numeric, errors="coerce")
    df = df.copy()
    df["affinity_nm"] = numeric.min(axis=1)
    df["label"] = (df["affinity_nm"] <= threshold_nm).astype(int)
    return df.dropna(subset=["smiles", "uniprot_id"]).reset_index(drop=True)
