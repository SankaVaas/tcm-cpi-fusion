"""
Parse TCMSP bulk CSV exports (downloaded manually from tcmsp-e.com).
Produces herb-compound-target triples with ADME properties.
"""
import pandas as pd
from pathlib import Path

TCMSP_DIR = Path("data/raw/tcmsp")


def load_herb_compounds(path: Path = TCMSP_DIR / "herb_compounds.csv") -> pd.DataFrame:
    """
    Expected columns: Herb_name, mol_id, Molecule_name, OB, DL, CAS, SMILES
    """
    df = pd.read_csv(path)
    required = {"Herb_name", "mol_id", "OB", "DL", "SMILES"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in TCMSP file: {missing}")
    return df


def adme_filter(df: pd.DataFrame, ob: float = 30.0, dl: float = 0.18) -> pd.DataFrame:
    """Standard TCMSP thresholds: Oral Bioavailability >= 30%, Drug-likeness >= 0.18"""
    return df[(df["OB"] >= ob) & (df["DL"] >= dl)].reset_index(drop=True)


def load_compound_targets(path: Path = TCMSP_DIR / "compound_targets.csv") -> pd.DataFrame:
    """Expected columns: mol_id, Target_name, UniProt_ID"""
    return pd.read_csv(path)


def build_cpi_triples(compounds: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    merged = compounds.merge(targets, on="mol_id", how="inner")
    merged = merged.rename(columns={"SMILES": "smiles", "UniProt_ID": "uniprot_id"})
    merged["label"] = 1  # all TCM-target pairs from TCMSP are known interactions
    return merged[["smiles", "uniprot_id", "label", "Herb_name", "Molecule_name", "OB", "DL"]]
