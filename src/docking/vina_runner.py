"""
CPU-native AutoDock Vina docking for validating top-predicted hits.
Requirements: pip install vina meeko
"""
import subprocess
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation

DOCKING_DIR = Path("data/processed/docking")


def smiles_to_pdbqt(smiles: str, out_path: Path) -> Path:
    """Prepare ligand PDBQT using RDKit 3D embedding + Meeko."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    mk_prep = MoleculePreparation()
    mk_prep.prepare(mol)
    mk_prep.write_pdbqt_file(str(out_path))
    return out_path


def run_vina(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center: tuple,
    box_size: tuple = (20, 20, 20),
    exhaustiveness: int = 8,
    out_pdbqt: str = None,
) -> dict:
    """Run AutoDock Vina; return best binding affinity (kcal/mol)."""
    cx, cy, cz = center
    sx, sy, sz = box_size
    out = out_pdbqt or ligand_pdbqt.replace(".pdbqt", "_out.pdbqt")
    log = ligand_pdbqt.replace(".pdbqt", "_log.txt")
    cmd = [
        "vina",
        "--receptor", receptor_pdbqt,
        "--ligand",   ligand_pdbqt,
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x",   str(sx), "--size_y",   str(sy), "--size_z",   str(sz),
        "--exhaustiveness", str(exhaustiveness),
        "--out", out,
        "--log", log,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return _parse_vina_log(proc.stdout + proc.stderr)


def _parse_vina_log(text: str) -> dict:
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == "1":
            return {"best_affinity_kcal_mol": float(parts[1])}
    return {"best_affinity_kcal_mol": None}


def run_vina_batch(
    hits: list,
    receptor_pdbqt: str,
    center: tuple,
    box_size: tuple = (20, 20, 20),
    exhaustiveness: int = 8,
) -> list:
    """
    Dock a list of (smiles, ml_score) tuples.
    Returns list of dicts with smiles, ml_score, best_affinity_kcal_mol.
    """
    DOCKING_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (smiles, ml_score) in enumerate(hits):
        lig_path = DOCKING_DIR / f"ligand_{i:04d}.pdbqt"
        try:
            smiles_to_pdbqt(smiles, lig_path)
            dock = run_vina(receptor_pdbqt, str(lig_path), center,
                            box_size=box_size, exhaustiveness=exhaustiveness)
            results.append({"smiles": smiles, "ml_score": ml_score, **dock})
        except Exception as e:
            results.append({"smiles": smiles, "ml_score": ml_score,
                            "best_affinity_kcal_mol": None, "error": str(e)})
    return results
