"""Download PDB structures for target proteins from RCSB."""
import requests
from pathlib import Path

PDB_DIR = Path("data/raw/pdb")


def fetch_pdb_structure(pdb_id: str) -> Path:
    pdb_id = pdb_id.upper()
    out = PDB_DIR / f"{pdb_id}.pdb"
    if out.exists():
        print(f"[cache] {pdb_id}.pdb")
        return out
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r   = requests.get(url, timeout=60)
    r.raise_for_status()
    out.write_text(r.text)
    print(f"Downloaded {pdb_id}.pdb ({len(r.text)//1024} KB)")
    return out
