"""Fetch canonical SMILES from PubChem by compound name or CAS number."""
import pubchempy as pcp


def fetch_smiles_pubchem(name: str) -> str | None:
    results = pcp.get_compounds(name, "name")
    if not results:
        return None
    return results[0].canonical_smiles
