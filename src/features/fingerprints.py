"""RDKit molecular fingerprints and descriptors for baseline models."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def morgan_fp(smiles: str, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp)


def rdkit_descriptors(smiles: str, n: int = 200) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n)
    desc_names = [d[0] for d in Descriptors._descList][:n]
    vals = []
    for name in desc_names:
        try:
            vals.append(float(getattr(Descriptors, name)(mol)))
        except Exception:
            vals.append(0.0)
    return np.array(vals)
