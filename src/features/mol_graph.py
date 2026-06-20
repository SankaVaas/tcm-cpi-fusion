"""Convert SMILES strings to PyTorch Geometric graph Data objects."""
import torch
from rdkit import Chem
from torch_geometric.data import Data

HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP:   0,
    Chem.rdchem.HybridizationType.SP2:  1,
    Chem.rdchem.HybridizationType.SP3:  2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2:4,
}


def atom_features(atom) -> list:
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        HYBRIDIZATION_MAP.get(atom.GetHybridization(), 5),
        int(atom.GetIsAromatic()),
    ]


def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    bonds = mol.GetBonds()
    if not bonds:
        return Data(x=x, edge_index=torch.zeros((2, 0), dtype=torch.long), smiles=smiles)
    src = [b.GetBeginAtomIdx() for b in bonds] + [b.GetEndAtomIdx() for b in bonds]
    dst = [b.GetEndAtomIdx() for b in bonds]   + [b.GetBeginAtomIdx() for b in bonds]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, smiles=smiles)


def batch_smiles_to_graphs(smiles_list: list) -> dict:
    graphs = {}
    for smi in smiles_list:
        g = smiles_to_graph(smi)
        if g is not None:
            graphs[smi] = g
    return graphs
