import pytest


def test_valid_smiles():
    from src.features.mol_graph import smiles_to_graph
    g = smiles_to_graph("CCO")
    assert g is not None
    assert g.x.shape[0] == 3


def test_invalid_smiles():
    from src.features.mol_graph import smiles_to_graph
    g = smiles_to_graph("INVALID$$##")
    assert g is None


def test_benzene_edges():
    from src.features.mol_graph import smiles_to_graph
    g = smiles_to_graph("c1ccccc1")
    assert g.x.shape[0] == 6
    assert g.edge_index.shape[1] == 12


def test_feature_dim():
    from src.features.mol_graph import smiles_to_graph
    g = smiles_to_graph("CC(=O)O")
    assert g.x.shape[1] == 6
