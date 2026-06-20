# Lazy imports — heavy deps (esm, torch_geometric) only loaded when actually used.
# This prevents test collection crashes when optional packages are missing.

def mol_graph():
    from .mol_graph import smiles_to_graph, batch_smiles_to_graphs
    return smiles_to_graph, batch_smiles_to_graphs

def protein_embed():
    from .protein_embed import embed_proteins_esm2
    return embed_proteins_esm2

def fingerprints():
    from .fingerprints import morgan_fp, rdkit_descriptors
    return morgan_fp, rdkit_descriptors

# Direct imports for convenience (still lazy at module level)
from .fingerprints import morgan_fp, rdkit_descriptors
from .mol_graph import smiles_to_graph, batch_smiles_to_graphs
