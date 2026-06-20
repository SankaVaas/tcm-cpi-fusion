"""
Compute ESM-2 protein embeddings.
Requires: pip install fair-esm
Use esm2_t6_8M_UR50D on local CPU; esm2_t12_35M_UR50D or larger on Colab T4.
"""
import pickle
import torch
from pathlib import Path


def embed_proteins_esm2(
    sequences: dict,
    save_path: Path = None,
    model_name: str = "esm2_t6_8M_UR50D",
) -> dict:
    """
    Args:
        sequences: {uniprot_id: amino_acid_sequence}
        save_path: if given, pickle the embeddings dict here
        model_name: ESM-2 checkpoint — 8M fits on CPU, 150M needs Colab GPU
    Returns:
        {uniprot_id: mean-pooled embedding tensor [embed_dim]}
    """
    try:
        import esm
    except ImportError:
        raise ImportError(
            "ESM-2 not installed. Run: pip install fair-esm\n"
            "Or on Windows: pip install git+https://github.com/facebookresearch/esm.git"
        )

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    data = [(uid, seq[:1022]) for uid, seq in sequences.items()]
    _, _, tokens = batch_converter(data)
    repr_layer = int(model_name.split("_t")[1].split("_")[0])

    embeddings = {}
    with torch.no_grad():
        results = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    for i, (uid, _) in enumerate(data):
        emb = results["representations"][repr_layer][i, 1:-1]
        embeddings[uid] = emb.mean(0)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Saved embeddings to {save_path}")

    return embeddings
