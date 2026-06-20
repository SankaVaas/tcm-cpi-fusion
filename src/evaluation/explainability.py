import torch
import numpy as np


def grad_cam_atoms(model, smiles_tokens: dict, graph, prot_emb: torch.Tensor,
                   device: str = "cpu") -> np.ndarray:
    """
    Args:
        model: trained DualStreamCPI
        smiles_tokens: tokenizer output for one SMILES
        graph: PyG Data for the same molecule
        prot_emb: [1, esm_dim] protein embedding
    Returns:
        atom_scores: np.ndarray [n_atoms] — higher = more important
    """
    model.eval()
    graph = graph.to(device)
    graph.x = graph.x.detach().requires_grad_(True)
    prot_emb = prot_emb.to(device)
    smiles_tokens = {k: v.to(device) for k, v in smiles_tokens.items()}

    logit = model(smiles_tokens, graph, prot_emb)
    logit.backward()

    grad    = graph.x.grad                           # [n_atoms, feat_dim]
    weights = grad.abs().mean(dim=-1, keepdim=True)  # [n_atoms, 1]
    scores  = (graph.x * weights).sum(dim=-1)        # [n_atoms]
    return scores.detach().cpu().numpy()
