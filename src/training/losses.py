"""Custom loss functions for imbalanced CPI datasets."""
import torch
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, labels: torch.Tensor,
               alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    bce   = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    pt    = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def weighted_bce(logits: torch.Tensor, labels: torch.Tensor,
                 pos_weight: float = 5.0) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits, labels,
        pos_weight=torch.tensor([pos_weight], device=logits.device),
    )
