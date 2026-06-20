"""Evaluation metrics for CPI prediction."""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix,
)


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_pred = (np.array(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc":     roc_auc_score(y_true, y_score),
        "pr_auc":      average_precision_score(y_true, y_score),
        "mcc":         matthews_corrcoef(y_true, y_pred),
        "sensitivity": tp / (tp + fn + 1e-8),
        "specificity": tn / (tn + fp + 1e-8),
        "precision":   tp / (tp + fp + 1e-8),
        "f1":          2 * tp / (2 * tp + fp + fn + 1e-8),
    }


def enrichment_factor(y_true, y_score, fraction: float = 0.01) -> float:
    """
    EF@{fraction*100}%: ratio of actives in top-fraction vs random expectation.
    EF = 15x means you find 15x more actives in the top 1% than random selection.
    """
    n_top = max(1, int(len(y_score) * fraction))
    idx   = np.argsort(y_score)[::-1][:n_top]
    hits  = np.array(y_true)[idx].sum()
    total_actives = np.array(y_true).sum()
    expected = total_actives / len(y_true)
    return (hits / n_top) / (expected + 1e-8)
