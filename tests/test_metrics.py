import numpy as np


def test_compute_metrics_perfect():
    from src.evaluation.metrics import compute_metrics
    y = [1, 1, 0, 0]
    s = [0.9, 0.8, 0.2, 0.1]
    m = compute_metrics(y, s)
    assert m["roc_auc"] == 1.0


def test_enrichment_factor():
    from src.evaluation.metrics import enrichment_factor
    y = [1] * 10 + [0] * 90
    s = list(range(100, 0, -1))
    ef = enrichment_factor(y, s, fraction=0.10)
    assert ef > 1.0
