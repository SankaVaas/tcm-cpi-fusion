import numpy as np
import pytest


def test_baseline_fit_predict():
    from src.models.baseline import BaselineCPI
    X = np.random.rand(200, 2048)
    y = np.array([1] * 100 + [0] * 100)
    clf = BaselineCPI(n_estimators=10)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (200,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_baseline_evaluate():
    from src.models.baseline import BaselineCPI
    X = np.random.rand(100, 2048)
    y = np.array([1] * 50 + [0] * 50)
    clf = BaselineCPI(n_estimators=5)
    clf.fit(X, y)
    m = clf.evaluate(X, y)
    assert "roc_auc" in m
    assert 0.0 <= m["roc_auc"] <= 1.0
