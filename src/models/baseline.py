"""
Baseline: Morgan fingerprint + Random Forest.
Provides the comparison point for DualStreamCPI.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib


class BaselineCPI:
    def __init__(self, n_estimators: int = 200, max_depth: int = 20, n_jobs: int = -1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=42,
            class_weight="balanced",
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        proba = self.predict_proba(X)
        return {
            "roc_auc": roc_auc_score(y, proba),
            "pr_auc":  average_precision_score(y, proba),
        }

    def save(self, path: str):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str):
        obj = cls()
        obj.model = joblib.load(path)
        return obj
