"""
src/anomaly_features.py
========================
Isolation Forest anomaly scores -- trained ONLY on old_label=0 dies.
Two separate detectors:
  1. Die-level: trained on parametric features (used by Model A and B)
  2. Block-level: trained on block aggregate statistics (used by Model B only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path
from typing import Optional, List


SEED = 42


class DieAnomalyDetector:
    """
    Isolation Forest trained on old_label=0 die parametric features.
    Produces a continuous anomaly score (higher = more anomalous).
    """

    def __init__(self, n_estimators: int = 200, contamination: float = 0.05):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self._model: Optional[IsolationForest] = None

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> "DieAnomalyDetector":
        """
        Fit on old_label=0 dies only.
        """
        train_mask = df["old_label"] == 0
        X_train = df.loc[train_mask, feature_cols].values
        print(f"[anomaly] Fitting die anomaly detector on {train_mask.sum():,} healthy dies...")
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=SEED,
            n_jobs=-1,
        )
        self._model.fit(X_train)
        return self

    def score(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        Return anomaly scores for all dies.
        IsolationForest.score_samples returns negative avg path length;
        we negate so higher = more anomalous.
        """
        if self._model is None:
            raise RuntimeError("Must call fit() before score()")
        X = df[feature_cols].values
        return -self._model.score_samples(X).astype(np.float32)

    def save(self, path: str) -> None:
        joblib.dump(self._model, path)

    def load(self, path: str) -> "DieAnomalyDetector":
        self._model = joblib.load(path)
        return self


class BlockAnomalyDetector:
    """
    Isolation Forest trained on block aggregate statistics of old_label=0 dies.
    """

    def __init__(self, n_estimators: int = 200, contamination: float = 0.05):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self._model: Optional[IsolationForest] = None

    def fit(
        self,
        df: pd.DataFrame,
        block_feat_df: pd.DataFrame,
        block_feature_cols: List[str],
    ) -> "BlockAnomalyDetector":
        """
        df must contain 'old_label'; block_feat_df has same index.
        Trains only on old_label=0 dies.
        """
        train_mask = (df["old_label"].values == 0)
        X_train = block_feat_df.loc[train_mask, block_feature_cols].values
        print(f"[anomaly] Fitting block anomaly detector on {train_mask.sum():,} healthy dies...")
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=SEED,
            n_jobs=-1,
        )
        self._model.fit(X_train)
        return self

    def score(self, block_feat_df: pd.DataFrame, block_feature_cols: List[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Must call fit() before score()")
        X = block_feat_df[block_feature_cols].values
        return -self._model.score_samples(X).astype(np.float32)

    def save(self, path: str) -> None:
        joblib.dump(self._model, path)

    def load(self, path: str) -> "BlockAnomalyDetector":
        self._model = joblib.load(path)
        return self
