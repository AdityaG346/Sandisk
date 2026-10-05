"""
src/calibration.py
==================
Probability calibration utilities for die yield risk models.
Provides Platt scaling (logistic regression on logit of raw score),
Expected Calibration Error (ECE) with equal-frequency bins, and Brier score.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Safely compute logit transform of probabilities."""
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


def fit_platt(y_true: np.ndarray, y_prob: np.ndarray) -> LogisticRegression:
    """
    Fit Platt scaling (logistic regression on logit of raw probability score).
    Must be fit ONLY on eligible validation dies.
    """
    z = logit(y_prob).reshape(-1, 1)
    # C=1e5 (effectively unregularized) for classical 2-parameter Platt scaling
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    calibrator.fit(z, y_true)
    return calibrator


def apply(calibrator: LogisticRegression, y_prob: np.ndarray) -> np.ndarray:
    """
    Apply fitted Platt calibrator to raw probability scores.
    Preserves strict monotonicity if slope > 0.
    """
    z = logit(y_prob).reshape(-1, 1)
    return calibrator.predict_proba(z)[:, 1]


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) using n_bins equal-frequency (quantile) bins.
    
    ECE = sum_{b=1}^B (N_b / N) * |acc(b) - conf(b)|
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return 0.0

    # Equal-frequency binning via quantiles
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.percentile(y_prob, quantiles * 100)
    # Ensure strictly monotonic bin edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) <= 1:
        return float(np.abs(np.mean(y_true) - np.mean(y_prob)))

    # Assign samples to bins
    bin_ids = np.digitize(y_prob, bin_edges[1:-1])
    
    total_ece = 0.0
    for b in range(len(bin_edges) - 1):
        mask = (bin_ids == b)
        n_b = np.sum(mask)
        if n_b > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            total_ece += (n_b / n) * np.abs(bin_acc - bin_conf)

    return float(total_ece)


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score loss: mean((y_prob - y_true)^2)."""
    return float(brier_score_loss(y_true, y_prob))
