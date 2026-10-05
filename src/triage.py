"""
src/triage.py
=============
Pure functions for budget-aware screening triage of semiconductor dies.
Implements top-k statistics and capture curves for risk-prioritized testing.
Ordering matches audit_numbers.py convention (np.argsort(score)[::-1]).
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Any, Optional


def topk_stats(
    y: np.ndarray,
    score: np.ndarray,
    k_frac: float,
    score_cal: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute triage statistics for top k_frac of dies prioritized by score (descending).
    
    Parameters:
        y: Binary ground truth labels (1 = fail, 0 = pass) for eligible dies
        score: Risk scores or probabilities used to rank dies
        k_frac: Fraction of total eligible dies to screen (e.g. 0.05 for 5%)
        score_cal: Optional calibrated probabilities to compute mean calibrated rate
        
    Returns:
        Dict containing:
            n_total: Total eligible dies
            n_screened: Dies screened (int(round(n_total * k_frac)))
            n_fails_total: Total true failures in dataset
            n_fails_captured: Failures captured in screened subset
            capture_rate: Fraction of all true failures captured (% recall)
            observed_fail_rate: Observed failure rate (% precision)
            precision: Same as observed_fail_rate
            base_rate: Overall failure prevalence in eligible dies
            lift: Factor increase over random screening (precision / base_rate)
            mean_score: Mean ranking score among screened dies
            mean_calibrated_rate: Mean calibrated probability (if score_cal provided)
    """
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n = len(y)
    if n == 0:
        return {
            "n_total": 0, "n_screened": 0, "n_fails_total": 0, "n_fails_captured": 0,
            "capture_rate": 0.0, "observed_fail_rate": 0.0, "precision": 0.0,
            "base_rate": 0.0, "lift": 1.0, "mean_score": 0.0, "mean_calibrated_rate": None
        }

    k = max(1, min(n, int(round(n * k_frac))))
    order = np.argsort(score)[::-1]
    top_indices = order[:k]
    
    top_y = y[top_indices]
    n_fails = int(np.sum(y))
    n_captured = int(np.sum(top_y))
    
    capture_rate = float(n_captured / n_fails) if n_fails > 0 else 0.0
    precision = float(n_captured / k)
    base_rate = float(n_fails / n) if n > 0 else 0.0
    lift = float(precision / base_rate) if base_rate > 0 else 1.0
    mean_score = float(np.mean(score[top_indices]))
    
    mean_cal = None
    if score_cal is not None:
        mean_cal = float(np.mean(score_cal[top_indices]))

    return {
        "n_total": n,
        "n_screened": k,
        "k_frac": float(k / n),
        "n_fails_total": n_fails,
        "n_fails_captured": n_captured,
        "capture_rate": capture_rate,
        "observed_fail_rate": precision,
        "precision": precision,
        "base_rate": base_rate,
        "lift": lift,
        "mean_score": mean_score,
        "mean_calibrated_rate": mean_cal,
    }


def capture_curve(
    y: np.ndarray,
    score: np.ndarray,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cumulative capture (gains) curve across screening fractions.
    
    Parameters:
        y: Binary ground truth labels (1 = fail, 0 = pass)
        score: Risk scores or probabilities
        n_points: Number of evaluation points along the curve (0.0 to 1.0)
        
    Returns:
        k_fracs: Array of screening fractions from 0.0 to 1.0
        capture_rates: Fraction of total failures captured at each k_frac
        precisions: Precision at each k_frac
    """
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n = len(y)
    n_fails = int(np.sum(y))
    
    if n == 0 or n_fails == 0:
        pts = np.linspace(0.0, 1.0, n_points)
        return pts, pts.copy(), np.zeros_like(pts)
        
    order = np.argsort(score)[::-1]
    cum_fails = np.cumsum(y[order])
    
    k_indices = np.linspace(1, n, n_points, dtype=int)
    k_fracs = k_indices / n
    capture_rates = cum_fails[k_indices - 1] / n_fails
    precisions = cum_fails[k_indices - 1] / k_indices
    
    # Prepend 0.0 point
    k_fracs = np.concatenate(([0.0], k_fracs))
    capture_rates = np.concatenate(([0.0], capture_rates))
    precisions = np.concatenate(([precisions[0]], precisions))
    
    return k_fracs, capture_rates, precisions
