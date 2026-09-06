"""
src/block_features.py
======================
Engineer per-die aggregate features from 2000 block readings.
Does NOT concatenate raw readings -- only summary statistics.

Known block reading parameters (from config.yaml):
  base_mean = 100.0
  base_std  = 15.0
  anomalous fraction ~= 5% of blocks per failing die
  anomalous shift ~= 0.3 * base_std = 4.5 units (N(4.5, 1.35))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import List

from src.data_loader import assert_aligned


# Reference parameters from config.yaml
BLOCK_BASE_MEAN = 100.0
BLOCK_BASE_STD = 15.0
ANOMALY_THRESHOLD_K = 3.0    # k * base_std from base_mean (wide, for reference)
ANOMALY_MAD_K = 2.0          # k * MAD from median (tighter, better for this dataset)


# ---------------------------------------------------------------------------
# Feature extraction for a single die's block readings
# ---------------------------------------------------------------------------

def _extract_one(readings: np.ndarray) -> dict:
    """Extract all block features for one die."""
    n = len(readings)
    if n == 0:
        return {}

    # --- Statistical features ---
    mean_ = float(np.mean(readings))
    std_ = float(np.std(readings))
    median_ = float(np.median(readings))
    q25 = float(np.percentile(readings, 25))
    q75 = float(np.percentile(readings, 75))
    iqr_ = q75 - q25
    min_ = float(np.min(readings))
    max_ = float(np.max(readings))
    skew_ = float(stats.skew(readings)) if n > 2 else 0.0
    kurt_ = float(stats.kurtosis(readings)) if n > 2 else 0.0

    # --- Tail / anomaly features (fixed threshold: base_mean ± k*base_std) ---
    lo_thresh = BLOCK_BASE_MEAN - ANOMALY_THRESHOLD_K * BLOCK_BASE_STD  # 55.0
    hi_thresh = BLOCK_BASE_MEAN + ANOMALY_THRESHOLD_K * BLOCK_BASE_STD  # 145.0
    anom_mask_fixed = (readings < lo_thresh) | (readings > hi_thresh)
    n_anom_fixed = int(anom_mask_fixed.sum())
    frac_anom_fixed = n_anom_fixed / n

    # --- Tail / anomaly features (MAD-based: robust) ---
    mad = float(np.median(np.abs(readings - median_)))
    if mad > 0:
        anom_mask_mad = np.abs(readings - median_) > ANOMALY_MAD_K * mad
    else:
        anom_mask_mad = np.zeros(n, dtype=bool)
    n_anom_mad = int(anom_mask_mad.sum())
    frac_anom_mad = n_anom_mad / n

    # --- Structural features ---
    max_deviation = float(np.max(np.abs(readings - BLOCK_BASE_MEAN)))
    longest_run = _longest_run(anom_mask_fixed)
    n_clusters = _count_clusters(anom_mask_fixed)

    return {
        "blk_mean": mean_,
        "blk_std": std_,
        "blk_median": median_,
        "blk_iqr": iqr_,
        "blk_skew": skew_,
        "blk_kurt": kurt_,
        "blk_min": min_,
        "blk_max": max_,
        "blk_n_anom_fixed": n_anom_fixed,
        "blk_frac_anom_fixed": frac_anom_fixed,
        "blk_n_anom_mad": n_anom_mad,
        "blk_frac_anom_mad": frac_anom_mad,
        "blk_max_deviation": max_deviation,
        "blk_longest_run": longest_run,
        "blk_n_clusters": n_clusters,
        "blk_mad": mad,
        "blk_range": max_ - min_,
        "blk_q25": q25,
        "blk_q75": q75,
    }


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest consecutive True run in a boolean array."""
    if not mask.any():
        return 0
    max_run = 0
    cur_run = 0
    for v in mask:
        if v:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


def _count_clusters(mask: np.ndarray) -> int:
    """Count connected components (clusters) of True values in a boolean array."""
    if not mask.any():
        return 0
    clusters = 0
    in_cluster = False
    for v in mask:
        if v and not in_cluster:
            clusters += 1
            in_cluster = True
        elif not v:
            in_cluster = False
    return clusters


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def compute_block_features(block_readings_array: np.ndarray) -> pd.DataFrame:
    """
    Given a 2D numpy array of shape (n_dies, n_blocks),
    return a DataFrame with all block feature columns.
    """
    rows = []
    for i in range(len(block_readings_array)):
        rows.append(_extract_one(block_readings_array[i]))
    return pd.DataFrame(rows)


def compute_block_features_from_series(series: pd.Series) -> pd.DataFrame:
    """
    Convenience wrapper: parse a Series of space-separated block strings
    and return block features DataFrame strictly aligned with the series index.
    """
    rows = []
    for val in series:
        if isinstance(val, str):
            readings = np.fromstring(val, dtype=np.float32, sep=" ")
        else:
            readings = np.array([], dtype=np.float32)
        rows.append(_extract_one(readings))
    out_df = pd.DataFrame(rows, index=series.index)
    assert_aligned(pd.DataFrame(index=series.index), out_df, context="block_features")
    return out_df


def get_block_feature_cols() -> List[str]:
    """Return the list of block feature column names (must match _extract_one keys)."""
    return [
        "blk_mean", "blk_std", "blk_median", "blk_iqr", "blk_skew", "blk_kurt",
        "blk_min", "blk_max",
        "blk_n_anom_fixed", "blk_frac_anom_fixed",
        "blk_n_anom_mad", "blk_frac_anom_mad",
        "blk_max_deviation", "blk_longest_run", "blk_n_clusters",
        "blk_mad", "blk_range", "blk_q25", "blk_q75",
    ]
