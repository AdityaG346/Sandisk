"""
src/spatial_features.py
========================
Compute leakage-safe spatial/neighborhood features from old_label + coordinates.

HARD RULE: All features here use ONLY old_label, die_row, die_col.
           They must NEVER touch the 'label' column.
           A unit test (test_no_leakage) validates this property.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.spatial import KDTree
from typing import Optional

from src.data_loader import assert_aligned


# ---------------------------------------------------------------------------
# Internal helpers -- operate on 2D wafer-map arrays
# ---------------------------------------------------------------------------

def _make_wafer_grid(
    die_rows: np.ndarray,
    die_cols: np.ndarray,
    old_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Build dense 2D grid arrays for a single wafer.
    Returns (old_fail_grid, valid_grid, n_rows, n_cols).
    """
    n_rows = int(die_rows.max()) + 1
    n_cols = int(die_cols.max()) + 1

    old_fail_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    valid_grid = np.zeros((n_rows, n_cols), dtype=np.float32)

    old_fail_grid[die_rows, die_cols] = old_labels.astype(np.float32)
    valid_grid[die_rows, die_cols] = 1.0

    return old_fail_grid, valid_grid, n_rows, n_cols


def _local_fail_density(
    old_fail_grid: np.ndarray,
    valid_grid: np.ndarray,
    window: int,
) -> np.ndarray:
    """Local old-fail density in a window×window neighborhood."""
    fail_sum = uniform_filter(old_fail_grid, size=window, mode="constant", cval=0.0)
    valid_sum = uniform_filter(valid_grid, size=window, mode="constant", cval=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(valid_sum > 0, fail_sum / valid_sum, 0.0)
    return density


def _distance_to_nearest_fail(
    die_rows: np.ndarray,
    die_cols: np.ndarray,
    old_labels: np.ndarray,
) -> np.ndarray:
    """
    For each die, Euclidean distance to nearest old_label=1 die.
    Dies with no old fails on the wafer get distance = max possible.
    """
    fail_mask = old_labels == 1
    if fail_mask.sum() == 0:
        # No old fails -> max distance for all
        diag = np.sqrt(die_rows.max() ** 2 + die_cols.max() ** 2) + 1.0
        return np.full(len(die_rows), diag, dtype=np.float32)

    fail_coords = np.column_stack([die_rows[fail_mask], die_cols[fail_mask]])
    all_coords = np.column_stack([die_rows, die_cols])
    tree = KDTree(fail_coords)
    dists, _ = tree.query(all_coords, k=1, workers=1)
    return dists.astype(np.float32)


def _zone_fail_yield(
    die_rows: np.ndarray,
    die_cols: np.ndarray,
    old_labels: np.ndarray,
    zone_rows: int = 4,
    zone_cols: int = 4,
) -> np.ndarray:
    """
    Zone-level old_label yield (fraction of passes) broadcast back to each die.
    """
    n_rows = int(die_rows.max()) + 1
    n_cols = int(die_cols.max()) + 1

    row_edges = np.linspace(0, n_rows, zone_rows + 1, dtype=int)
    col_edges = np.linspace(0, n_cols, zone_cols + 1, dtype=int)

    zone_yield = np.ones(len(die_rows), dtype=np.float32)

    for zr in range(zone_rows):
        for zc in range(zone_cols):
            r0, r1 = row_edges[zr], row_edges[zr + 1]
            c0, c1 = col_edges[zc], col_edges[zc + 1]
            mask = (die_rows >= r0) & (die_rows < r1) & (die_cols >= c0) & (die_cols < c1)
            if mask.sum() > 0:
                y = 1.0 - old_labels[mask].mean()
                zone_yield[mask] = y

    return zone_yield


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all spatial features for every die in df.
    Uses ONLY: wafer_id, die_row, die_col, old_label.
    Never reads 'label'.

    Returns a new DataFrame (strictly aligned to df) with spatial feature columns.
    """
    out_records = []

    for wafer_id, wdf in df.groupby("wafer_id", sort=False):
        die_rows = wdf["die_row"].values.astype(int)
        die_cols = wdf["die_col"].values.astype(int)
        old_labels = wdf["old_label"].values.astype(np.float32)

        old_fail_grid, valid_grid, n_rows, n_cols = _make_wafer_grid(
            die_rows, die_cols, old_labels
        )

        # ---- Radial distance from wafer center (normalized 0-1) ----
        cy, cx = n_rows / 2.0, n_cols / 2.0
        r_dist = np.sqrt((die_rows - cy) ** 2 + (die_cols - cx) ** 2)
        max_r = np.sqrt(cy ** 2 + cx ** 2)
        radial_norm = (r_dist / max_r).astype(np.float32) if max_r > 0 else np.zeros_like(r_dist, dtype=np.float32)

        # ---- Normalized row / col ----
        row_norm = (die_rows / max(n_rows - 1, 1)).astype(np.float32)
        col_norm = (die_cols / max(n_cols - 1, 1)).astype(np.float32)

        # ---- Edge proximity (1 = on edge row/col, 0 = center) ----
        edge_row = np.minimum(die_rows, n_rows - 1 - die_rows).astype(np.float32)
        edge_col = np.minimum(die_cols, n_cols - 1 - die_cols).astype(np.float32)
        edge_prox = 1.0 - (np.minimum(edge_row, edge_col) / max(min(n_rows, n_cols) / 2.0, 1.0))
        edge_prox = np.clip(edge_prox, 0.0, 1.0).astype(np.float32)

        # ---- Local old-fail density at 3×3, 5×5, 7×7 windows ----
        density_3 = _local_fail_density(old_fail_grid, valid_grid, 3)
        density_5 = _local_fail_density(old_fail_grid, valid_grid, 5)
        density_7 = _local_fail_density(old_fail_grid, valid_grid, 7)

        d3 = density_3[die_rows, die_cols].astype(np.float32)
        d5 = density_5[die_rows, die_cols].astype(np.float32)
        d7 = density_7[die_rows, die_cols].astype(np.float32)

        # ---- Zone-level old_label yield (4×4 grid) ----
        zone_yield = _zone_fail_yield(die_rows, die_cols, old_labels, 4, 4)

        # ---- Distance to nearest old-fail die ----
        dist_to_fail = _distance_to_nearest_fail(die_rows, die_cols, old_labels)
        # Normalize by wafer diagonal
        diag = np.sqrt(n_rows ** 2 + n_cols ** 2)
        dist_to_fail_norm = (dist_to_fail / diag).astype(np.float32)

        # ---- old_label itself (pre-test status) ----
        # Included as a feature -- known at prediction time
        old_label_feat = old_labels

        df_wafer = pd.DataFrame(
            {
                "wafer_id": wafer_id,
                "die_row": die_rows,
                "die_col": die_cols,
                "sp_row_norm": row_norm,
                "sp_col_norm": col_norm,
                "sp_radial": radial_norm,
                "sp_edge_prox": edge_prox,
                "sp_old_fail_density_3": d3,
                "sp_old_fail_density_5": d5,
                "sp_old_fail_density_7": d7,
                "sp_zone_yield": zone_yield,
                "sp_dist_to_fail": dist_to_fail_norm,
                "sp_old_label": old_label_feat,
            }
        )
        out_records.append(df_wafer)

    result = pd.concat(out_records, ignore_index=True)

    # Merge back on (wafer_id, die_row, die_col) to strictly align with original df index
    sp_cols = [c for c in result.columns if c not in ("wafer_id", "die_row", "die_col")]
    merged = df[["wafer_id", "die_row", "die_col"]].merge(
        result, on=["wafer_id", "die_row", "die_col"], how="left"
    )
    assert_aligned(df, merged, key_cols=["wafer_id", "die_row", "die_col"], context="spatial_features_merge")
    out_df = merged[sp_cols].copy()
    out_df.index = df.index
    assert_aligned(df, out_df, context="spatial_features_final")
    return out_df


# ---------------------------------------------------------------------------
# Leakage unit test
# ---------------------------------------------------------------------------

def test_no_leakage(df: pd.DataFrame) -> bool:
    """
    Verify that spatial features are identical whether 'label' is present or not.
    Raises AssertionError if any leakage detected; returns True otherwise.
    """
    # Compute with label present
    feats_with_label = compute_spatial_features(df)

    # Compute without label column
    df_no_label = df.drop(columns=["label"], errors="ignore")
    feats_without_label = compute_spatial_features(df_no_label)

    # Compare all spatial feature columns
    sp_cols = [c for c in feats_with_label.columns]
    for col in sp_cols:
        if not np.allclose(
            feats_with_label[col].values,
            feats_without_label[col].values,
            equal_nan=True,
        ):
            raise AssertionError(
                f"LEAKAGE DETECTED: column '{col}' differs when 'label' is present vs absent!"
            )

    print("[leakage_test] PASSED: all spatial features are identical with/without label column.")
    return True
