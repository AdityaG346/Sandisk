"""
src/data_loader.py
==================
Load train/test CSVs and provide wafer-level train/validation splits.
Includes guardrails for row alignment and wafer split disjointness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Column groups (defined once, used everywhere)
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return the 500 die-level parametric feature column names."""
    return [c for c in df.columns if c.startswith("feature_")]


def get_id_cols() -> List[str]:
    return ["wafer_id", "die_row", "die_col"]


# ---------------------------------------------------------------------------
# Guardrails: Index alignment and Split integrity assertions
# ---------------------------------------------------------------------------

def assert_aligned(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    key_cols: Optional[List[str]] = None,
    context: str = "",
) -> None:
    """
    Assert that df_a and df_b have identical length and that either key_cols
    (if present in both) or their indices match row-for-row exactly.
    Raises AssertionError with informative error messages.
    """
    prefix = f"[{context}] " if context else ""
    if len(df_a) != len(df_b):
        raise AssertionError(
            f"{prefix}Length mismatch: len(df_a)={len(df_a):,} vs len(df_b)={len(df_b):,}"
        )

    checked_keys = False
    if key_cols is not None:
        common_keys = [c for c in key_cols if c in df_a.columns and c in df_b.columns]
        if common_keys:
            checked_keys = True
            for col in common_keys:
                diffs = (df_a[col].values != df_b[col].values)
                if diffs.any():
                    n_diff = int(diffs.sum())
                    first_bad = int(np.where(diffs)[0][0])
                    raise AssertionError(
                        f"{prefix}Key column '{col}' has {n_diff} mismatched rows! "
                        f"First mismatch at index {first_bad}: {df_a[col].values[first_bad]} vs {df_b[col].values[first_bad]}"
                    )

    if not checked_keys:
        if not (df_a.index == df_b.index).all():
            n_diff = int((df_a.index != df_b.index).sum())
            raise AssertionError(f"{prefix}Row index mismatch across {n_diff} rows!")


def assert_split_disjoint(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
) -> None:
    """Ensure zero wafer_id leakage across splits."""
    tr_wafers = set(train_df["wafer_id"].unique())
    val_wafers = set(val_df["wafer_id"].unique())
    overlap_tr_val = tr_wafers & val_wafers
    if overlap_tr_val:
        raise AssertionError(f"[SplitLeakage] Wafers overlap between train and val: {overlap_tr_val}")

    if test_df is not None:
        te_wafers = set(test_df["wafer_id"].unique())
        overlap_tr_te = tr_wafers & te_wafers
        overlap_val_te = val_wafers & te_wafers
        if overlap_tr_te:
            raise AssertionError(f"[SplitLeakage] Wafers overlap between train and test: {overlap_tr_te}")
        if overlap_val_te:
            raise AssertionError(f"[SplitLeakage] Wafers overlap between val and test: {overlap_val_te}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_train(input_dir: str = "input") -> pd.DataFrame:
    path = Path(input_dir) / "train.csv"
    if not path.exists():
        raise FileNotFoundError(f"train.csv not found at {path}. Run generate_data.py first.")
    df = pd.read_csv(path, low_memory=False)
    print(f"[data_loader] Loaded train: {len(df):,} rows, {df['wafer_id'].nunique()} wafers")
    return df


def load_test(input_dir: str = "input") -> pd.DataFrame:
    path = Path(input_dir) / "test.csv"
    if not path.exists():
        raise FileNotFoundError(f"test.csv not found at {path}. Run generate_data.py first.")
    df = pd.read_csv(path, low_memory=False)
    print(f"[data_loader] Loaded test:  {len(df):,} rows, {df['wafer_id'].nunique()} wafers")
    return df


# ---------------------------------------------------------------------------
# Block readings parsing
# ---------------------------------------------------------------------------

def parse_block_readings(series: pd.Series) -> np.ndarray:
    """
    Convert a Series of space-separated string block readings
    to a 2D numpy array of shape (n_dies, n_blocks).
    Memory-efficient: processes row by row.
    """
    result = []
    for val in series:
        if isinstance(val, str):
            result.append(np.fromstring(val, dtype=np.float32, sep=" "))
        else:
            result.append(np.array([], dtype=np.float32))
    return np.vstack(result)


# ---------------------------------------------------------------------------
# Wafer-level train/validation split
# ---------------------------------------------------------------------------

def wafer_level_split(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by wafer_id (never by row).
    Stratifies by whether the wafer has ANY new failures (label=1, old_label=0).
    Returns (train_df, val_df).
    """
    rng = np.random.default_rng(seed)
    wafer_ids = df["wafer_id"].unique()

    # Compute per-wafer "has new fails" flag for stratification
    wafer_stats = (
        df[df["old_label"] == 0]
        .groupby("wafer_id")["label"]
        .max()
        .reindex(wafer_ids, fill_value=0)
    )

    has_new_fail = wafer_stats[wafer_ids].values
    active_wafers = wafer_ids[has_new_fail == 1]
    clean_wafers = wafer_ids[has_new_fail == 0]

    # Sample val set: val_fraction from each stratum
    n_val_active = max(1, int(len(active_wafers) * val_fraction))
    n_val_clean = max(1, int(len(clean_wafers) * val_fraction))

    val_active = rng.choice(active_wafers, size=n_val_active, replace=False)
    val_clean = rng.choice(clean_wafers, size=n_val_clean, replace=False)
    val_wafers = set(val_active) | set(val_clean)

    val_df = df[df["wafer_id"].isin(val_wafers)].copy().reset_index(drop=True)
    train_df = df[~df["wafer_id"].isin(val_wafers)].copy().reset_index(drop=True)

    # Disjointness assertion
    assert_split_disjoint(train_df, val_df)

    n_val_w = val_df["wafer_id"].nunique()
    n_tr_w = train_df["wafer_id"].nunique()
    print(f"[data_loader] Split -> train: {len(train_df):,} rows ({n_tr_w} wafers) | "
          f"val: {len(val_df):,} rows ({n_val_w} wafers)")
    return train_df, val_df
