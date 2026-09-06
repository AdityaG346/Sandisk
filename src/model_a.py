"""
src/model_a.py
==============
Model A: die-level parametric features + spatial features + die anomaly score.
Uses LightGBM with class weights (scale_pos_weight = 12.0 default, tuned).
Includes alignment assertions and training sanity assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from src.data_loader import assert_aligned
from src.evaluation import evaluate, select_threshold


SEED = 42


def get_feature_set_a(
    df: pd.DataFrame,
    sp_df: pd.DataFrame,
    die_anomaly_scores: np.ndarray,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Assemble Model A feature matrix with strict alignment assertions.
    Returns (X_df, feature_cols).
    """
    assert_aligned(df, sp_df, key_cols=["wafer_id", "die_row", "die_col"], context="model_a_spatial_alignment")
    if len(df) != len(die_anomaly_scores):
        raise AssertionError(
            f"[model_a] die_anomaly_scores length ({len(die_anomaly_scores)}) != df length ({len(df)})"
        )

    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    sp_cols = [c for c in sp_df.columns if not c.startswith("wafer_") and c not in ("die_row", "die_col")]

    X = df[feat_cols].copy().reset_index(drop=True)
    X_sp = sp_df[sp_cols].copy().reset_index(drop=True)
    X = pd.concat([X, X_sp], axis=1)
    X["die_anomaly_score"] = np.asarray(die_anomaly_scores, dtype=np.float32)

    all_cols = feat_cols + sp_cols + ["die_anomaly_score"]
    return X[all_cols], all_cols


def compute_class_weight(y: np.ndarray) -> float:
    """Compute baseline scale_pos_weight = n_negative / n_positive."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def train_model_a(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    sp_train: pd.DataFrame,
    sp_val: pd.DataFrame,
    die_anom_train: np.ndarray,
    die_anom_val: np.ndarray,
    output_dir: str = "outputs",
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    scale_pos_weight: float = 12.0,
    seed: int = SEED,
) -> Tuple[lgb.LGBMClassifier, float, List[str], Dict]:
    """
    Train Model A and select threshold on validation set.
    Includes training sanity floor assertion (best_iteration_ > 5).
    Returns (model, threshold, feature_cols, val_metrics).
    """
    X_train, feat_cols = get_feature_set_a(train_df, sp_train, die_anom_train)
    X_val, _ = get_feature_set_a(val_df, sp_val, die_anom_val)

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    print(f"[model_a] scale_pos_weight = {scale_pos_weight:.2f}  |  features: {len(feat_cols)}")

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train.values, y_train,
        eval_set=[(X_val.values, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    best_iter = getattr(model, "best_iteration_", None) or n_estimators
    if best_iter <= 5:
        raise AssertionError(
            f"[TrainingSanityError] Model A stopped suspiciously early at iteration {best_iter} <= 5! "
            "This indicates loss divergence, severe overfitting, or misaligned features."
        )

    # Threshold selection on validation set
    y_prob_val = model.predict_proba(X_val.values)[:, 1]
    threshold, val_f1 = select_threshold(val_df, y_prob_val)

    # Evaluate with chosen threshold
    y_pred_val = (y_prob_val >= threshold).astype(int)
    val_metrics = evaluate(val_df, y_pred_val, y_prob_val, threshold=threshold, verbose=False)
    val_metrics["val_fail_f1"] = val_f1
    val_metrics["best_iteration"] = best_iter

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model_a.pkl")
    joblib.dump({"threshold": threshold, "feat_cols": feat_cols, "spw": scale_pos_weight, "val_fail_f1": val_f1},
                out / "model_a_meta.pkl")
    print(f"[model_a] Saved to {out}/model_a.pkl (best_iter={best_iter}, thresh={threshold:.4f})")

    return model, threshold, feat_cols, val_metrics


def predict_model_a(
    model: lgb.LGBMClassifier,
    df: pd.DataFrame,
    sp_df: pd.DataFrame,
    die_anom_scores: np.ndarray,
    feat_cols: List[str],
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict with Model A. Returns (predicted_labels, probabilities).
    old_label=1 dies are forced to predict 1.
    """
    X, _ = get_feature_set_a(df, sp_df, die_anom_scores)
    X = X[feat_cols]
    y_prob = model.predict_proba(X.values)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Force old fails to be predicted as fail
    y_pred[df["old_label"].values == 1] = 1

    return y_pred, y_prob
