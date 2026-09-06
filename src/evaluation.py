"""
src/evaluation.py
==================
Reproduce the exact evaluation logic from README.md.
Evaluation is RESTRICTED to old_label=0 dies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve,
    auc, f1_score, precision_score, recall_score,
)
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Core evaluation (restricted to old_label=0 dies)
# ---------------------------------------------------------------------------

def evaluate(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate predictions restricted to old_label=0 rows (eligible dies).
    
    Args:
        df: DataFrame with 'old_label' and 'label' columns
        y_pred: predicted labels (0/1) for ALL rows in df
        y_prob: predicted probabilities for ALL rows (for PR-AUC)
        threshold: decision threshold used (for reporting)
        verbose: print confusion matrix

    Returns:
        dict of metric name -> value
    """
    eligible = df["old_label"] == 0
    y_true_elig = df.loc[eligible, "label"].values
    y_pred_elig = np.array(y_pred)[eligible.values]

    # Confusion matrix: rows = actual, cols = predicted
    # label=1 is "Fail", label=0 is "Pass"
    tn, fp, fn, tp = confusion_matrix(y_true_elig, y_pred_elig, labels=[0, 1]).ravel()

    n_total = len(y_true_elig)
    overall_acc = (tp + tn) / n_total if n_total > 0 else 0.0

    # Fail metrics (class 1)
    fail_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Fail Accuracy
    fail_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0     # Fail Precision
    fail_f1 = (2 * fail_prec * fail_recall / (fail_prec + fail_recall)
               if (fail_prec + fail_recall) > 0 else 0.0)

    # Pass metrics (class 0)
    pass_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # Pass Accuracy
    pass_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0     # Pass Precision
    pass_f1 = (2 * pass_prec * pass_recall / (pass_prec + pass_recall)
               if (pass_prec + pass_recall) > 0 else 0.0)

    # PR-AUC (on eligible dies only, if probs given)
    pr_auc = 0.0
    if y_prob is not None:
        y_prob_elig = np.array(y_prob)[eligible.values]
        prec_curve, rec_curve, _ = precision_recall_curve(y_true_elig, y_prob_elig)
        pr_auc = auc(rec_curve, prec_curve)

    metrics = {
        "threshold": threshold,
        "overall_accuracy": overall_acc,
        "fail_recall": fail_recall,        # = Fail Accuracy in README
        "fail_precision": fail_prec,
        "fail_f1": fail_f1,
        "pass_recall": pass_recall,        # = Pass Accuracy in README
        "pass_precision": pass_prec,
        "pass_f1": pass_f1,
        "pr_auc": pr_auc,
        "n_eligible": n_total,
        "n_actual_fail": int(tp + fn),
        "n_actual_pass": int(tn + fp),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    if verbose:
        _print_confusion(tp, tn, fp, fn, metrics)

    return metrics


def _print_confusion(tp, tn, fp, fn, metrics):
    """Print confusion matrix in README.md format."""
    print(f"\n{'':30s}  Pred Fail  Pred Pass       Metric               Value")
    print(f"  Actual Fail  {tp:12d} {fn:10d}       Fail Accuracy     {metrics['fail_recall']:.6f}")
    print(f"  Actual Pass  {fp:12d} {tn:10d}       Pass Accuracy     {metrics['pass_recall']:.6f}")
    print(f"\n  Overall Accuracy : {metrics['overall_accuracy']:.6f}")
    print(f"  Fail Precision   : {metrics['fail_precision']:.6f}")
    print(f"  Fail F1          : {metrics['fail_f1']:.6f}")
    print(f"  PR-AUC           : {metrics['pr_auc']:.6f}")
    print(f"  Threshold used   : {metrics['threshold']:.4f}")


# ---------------------------------------------------------------------------
# Threshold selection on validation set
# ---------------------------------------------------------------------------

def select_threshold(
    df_val: pd.DataFrame,
    y_prob_val: np.ndarray,
    n_thresholds: int = 200,
) -> tuple[float, float]:
    """
    Sweep thresholds on validation set (old_label=0 only) and pick the one
    that maximizes Fail F1.

    Returns (best_threshold, best_fail_f1).
    """
    eligible = df_val["old_label"] == 0
    y_true = df_val.loc[eligible, "label"].values
    y_prob = np.array(y_prob_val)[eligible.values]

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_thresh = 0.5
    best_f1 = 0.0

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"[threshold] Best threshold: {best_thresh:.4f} -> Fail F1 = {best_f1:.4f}")
    return float(best_thresh), float(best_f1)


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def comparison_table(metrics_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build comparison table from a dict of {model_name: metrics_dict}.
    Shows absolute values + improvement vs. baseline (first entry).
    """
    rows = []
    baseline_key = list(metrics_dict.keys())[0]
    baseline = metrics_dict[baseline_key]

    for name, m in metrics_dict.items():
        row = {"Model": name}
        for col in ["pr_auc", "fail_f1", "fail_recall", "fail_precision", "overall_accuracy"]:
            row[col] = m.get(col, np.nan)
            if name != baseline_key:
                delta = m.get(col, np.nan) - baseline.get(col, np.nan)
                pct = (delta / baseline.get(col, 1e-9)) * 100
                row[f"{col}_delta"] = delta
                row[f"{col}_pct"] = pct
        rows.append(row)

    return pd.DataFrame(rows)


def ablation_table(ablation_metrics: Dict[str, Dict]) -> pd.DataFrame:
    """Build ablation table from dict of {feature_set_name: metrics_dict}."""
    rows = []
    for name, m in ablation_metrics.items():
        rows.append({
            "Feature Set": name,
            "PR-AUC": m.get("pr_auc", np.nan),
            "Fail F1": m.get("fail_f1", np.nan),
            "Fail Recall": m.get("fail_recall", np.nan),
            "Fail Precision": m.get("fail_precision", np.nan),
            "Accuracy": m.get("overall_accuracy", np.nan),
        })
    return pd.DataFrame(rows)
