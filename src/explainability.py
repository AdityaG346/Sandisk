"""
src/explainability.py
======================
SHAP global summaries, native feature importance, and per-die explanations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import lightgbm as lgb


# ---------------------------------------------------------------------------
# Global SHAP summary
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    max_samples: int = 5000,
) -> np.ndarray:
    """
    Compute SHAP values (for class=1) using TreeExplainer.
    Subsample if X is large.
    """
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X.iloc[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)

    # LightGBM binary returns list [neg_class, pos_class]; take pos_class
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    return shap_vals, X_sample


def plot_shap_summary(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    feat_cols: List[str],
    model_name: str,
    output_dir: str = "outputs/shap",
    max_samples: int = 3000,
    max_display: int = 30,
) -> None:
    """Plot and save SHAP beeswarm summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[shap] Computing SHAP values for {model_name}...")
    shap_vals, X_sample = compute_shap_values(model, X[feat_cols], max_samples)

    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(
        shap_vals, X_sample,
        plot_type="dot",
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Summary -- {model_name}", fontsize=14)
    plt.tight_layout()
    fig.savefig(out / f"shap_summary_{model_name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[shap] Saved: {out}/shap_summary_{model_name.lower().replace(' ', '_')}.png")


def plot_feature_importance(
    model: lgb.LGBMClassifier,
    feat_cols: List[str],
    model_name: str,
    output_dir: str = "outputs/shap",
    top_n: int = 40,
) -> None:
    """Plot native LightGBM feature importance (gain)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    importances = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feat_cols, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="#4C72B0")
    ax.set_xlabel("Feature Importance (split gain)")
    ax.set_title(f"Top {top_n} Feature Importances -- {model_name}")
    plt.tight_layout()
    fig.savefig(out / f"feature_importance_{model_name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[shap] Saved: {out}/feature_importance_{model_name.lower().replace(' ', '_')}.png")


# ---------------------------------------------------------------------------
# Per-die explanations
# ---------------------------------------------------------------------------

def explain_predicted_fail_dies(
    model: lgb.LGBMClassifier,
    df: pd.DataFrame,
    X: pd.DataFrame,
    feat_cols: List[str],
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_examples: int = 5,
    sp_cols: Optional[List[str]] = None,
    blk_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    For n predicted-fail dies (from old_label=0 pool), produce human-readable
    explanations using SHAP values.

    Returns list of explanation strings.
    """
    die_feat_cols = [c for c in feat_cols if c.startswith("feature_")]
    sp_cols_used = sp_cols or [c for c in feat_cols if c.startswith("sp_")]
    blk_cols_used = blk_cols or [c for c in feat_cols if c.startswith("blk_")]

    # Filter to eligible predicted-fail dies from old_label=0
    eligible_mask = (df["old_label"].values == 0) & (np.array(y_pred) == 1)
    eligible_indices = np.where(eligible_mask)[0]

    if len(eligible_indices) == 0:
        return ["No predicted-fail eligible dies found."]

    # Sample up to n_examples
    chosen = eligible_indices[:n_examples]

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    X_chosen = X[feat_cols].iloc[chosen]
    shap_vals = explainer.shap_values(X_chosen)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # positive class

    explanations = []
    for i, (global_idx, local_i) in enumerate(zip(chosen, range(len(chosen)))):
        row = df.iloc[global_idx]
        prob = y_prob[global_idx]
        row_coord = int(row["die_row"])
        col_coord = int(row["die_col"])
        wafer = row["wafer_id"]

        # Get top contributing features
        sv = shap_vals[local_i]
        feat_shap = sorted(zip(feat_cols, sv), key=lambda x: abs(x[1]), reverse=True)

        # Group contributions by type
        die_contrib = sum(abs(v) for f, v in feat_shap if f.startswith("feature_"))
        sp_contrib = sum(abs(v) for f, v in feat_shap if f.startswith("sp_"))
        blk_contrib = sum(abs(v) for f, v in feat_shap if f.startswith("blk_"))
        anom_contrib = sum(abs(v) for f, v in feat_shap if "anomaly" in f)

        top3 = feat_shap[:3]
        reasons = []
        for rank, (feat_name, sv_val) in enumerate(top3, 1):
            direction = "↑" if sv_val > 0 else "↓"
            feat_val = X[feat_cols].iloc[global_idx][feat_name]
            reasons.append(f"{rank}) {feat_name} = {feat_val:.3g} (SHAP {direction}{abs(sv_val):.4f})")

        expl = (
            f"Die ({row_coord},{col_coord}) on wafer {wafer}: "
            f"failure probability {prob*100:.1f}%\n"
            f"   SHAP group contributions -> "
            f"die-params: {die_contrib:.4f} | spatial: {sp_contrib:.4f} | "
            f"block: {blk_contrib:.4f} | anomaly: {anom_contrib:.4f}\n"
            f"   Main reasons:\n" +
            "\n".join(f"   {r}" for r in reasons)
        )
        explanations.append(expl)

    return explanations
