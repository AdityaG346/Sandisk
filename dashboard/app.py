"""
dashboard/app.py
================
Interactive Streamlit Dashboard for Semiconductor Die Yield Prediction.
Presents precomputed model results, wafer maps, continuous risk fields,
per-die SHAP attributions, sub-die block readings, and Model A vs B comparisons.

Usage:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
os.environ["PYTHONUTF8"] = "1"
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.triage import topk_stats, capture_curve

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import shap
import streamlit as st


# -----------------------------------------------------------------------------
# Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Semiconductor Die Yield Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium dark theme styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .disclaimer-box {
        background-color: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #cbd5e1;
        margin: 10px 0;
    }
    .badge-pass {
        background-color: #166534;
        color: #bbf7d0;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-fail {
        background-color: #991b1b;
        color: #fecaca;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
INPUT_DIR = REPO_ROOT / "input"


# -----------------------------------------------------------------------------
# Cached Artifact Loaders (Zero-Retraining, Memory-Optimized)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading test dataset metadata...")
def load_test_metadata() -> pd.DataFrame:
    """Load core test metadata (wafer_id, die_row, die_col, old_label, label)."""
    meta_pq = CACHE_DIR / "test_meta.parquet"
    if meta_pq.exists():
        return pd.read_parquet(meta_pq)
    
    test_csv = INPUT_DIR / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"test dataset not found at {test_csv} or {meta_pq}!")
    cols = ["wafer_id", "die_row", "die_col", "old_label"]
    df_sample = pd.read_csv(test_csv, nrows=1)
    if "label" in df_sample.columns:
        cols.append("label")
    df = pd.read_csv(test_csv, usecols=cols)
    return df


@st.cache_data(show_spinner="Loading precomputed predictions...")
def load_predictions() -> pd.DataFrame:
    """Load final predictions CSV."""
    pred_path = OUTPUT_DIR / "predictions.csv"
    if pred_path.exists():
        return pd.read_csv(pred_path)
    return pd.DataFrame()


@st.cache_resource(show_spinner="Loading production models...")
def load_models() -> Tuple[object, dict, object, dict]:
    """Load Model A and Model B with metadata."""
    model_a = joblib.load(OUTPUT_DIR / "model_a.pkl")
    meta_a = joblib.load(OUTPUT_DIR / "model_a_meta.pkl")
    model_b = joblib.load(OUTPUT_DIR / "model_b.pkl")
    meta_b = joblib.load(OUTPUT_DIR / "model_b_meta.pkl")
    return model_a, meta_a, model_b, meta_b


@st.cache_data(show_spinner="Loading precomputed probability vectors...")
def load_probabilities() -> Tuple[np.ndarray, np.ndarray]:
    """Load pre-scored probabilities for Model A and Model B."""
    prob_a_path = CACHE_DIR / "test_probs_a.npy"
    prob_b_path = CACHE_DIR / "test_probs_b.npy"
    if prob_a_path.exists() and prob_b_path.exists():
        return np.load(prob_a_path), np.load(prob_b_path)
    
    # Fallback compute if cache files are missing
    test_df = load_test_metadata()
    model_a, meta_a, model_b, meta_b = load_models()
    X_a = pd.read_parquet(CACHE_DIR / "X_test_a.parquet")
    X_b = pd.read_parquet(CACHE_DIR / "X_test_b.parquet")
    prob_a = model_a.predict_proba(X_a[meta_a["feat_cols"]].values)[:, 1]
    prob_b = model_b.predict_proba(X_b[meta_b["feat_cols"]].values)[:, 1]
    return prob_a, prob_b


@st.cache_data(show_spinner="Loading calibrated probability vectors...")
def load_calibrated_probabilities() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load Platt calibrated probabilities if available."""
    cal_a_path = CACHE_DIR / "test_probs_a_cal.npy"
    cal_b_path = CACHE_DIR / "test_probs_b_cal.npy"
    prob_a_cal = np.load(cal_a_path) if cal_a_path.exists() else None
    prob_b_cal = np.load(cal_b_path) if cal_b_path.exists() else None
    return prob_a_cal, prob_b_cal


@st.cache_data(show_spinner="Loading benchmark evaluation tables...")
def load_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load comparison, multiseed ablation, and failure signatures tables."""
    comp_df = pd.read_csv(OUTPUT_DIR / "comparison_table.csv")
    abl_df = pd.read_csv(OUTPUT_DIR / "ablation_table_multiseed.csv")
    sig_path = OUTPUT_DIR / "failure_signatures.csv"
    sig_df = pd.read_csv(sig_path) if sig_path.exists() else pd.DataFrame()
    return comp_df, abl_df, sig_df


@st.cache_data(show_spinner="Loading sub-die block readings...")
def load_wafer_block_readings(wafer_id: str) -> pd.DataFrame:
    """Load block readings for a specific wafer."""
    w_blk_path = CACHE_DIR / "wafer_blocks" / f"{wafer_id}.parquet"
    if w_blk_path.exists():
        return pd.read_parquet(w_blk_path)
    test_csv = INPUT_DIR / "test.csv"
    if test_csv.exists():
        df = pd.read_csv(test_csv, usecols=["wafer_id", "die_row", "die_col", "block_readings"])
        return df[df["wafer_id"] == wafer_id]
    return pd.DataFrame()


@st.cache_data(show_spinner="Computing SHAP explanations for selected wafer...")
def compute_wafer_shap(wafer_id: str, model_type: str = "Model B") -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Compute and cache SHAP values for all dies on the selected wafer."""
    wafer_feat_path = CACHE_DIR / "wafer_features" / f"{wafer_id}.parquet"
    if not wafer_feat_path.exists():
        raise FileNotFoundError(f"Wafer features not found at {wafer_feat_path}!")

    X_wafer_full = pd.read_parquet(wafer_feat_path)

    model_a, meta_a, model_b, meta_b = load_models()
    if model_type == "Model A":
        model = model_a
        cols = meta_a["feat_cols"]
    else:
        model = model_b
        cols = meta_b["feat_cols"]

    X_wafer = X_wafer_full[cols]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_wafer.values)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    return shap_vals, cols, X_wafer


@st.cache_data(show_spinner="Loading population medians...")
def load_pass_medians() -> Tuple[pd.Series, pd.Series]:
    """Load feature medians of passing dies for counterfactual evaluation."""
    med_b = pd.read_parquet(CACHE_DIR / "pass_medians_b.parquet")["median"]
    med_a = pd.read_parquet(CACHE_DIR / "pass_medians_a.parquet")["median"]
    return med_a, med_b


# -----------------------------------------------------------------------------
# Precomputed Representative Benchmark Explanations (Part 4)
# -----------------------------------------------------------------------------
PRECOMPUTED_REPRESENTATIVES = {
    ("W_F_0014", 40, 18): {
        "title": "Die 1: Edge-Region Die (Wafer W_F_0014, Row 40, Col 18)",
        "probability": 0.905,
        "confidence": "Moderate Confidence (ensemble σ = 0.101)",
        "cluster": "Cluster 2 (Defect Neighborhood Proximity & Spatial Clustering)",
        "top_features": [
            ("sp_dist_to_fail", -1.0208),
            ("blk_mean", +0.7047),
            ("blk_q75", +0.3758),
        ],
        "trajectory": "Original (90.5%) → Normalizing blk_mean (81.8%) → +blk_q75 (76.2%)",
        "reduction": "14.3% risk reduction to 76.2%",
    },
    ("W_F_0019", 22, 22): {
        "title": "Die 2: Wafer-Interior Die (Wafer W_F_0019, Row 22, Col 22)",
        "probability": 0.740,
        "confidence": "Moderate Confidence (ensemble σ = 0.095)",
        "cluster": "Cluster 2 (Defect Neighborhood Proximity & Spatial Clustering)",
        "top_features": [
            ("sp_dist_to_fail", -0.9181),
            ("blk_mean", +0.2001),
            ("blk_std", -0.1873),
        ],
        "trajectory": "Original (74.0%) → Normalizing blk_mean (63.0%) → +blk_std (69.7%)",
        "reduction": "4.3% risk reduction to 69.7%",
    },
    ("W_F_0009", 9, 39): {
        "title": "Die 3: High Anomaly-Score Die (Wafer W_F_0009, Row 9, Col 39)",
        "probability": 0.883,
        "confidence": "High Confidence (ensemble σ = 0.029)",
        "cluster": "Cluster 2 (Defect Neighborhood Proximity & Spatial Clustering)",
        "top_features": [
            ("sp_dist_to_fail", -0.9906),
            ("blk_mean", +0.9579),
            ("feature_291", -0.2176),
        ],
        "trajectory": "Original (88.3%) → Normalizing blk_mean (72.3%) → +feature_291 (75.0%)",
        "reduction": "13.3% risk reduction to 75.0%",
    },
    ("W_F_0010", 48, 6): {
        "title": "Die 4: Spatial Defect-Cluster Die (Wafer W_F_0010, Row 48, Col 6)",
        "probability": 0.984,
        "confidence": "High Confidence (ensemble σ = 0.012)",
        "cluster": "Cluster -1 (Mixed / Boundary Pattern)",
        "top_features": [
            ("blk_mean", +0.7140),
            ("sp_dist_to_fail", -0.5571),
            ("feature_97", -0.1885),
        ],
        "trajectory": "Original (98.4%) → Normalizing blk_mean (97.1%) → +feature_97 (95.5%)",
        "reduction": "2.9% risk reduction to 95.5%",
    },
    ("W_F_0016", 21, 12): {
        "title": "Die 5: Marginal Threshold Case (Wafer W_F_0016, Row 21, Col 12)",
        "probability": 0.521,
        "confidence": "Moderate Confidence (ensemble σ = 0.079)",
        "cluster": "Cluster -1 (Mixed / Boundary Pattern)",
        "top_features": [
            ("blk_mean", +1.4180),
            ("sp_dist_to_fail", -1.0596),
            ("blk_q75", +0.2558),
        ],
        "trajectory": "Original (52.1%) → Normalizing blk_mean (18.3%, Crosses Boundary to Pass!) → +blk_q75 (13.5%)",
        "reduction": "38.6% risk reduction (flips to Pass at 18.3%)",
    },
}


# -----------------------------------------------------------------------------
# Visualization Rendering Functions
# -----------------------------------------------------------------------------
def render_wafer_4panel(
    df_wafer: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    wafer_id: str,
    model_name: str,
    has_ground_truth: bool = True,
) -> matplotlib.figure.Figure:
    """
    Render 4-panel wafer map matching Part 4 styling:
    1. Pre-test map (old_label)
    2. Ground-truth post-test map (label) if available
    3. Model predicted failure probability heatmap
    4. Continuous 2D Gaussian risk field with 90th percentile hotspot contours
    """
    rows = df_wafer["die_row"].values.astype(int)
    cols = df_wafer["die_col"].values.astype(int)
    n_r = int(rows.max()) + 1
    n_c = int(cols.max()) + 1

    pretest_grid = np.full((n_r, n_c), np.nan)
    newfail_grid = np.full((n_r, n_c), np.nan)
    prob_grid = np.full((n_r, n_c), np.nan)

    old_labels = df_wafer["old_label"].values
    has_lbl_col = has_ground_truth and ("label" in df_wafer.columns)
    labels = df_wafer["label"].values if has_lbl_col else np.zeros(len(df_wafer))

    for r, c, old_l, true_l, p in zip(rows, cols, old_labels, labels, y_prob):
        pretest_grid[r, c] = old_l
        if old_l == 0:
            if has_lbl_col:
                newfail_grid[r, c] = true_l
            prob_grid[r, c] = p

    # Continuous 2D Gaussian blur for risk-field
    prob_filled = np.nan_to_num(prob_grid, nan=0.0)
    valid_mask = ~np.isnan(prob_grid)
    norm_mask = gaussian_filter(valid_mask.astype(float), sigma=1.5)
    norm_mask = np.where(norm_mask > 0.05, norm_mask, 1.0)
    smoothed_risk = gaussian_filter(prob_filled, sigma=1.5) / norm_mask
    smoothed_risk[~valid_mask] = np.nan

    # Hotspots: regions with risk in top 10%
    valid_risks = smoothed_risk[valid_mask]
    hotspot_thresh = np.percentile(valid_risks, 90) if len(valid_risks) > 0 else 0.5
    hotspot_mask = (smoothed_risk >= hotspot_thresh) & valid_mask

    # Setup dark theme Matplotlib figure
    n_panels = 4 if has_lbl_col else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), facecolor="#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Panel 1: Pre-test map
    cmap_pre = mcolors.ListedColormap(["#22c55e", "#ef4444"])
    axes[0].imshow(pretest_grid, cmap=cmap_pre, vmin=0, vmax=1, aspect="equal")
    axes[0].set_title(f"1. Pre-Test Status\n(Green=Pass, Red=Old Fail)", color="#f8fafc", fontsize=9, pad=8)

    idx_curr = 1
    # Panel 2: Ground Truth (if available)
    if has_lbl_col:
        cmap_new = mcolors.ListedColormap(["#334155", "#ef4444"])
        axes[idx_curr].imshow(newfail_grid, cmap=cmap_new, vmin=0, vmax=1, aspect="equal")
        axes[idx_curr].set_title("2. Ground-Truth New Fails\n(Red=New Fail, Grey=Pass)", color="#f8fafc", fontsize=9, pad=8)
        idx_curr += 1

    # Panel 3: Predicted Probability
    im_prob = axes[idx_curr].imshow(prob_grid, cmap="plasma", vmin=0, vmax=1, aspect="equal")
    axes[idx_curr].set_title(f"{idx_curr+1}. {model_name} Predicted Prob\n(Threshold = {threshold:.3f})", color="#f8fafc", fontsize=9, pad=8)
    cb1 = plt.colorbar(im_prob, ax=axes[idx_curr], fraction=0.046, pad=0.04)
    cb1.ax.tick_params(colors="#94a3b8", labelsize=7)
    idx_curr += 1

    # Panel 4: Risk Field with Hotspot Contours
    vmax_risk = max(float(np.nanmax(smoothed_risk)) if np.any(~np.isnan(smoothed_risk)) else 0.8, 0.8)
    im_risk = axes[idx_curr].imshow(smoothed_risk, cmap="magma", vmin=0, vmax=vmax_risk, aspect="equal")
    if np.any(hotspot_mask):
        axes[idx_curr].contour(hotspot_mask, levels=[0.5], colors=["#38bdf8"], linewidths=[1.4])
    axes[idx_curr].set_title(f"{idx_curr+1}. Continuous Risk Field (2D Gaussian)\n(Cyan = Hotspots > 90th %ile)", color="#f8fafc", fontsize=9, pad=8)
    cb2 = plt.colorbar(im_risk, ax=axes[idx_curr], fraction=0.046, pad=0.04)
    cb2.ax.tick_params(colors="#94a3b8", labelsize=7)

    plt.tight_layout()
    return fig


def render_block_strip(
    readings: np.ndarray,
    die_row: int,
    die_col: int,
    wafer_id: str,
    mad_k: float = 2.0,
    base_mean: float = 100.0,
) -> matplotlib.figure.Figure:
    """
    Render strip plot of 2,000 sub-die block readings using the actual robust MAD-based
    anomaly threshold (|readings - median| > 2.0 * MAD) from src/block_features.py.
    """
    n = len(readings)
    indices = np.arange(n)

    median_ = float(np.median(readings))
    mad = float(np.median(np.abs(readings - median_)))
    if mad > 0:
        anom_mask = np.abs(readings - median_) > mad_k * mad
        lo_bound = median_ - mad_k * mad
        hi_bound = median_ + mad_k * mad
    else:
        anom_mask = np.zeros(n, dtype=bool)
        lo_bound = median_
        hi_bound = median_

    fig, ax = plt.subplots(figsize=(13, 3.8), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.plot(indices, readings, color="#38bdf8", linewidth=0.5, alpha=0.85, label="Sub-die block readings")
    if anom_mask.any():
        ax.scatter(
            indices[anom_mask], readings[anom_mask],
            color="#ef4444", s=14, zorder=5, label=f"Anomalous readings (>{mad_k}×MAD: {anom_mask.sum():,} blocks / {100*anom_mask.mean():.1f}%)"
        )

    ax.axhline(lo_bound, color="#f59e0b", linestyle="--", linewidth=0.9, label=f"Robust threshold: median ± {mad_k}×MAD [{lo_bound:.1f}, {hi_bound:.1f}]")
    ax.axhline(hi_bound, color="#f59e0b", linestyle="--", linewidth=0.9)
    ax.axhline(median_, color="#22c55e", linestyle="-", linewidth=1.0, alpha=0.8, label=f"Die median ({median_:.1f})")
    ax.axhline(base_mean, color="#94a3b8", linestyle=":", linewidth=0.8, alpha=0.6, label=f"Base population mean ({base_mean:.1f})")

    ax.set_xlabel("Sequential Block Array Index (0 to 1999 — NOTE: Index-Position only, NOT physical die geometry)", color="#cbd5e1", fontsize=9)
    ax.set_ylabel("Signal Reading Value", color="#cbd5e1", fontsize=9)
    ax.set_title(
        f"Die ({die_row}, {die_col}) on Wafer {wafer_id} — Sub-Die Block Signal Profile\n"
        f"Robust Anomalous Blocks (|x - median| > {mad_k}×MAD): {anom_mask.sum():,} / {n:,} ({100*anom_mask.mean():.2f}%)",
        color="#f8fafc", fontsize=10, pad=8
    )
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8, loc="upper right")
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def main():
    # Header
    st.markdown('<div class="main-header">Semiconductor Die Yield Prediction & Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production Model Comparison, Wafer Risk Fields, and Per-Die Explainability</div>', unsafe_allow_html=True)
    st.info("ℹ️ **Data Disclosure**: Real WM-811K wafer geometry and pre-test maps combined with synthetic die-level parametric measurements, synthetic sub-die block readings, and controlled new-failure labels.")

    # 1. Load data and models
    meta_df = load_test_metadata()
    pred_df = load_predictions()
    model_a, meta_a, model_b, meta_b = load_models()
    prob_a, prob_b = load_probabilities()
    prob_a_cal, prob_b_cal = load_calibrated_probabilities()
    comp_df, abl_df, sig_df = load_tables()
    pass_med_a, pass_med_b = load_pass_medians()

    wafers = sorted(meta_df["wafer_id"].unique())

    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Navigation & Controls")

        # Demo Presets
        st.markdown("#### 🎯 Presentation Demo Presets")
        preset_names = [
            "Manual Exploration",
            "Preset 1: W_F_0014 (40,18) — Spatial Context True Fail",
            "Preset 2: W_N_0066 (2,13) — Block-driven B-only Catch",
            "Preset 3: W_F_0047 (4,10) — Second B-only Catch",
            "Preset 4: W_F_0016 (21,12) — Borderline False Alarm (Pass)",
        ]
        chosen_preset = st.selectbox(
            "Demo Presets",
            options=preset_names,
            index=0,
            key="preset_select",
            help="Select one of the 4 benchmark hackathon demonstration dies."
        )

        preset_map = {
            "Preset 1: W_F_0014 (40,18) — Spatial Context True Fail": ("W_F_0014", 40, 18, "preset_1"),
            "Preset 2: W_N_0066 (2,13) — Block-driven B-only Catch": ("W_N_0066", 2, 13, "preset_2"),
            "Preset 3: W_F_0047 (4,10) — Second B-only Catch": ("W_F_0047", 4, 10, "preset_3"),
            "Preset 4: W_F_0016 (21,12) — Borderline False Alarm (Pass)": ("W_F_0016", 21, 12, "preset_4"),
        }

        preset_data = preset_map.get(chosen_preset)
        if preset_data is not None:
            preset_wafer, preset_row, preset_col, preset_id = preset_data
            wafer_default_idx = wafers.index(preset_wafer) if preset_wafer in wafers else 0
        else:
            preset_wafer, preset_row, preset_col, preset_id = None, None, None, "manual"
            wafer_default_idx = wafers.index("W_F_0014") if "W_F_0014" in wafers else 0

        # Wafer selection
        selected_wafer = st.selectbox(
            "Select Wafer ID",
            options=wafers,
            index=wafer_default_idx,
            key=f"wafer_select_{preset_id}",
            help="Choose a test wafer to inspect wafer spatial patterns and per-die diagnostics."
        )

        # Model mode
        model_mode = st.radio(
            "Model Selection",
            options=["Model B (Full Diagnostic)", "Model A (Baseline + Spatial)", "Side-by-side Comparison"],
            index=0,
            key="model_mode_select",
            help="Switch between Model A, Model B, or view both side-by-side."
        )

        st.markdown("---")
        st.markdown("### ℹ️ Dataset & Model Specs")
        st.markdown(f"""
        - **Total Test Wafers**: {len(wafers)}
        - **Test Dies**: {len(meta_df):,}
        - **Model A Threshold**: `{meta_a['threshold']:.4f}`
        - **Model B Threshold**: `{meta_b['threshold']:.4f}`
        - **Algorithm**: LightGBM Classifier
        - **Calibration**: {'Platt Scaling (Active)' if prob_a_cal is not None else 'Uncalibrated'}
        """)
        st.markdown("---")
        st.caption("Sandisk Die Yield Prediction Hackathon Deliverable")

    # Slice data for selected wafer
    w_mask = meta_df["wafer_id"] == selected_wafer
    w_indices = np.where(w_mask)[0]
    w_meta = meta_df.iloc[w_indices].reset_index(drop=True)
    w_prob_a = prob_a[w_indices]
    w_prob_b = prob_b[w_indices]
    w_prob_a_cal = prob_a_cal[w_indices] if prob_a_cal is not None else None
    w_prob_b_cal = prob_b_cal[w_indices] if prob_b_cal is not None else None

    n_w_dies = len(w_meta)
    n_w_old_fails = int((w_meta["old_label"] == 1).sum())
    n_w_eligible = int((w_meta["old_label"] == 0).sum())
    has_gt = "label" in w_meta.columns
    n_w_true_new_fails = int((w_meta.loc[w_meta["old_label"] == 0, "label"] == 1).sum()) if has_gt else None

    # Overview KPI Cards
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    with col_kpi1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Selected Wafer</div><div class="metric-value" style="font-size: 1.3rem;">{selected_wafer}</div></div>', unsafe_allow_html=True)
    with col_kpi2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Dies</div><div class="metric-value">{n_w_dies:,}</div></div>', unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Pre-Test Fails</div><div class="metric-value">{n_w_old_fails:,}</div></div>', unsafe_allow_html=True)
    with col_kpi4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Eligible Dies</div><div class="metric-value">{n_w_eligible:,}</div></div>', unsafe_allow_html=True)
    with col_kpi5:
        gt_display = f"{n_w_true_new_fails:,}" if has_gt else "N/A"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Actual New Fails</div><div class="metric-value">{gt_display}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 1: 4-Panel Wafer View
    # -------------------------------------------------------------------------
    st.markdown("### 🗺️ Wafer Spatial Map & Continuous Risk Fields")

    if model_mode == "Side-by-side Comparison":
        st.markdown("#### Model A (Die + Spatial + Die Anomaly)")
        fig_a = render_wafer_4panel(
            w_meta, w_prob_a, meta_a["threshold"], selected_wafer, "Model A", has_ground_truth=has_gt
        )
        st.pyplot(fig_a)
        plt.close(fig_a)

        st.markdown("#### Model B (Model A + Block Summary & Anomaly)")
        fig_b = render_wafer_4panel(
            w_meta, w_prob_b, meta_b["threshold"], selected_wafer, "Model B", has_ground_truth=has_gt
        )
        st.pyplot(fig_b)
        plt.close(fig_b)
    elif model_mode == "Model A (Baseline + Spatial)":
        fig = render_wafer_4panel(
            w_meta, w_prob_a, meta_a["threshold"], selected_wafer, "Model A", has_ground_truth=has_gt
        )
        st.pyplot(fig)
        plt.close(fig)
    else:  # Model B
        fig = render_wafer_4panel(
            w_meta, w_prob_b, meta_b["threshold"], selected_wafer, "Model B", has_ground_truth=has_gt
        )
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("<hr style='margin: 25px 0; border-color: #334155;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 2: Die Selector & Detailed Explanation Panel
    # -------------------------------------------------------------------------
    st.markdown("### 🔍 Die-Level Diagnostic Inspector & SHAP Attribution")

    # Determine active model parameters for die inspection
    active_is_model_b = ("Model B" in model_mode) or (model_mode == "Side-by-side Comparison")
    active_model_name = "Model B" if active_is_model_b else "Model A"
    active_probs = w_prob_b if active_is_model_b else w_prob_a
    active_threshold = meta_b["threshold"] if active_is_model_b else meta_a["threshold"]
    active_model = model_b if active_is_model_b else model_a
    active_pass_med = pass_med_b if active_is_model_b else pass_med_a

    # Build die options list
    w_meta["_prob"] = active_probs
    w_meta["_pred"] = (active_probs >= active_threshold).astype(int)
    w_meta.loc[w_meta["old_label"] == 1, "_pred"] = 1

    # Check if this wafer contains any benchmark dies
    wafer_reps = [coord for coord in PRECOMPUTED_REPRESENTATIVES.keys() if coord[0] == selected_wafer]

    col_sel1, col_sel2 = st.columns([2, 3])
    with col_sel1:
        # Pre-select benchmark die if present, else highest risk eligible die
        top_risk_dies = w_meta.sort_values("_prob", ascending=False)
        die_coord_options = [
            f"Row {int(r)}, Col {int(c)} (Prob: {p*100:.1f}%, Status: {'FAIL' if pr==1 else 'PASS'})"
            for r, c, p, pr in zip(top_risk_dies["die_row"], top_risk_dies["die_col"], top_risk_dies["_prob"], top_risk_dies["_pred"])
        ]

        # Check default index
        default_die_idx = 0
        if preset_row is not None and preset_col is not None and selected_wafer == preset_wafer:
            for i, opt in enumerate(die_coord_options):
                if f"Row {preset_row}, Col {preset_col} " in opt:
                    default_die_idx = i
                    break
        elif wafer_reps:
            rep_r, rep_c = wafer_reps[0][1], wafer_reps[0][2]
            for i, opt in enumerate(die_coord_options):
                if f"Row {rep_r}, Col {rep_c} " in opt:
                    default_die_idx = i
                    break

        selected_die_str = st.selectbox(
            "Select Die to Inspect (Sorted by Failure Probability)",
            options=die_coord_options,
            index=default_die_idx,
            key=f"die_{selected_wafer}_{preset_id}",
            help="Select any die on the wafer grid to inspect local feature SHAP attributions, failure signature clustering, and counterfactuals."
        )

        # Parse selected coordinates
        import re
        match = re.search(r"Row (\d+), Col (\d+)", selected_die_str)
        sel_row = int(match.group(1))
        sel_col = int(match.group(2))

    # Retrieve selected die metadata and local index
    sel_local_idx = int(w_meta[(w_meta["die_row"] == sel_row) & (w_meta["die_col"] == sel_col)].index[0])
    sel_global_idx = int(w_indices[sel_local_idx])
    sel_die_row = w_meta.iloc[sel_local_idx]

    sel_prob = float(active_probs[sel_local_idx])
    sel_pred = int(w_meta.iloc[sel_local_idx]["_pred"])
    sel_old_label = int(sel_die_row["old_label"])
    sel_gt = int(sel_die_row["label"]) if has_gt else None

    with col_sel2:
        st.markdown(f"**Die Coordinates: ({sel_row}, {sel_col}) on Wafer `{selected_wafer}`**")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Predicted Failure Probability", f"{sel_prob*100:.2f}%")
        with col_m2:
            st.metric("Tuned Decision Threshold", f"{active_threshold:.4f}")
        with col_m3:
            st.metric("Pre-Test Status", "FAIL (old_label=1)" if sel_old_label == 1 else "PASS (Eligible)")
        with col_m4:
            gt_text = ("FAIL" if sel_gt == 1 else "PASS") if has_gt else "N/A"
            st.metric("Ground-Truth Target", gt_text)

        # 5.2 A-vs-B Decision Strip
        p_a_die = float(w_prob_a[sel_local_idx])
        p_b_die = float(w_prob_b[sel_local_idx])
        dec_a = (p_a_die >= meta_a["threshold"]) if sel_old_label == 0 else True
        dec_b = (p_b_die >= meta_b["threshold"]) if sel_old_label == 0 else True
        diff_pct = (p_b_die - p_a_die) * 100

        st.markdown("---")
        st.markdown("**⚖️ Model A vs. Model B Head-to-Head Decision Strip**")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(f"**Model A**: `{p_a_die*100:.1f}%` → **{'FAIL' if dec_a else 'PASS'}** (cutoff: `{meta_a['threshold']:.4f}`)")
        with col_s2:
            st.markdown(f"**Model B**: `{p_b_die*100:.1f}%` → **{'FAIL' if dec_b else 'PASS'}** (cutoff: `{meta_b['threshold']:.4f}`)")
        with col_s3:
            st.markdown(f"**Shift (B − A)**: `{diff_pct:+.1f}%` ({'Both Agree' if dec_a == dec_b else 'Decision Diverges'})")

    # Compute SHAP values for this wafer (cached on wafer selection)
    shap_vals, feat_cols, X_wafer = compute_wafer_shap(selected_wafer, model_type=active_model_name)
    die_shap = shap_vals[sel_local_idx]
    die_X = X_wafer.iloc[sel_local_idx]

    # Display SHAP breakdown
    col_shap1, col_shap2 = st.columns([3, 2])
    with col_shap1:
        st.markdown(f"#### Top Contributing Features for Die ({sel_row}, {sel_col}) — {active_model_name}")

        # Sort features by absolute SHAP attribution
        feat_order = np.argsort(np.abs(die_shap))[::-1]
        top_n = 8
        top_indices = feat_order[:top_n]

        top_names = [feat_cols[i] for i in top_indices]
        top_shaps = [die_shap[i] for i in top_indices]
        top_vals = [die_X[feat_cols[i]] for i in top_indices]

        # Horizontal bar chart of top features
        fig_bar, ax_bar = plt.subplots(figsize=(7, 3.8), facecolor="#0f172a")
        ax_bar.set_facecolor("#1e293b")
        ax_bar.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax_bar.spines.values():
            spine.set_color("#334155")

        y_positions = np.arange(top_n)[::-1]
        colors = ["#ef4444" if s > 0 else "#22c55e" for s in top_shaps]
        ax_bar.barh(y_positions, top_shaps, color=colors, height=0.6)
        ax_bar.axvline(0, color="#94a3b8", linewidth=0.8, linestyle="--")

        formatted_labels = [
            f"{name} ({val:.2g}) (context, not an intervention)" if name.startswith("sp_")
            else f"{name} ({val:.2g})"
            for name, val in zip(top_names, top_vals)
        ]
        ax_bar.set_yticks(y_positions)
        ax_bar.set_yticklabels(formatted_labels, color="#f8fafc", fontsize=8)
        ax_bar.set_xlabel("SHAP Attribution (Red = Increases Failure Risk, Green = Reduces Risk)", color="#cbd5e1", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    with col_shap2:
        st.markdown("#### Domain Attribution Breakdown")
        
        # Domain contributions
        die_contrib = float(sum(abs(die_shap[i]) for i, f in enumerate(feat_cols) if f.startswith("feature_")))
        sp_contrib = float(sum(abs(die_shap[i]) for i, f in enumerate(feat_cols) if f.startswith("sp_")))
        blk_contrib = float(sum(abs(die_shap[i]) for i, f in enumerate(feat_cols) if f.startswith("blk_")))
        anom_contrib = float(sum(abs(die_shap[i]) for i, f in enumerate(feat_cols) if "anomaly" in f))
        total_mass = max(die_contrib + sp_contrib + blk_contrib + anom_contrib, 1e-9)

        domain_df = pd.DataFrame({
            "Domain": [
                "Die Parametric (500)",
                "Spatial Neighborhood (10) (context, not an intervention)",
                "Sub-Die Block (19)",
                "Anomaly Scores (1-2)"
            ],
            "Attribution Mass": [die_contrib, sp_contrib, blk_contrib, anom_contrib],
            "Percentage": [
                f"{die_contrib/total_mass*100:.1f}%",
                f"{sp_contrib/total_mass*100:.1f}%",
                f"{blk_contrib/total_mass*100:.1f}%",
                f"{anom_contrib/total_mass*100:.1f}%",
            ]
        })
        st.dataframe(domain_df, use_container_width=True, hide_index=True)

        # Failure Signature matching
        st.markdown("#### Signature (rule-based on top SHAP driver; HDBSCAN clusters derived offline)")
        if sel_pred == 1 or sel_prob >= active_threshold:
            # Map top feature to signature
            top_f = top_names[0]
            if "blk_" in top_f:
                sig_text = "**Cluster 0 / 1: Block-Reading Signal Drift / Memory Array Shift**"
                sig_desc = "Driven by sub-die block voltage anomalies indicating local memory array degradation."
            elif "sp_dist" in top_f or "sp_old" in top_f:
                sig_text = "**Cluster 2 / 3: Defect Neighborhood Proximity & Spatial Clustering**"
                sig_desc = "Driven by physical proximity to existing pre-test wafer defect clusters."
            else:
                sig_text = "**Cluster -1: Mixed / Parametric Electrical Breakdown**"
                sig_desc = "Multi-parametric electrical shift across die-level measurements."
            
            st.info(f"{sig_text}\n\n_{sig_desc}_")
        else:
            st.success("Die is predicted as **PASS** (Normal operating population).")

    # Counterfactual explanation box
    st.markdown("#### 🔄 Model-Based Counterfactual Sensitivity Analysis")
    rep_key = (selected_wafer, sel_row, sel_col)
    if rep_key in PRECOMPUTED_REPRESENTATIVES:
        rep_info = PRECOMPUTED_REPRESENTATIVES[rep_key]
        st.markdown(f"**Representative Benchmark Die Identified**: `{rep_info['title']}`")
        st.markdown(f"""
        - **Model Confidence**: {rep_info['confidence']}
        - **Failure Signature**: {rep_info['cluster']}
        - **Stepwise Normalization Trajectory**:  
          `{rep_info['trajectory']}`
        - **Total Achievable Risk Reduction**: **{rep_info['reduction']}**
        """)
    else:
        st.markdown(f"""
        <div class="disclaimer-box">
            <b>Note on Counterfactual Explanations:</b> Full 5-model bootstrap ensemble uncertainty trajectories were precomputed for representative benchmark cases (e.g. Wafer <code>W_F_0014</code> Row 40, Col 18; Wafer <code>W_F_0019</code> Row 22, Col 22; Wafer <code>W_F_0016</code> Row 21, Col 12).
        </div>
        """, unsafe_allow_html=True)
        
        # Lightweight single-feature live counterfactual re-score (using top non-spatial driver)
        non_spatial_drivers = [f for f in top_names if not f.startswith("sp_")]
        top_driver = non_spatial_drivers[0] if non_spatial_drivers else top_names[0]
        if top_driver in active_pass_med:
            die_vec_cf = die_X.copy().to_frame().T
            die_vec_cf[top_driver] = active_pass_med[top_driver]
            p_cf = float(active_model.predict_proba(die_vec_cf[feat_cols].values)[0, 1])
            delta_p = sel_prob - p_cf
            st.markdown(f"""
            - **Live Sensitivity Test**: Normalizing top non-spatial driver `{top_driver}` from `{die_X[top_driver]:.3g}` to healthy median `{active_pass_med[top_driver]:.3g}` shifts failure probability from **{sel_prob*100:.1f}%** to **{p_cf*100:.1f}%** (Δ = {delta_p*100:+.1f}%).
            """)

    st.caption("⚠️ **Disclaimer**: Model-based mathematical risk adjustment estimate, NOT a physical semiconductor manufacturing simulation or causal intervention.")

    # -------------------------------------------------------------------------
    # SECTION 3: Sub-Die Block View (Model B Only)
    # -------------------------------------------------------------------------
    if active_is_model_b:
        st.markdown("<hr style='margin: 25px 0; border-color: #334155;'>", unsafe_allow_html=True)
        st.markdown("### 📊 Sub-Die Block Signal Profile (Model B Feature View)")
        
        w_blk_df = load_wafer_block_readings(selected_wafer)
        die_blk_row = w_blk_df[
            (w_blk_df["die_row"] == sel_row) &
            (w_blk_df["die_col"] == sel_col)
        ]

        if not die_blk_row.empty:
            raw_str = die_blk_row.iloc[0]["block_readings"]
            readings_arr = np.fromstring(raw_str, sep=" ")
            
            fig_strip = render_block_strip(readings_arr, sel_row, sel_col, selected_wafer)
            st.pyplot(fig_strip)
            plt.close(fig_strip)
            
            st.caption(
                "📌 **Physical Mapping & Threshold Notice**: Anomalous points are highlighted using the robust MAD threshold from `src/block_features.py` (|reading - median| > 2.0 × MAD). "
                "The X-axis indicates sequential index position within the stream (0..1999) — NOT genuine physical 2D/3D spatial coordinates within the die stack."
            )
            st.info(
                "📌 **Sub-Die Feature Honesty Note**: Mean, spread and quartile statistics carry the signal; "
                "run/cluster statistics (`blk_n_anom_fixed`, `blk_frac_anom_fixed`, `blk_longest_run`, `blk_n_clusters`) "
                "were inert in this setting."
            )
        else:
            st.info(f"Block reading stream for Die ({sel_row}, {sel_col}) is not available in test store.")

    # -------------------------------------------------------------------------
    # SECTION: Budget-Aware Screening Triage & Risk Prioritization (Phase 4)
    # -------------------------------------------------------------------------
    st.markdown("<hr style='margin: 25px 0; border-color: #334155;'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Budget-Aware Screening Triage & Risk Prioritization")
    st.markdown(
        "In production wafer testing, inspection bandwidth is strictly bounded. Rather than relying solely on a fixed threshold, "
        "manufacturing teams prioritize dies by predicted risk score. Compare Model A and Model B failure capture rates "
        "across varying inspection budgets."
    )

    col_tr_ctrl1, col_tr_ctrl2 = st.columns([1, 1])
    with col_tr_ctrl1:
        triage_scope = st.radio(
            "Evaluation Scope",
            options=["All Test Wafers (40 Wafers, 32,598 Eligible Dies)", f"Selected Wafer ({selected_wafer})"],
            horizontal=True,
            key="triage_scope_select"
        )
    with col_tr_ctrl2:
        quick_pick = st.radio(
            "Quick Inspection Budget Presets",
            options=["5.0% (Default)", "1.0%", "2.0%", "10.0%", "Custom Slider"],
            horizontal=True,
            index=0,
            key="triage_quick_pick"
        )

    if quick_pick == "1.0%":
        screen_k_pct = 1.0
    elif quick_pick == "2.0%":
        screen_k_pct = 2.0
    elif quick_pick == "5.0% (Default)":
        screen_k_pct = 5.0
    elif quick_pick == "10.0%":
        screen_k_pct = 10.0
    else:
        screen_k_pct = st.slider(
            "Screen top K% of eligible dies",
            min_value=0.5,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="triage_k_slider"
        )

    # Determine slice
    use_cal = (prob_a_cal is not None and prob_b_cal is not None)
    if triage_scope.startswith("All Test Wafers"):
        el_mask = (meta_df["old_label"] == 0).values
        y_scope = meta_df.loc[el_mask, "label"].values if "label" in meta_df else np.zeros(el_mask.sum())
        s_a = prob_a[el_mask]
        s_b = prob_b[el_mask]
        s_a_cal = prob_a_cal[el_mask] if use_cal else None
        s_b_cal = prob_b_cal[el_mask] if use_cal else None
    else:
        el_mask = (w_meta["old_label"] == 0).values
        y_scope = w_meta.loc[el_mask, "label"].values if "label" in w_meta else np.zeros(el_mask.sum())
        s_a = w_prob_a[el_mask]
        s_b = w_prob_b[el_mask]
        s_a_cal = w_prob_a_cal[el_mask] if use_cal else None
        s_b_cal = w_prob_b_cal[el_mask] if use_cal else None

    # Compute metrics
    stats_a = topk_stats(y_scope, s_a, screen_k_pct / 100.0, score_cal=s_a_cal)
    stats_b = topk_stats(y_scope, s_b, screen_k_pct / 100.0, score_cal=s_b_cal)

    # Side-by-side metrics
    score_label = "Mean Calibrated Failure Risk" if use_cal else "Mean Risk Score (Uncalibrated)"
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### Model A (Spatial Baseline)")
        col_sub1, col_sub2, col_sub3 = st.columns(3)
        with col_sub1:
            st.metric("Dies Screened", f"{stats_a['n_screened']:,} ({screen_k_pct:.1f}%)")
            st.metric("Observed Failure Rate (Precision)", f"{stats_a['observed_fail_rate']*100:.1f}%")
        with col_sub2:
            st.metric("Failures Captured", f"{stats_a['n_fails_captured']:,} / {stats_a['n_fails_total']:,}")
            st.metric("% of All Failures Caught", f"{stats_a['capture_rate']*100:.1f}%")
        with col_sub3:
            st.metric("Screening Lift", f"{stats_a['lift']:.2f}×")
            cal_rate_a = f"{stats_a['mean_calibrated_rate']*100:.1f}%" if stats_a['mean_calibrated_rate'] is not None else f"{stats_a['mean_score']:.3f}"
            st.metric(score_label, cal_rate_a)

    with col_t2:
        st.markdown("#### Model B (Full Diagnostic Model)")
        col_sub1, col_sub2, col_sub3 = st.columns(3)
        with col_sub1:
            st.metric("Dies Screened", f"{stats_b['n_screened']:,} ({screen_k_pct:.1f}%)")
            diff_prec = (stats_b['observed_fail_rate'] - stats_a['observed_fail_rate']) * 100
            st.metric("Observed Failure Rate (Precision)", f"{stats_b['observed_fail_rate']*100:.1f}%", delta=f"{diff_prec:+.1f}%")
        with col_sub2:
            diff_fails = stats_b['n_fails_captured'] - stats_a['n_fails_captured']
            st.metric("Failures Captured", f"{stats_b['n_fails_captured']:,} / {stats_b['n_fails_total']:,}", delta=f"{diff_fails:+d} dies")
            diff_cap = (stats_b['capture_rate'] - stats_a['capture_rate']) * 100
            st.metric("% of All Failures Caught", f"{stats_b['capture_rate']*100:.1f}%", delta=f"{diff_cap:+.1f}%")
        with col_sub3:
            diff_lift = stats_b['lift'] - stats_a['lift']
            st.metric("Screening Lift", f"{stats_b['lift']:.2f}×", delta=f"{diff_lift:+.2f}×")
            cal_rate_b = f"{stats_b['mean_calibrated_rate']*100:.1f}%" if stats_b['mean_calibrated_rate'] is not None else f"{stats_b['mean_score']:.3f}"
            st.metric(score_label, cal_rate_b)

    # Render Gains Curve
    k_pts_a, cap_curve_a, _ = capture_curve(y_scope, s_a, n_points=200)
    k_pts_b, cap_curve_b, _ = capture_curve(y_scope, s_b, n_points=200)

    fig_gain, ax_gain = plt.subplots(figsize=(10, 4.2), facecolor="#0f172a")
    ax_gain.set_facecolor("#1e293b")
    ax_gain.plot([0, 100], [0, 100], "--", color="#64748b", linewidth=1.2, label="Random Screening Baseline (Lift = 1.0×)")
    ax_gain.plot(k_pts_a * 100, cap_curve_a * 100, color="#38bdf8", linewidth=2.0, label="Model A (Spatial Baseline)")
    ax_gain.plot(k_pts_b * 100, cap_curve_b * 100, color="#a855f7", linewidth=2.2, label="Model B (Full Diagnostic)")

    ax_gain.axvline(screen_k_pct, color="#10b981", linestyle=":", linewidth=1.5, label=f"Current Budget: {screen_k_pct:.1f}%")
    ax_gain.scatter([screen_k_pct], [stats_a['capture_rate'] * 100], color="#38bdf8", s=45, zorder=5)
    ax_gain.scatter([screen_k_pct], [stats_b['capture_rate'] * 100], color="#a855f7", s=45, zorder=5)

    ax_gain.set_xlim(0, 20)
    ax_gain.set_ylim(0, 80)
    ax_gain.set_xlabel("Screening Budget (% of Eligible Dies Inspected)", color="#cbd5e1", fontsize=9)
    ax_gain.set_ylabel("Defect Capture Rate (% of True Failures Caught)", color="#cbd5e1", fontsize=9)
    ax_gain.set_title("Screening Gains Curve — Defect Capture vs. Inspection Budget", color="#f8fafc", fontsize=10, pad=8)
    ax_gain.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax_gain.spines.values():
        spine.set_color("#334155")
    ax_gain.grid(True, linestyle=":", alpha=0.4, color="#475569")
    ax_gain.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig_gain)
    plt.close(fig_gain)

    # -------------------------------------------------------------------------
    # SECTION 4: Model A vs Model B Comparison & Audit Findings
    # -------------------------------------------------------------------------
    st.markdown("<hr style='margin: 25px 0; border-color: #334155;'>", unsafe_allow_html=True)
    st.markdown("### 🏆 Production Model Comparison & Ablation Audit")

    st.markdown("""
    > **Key Architectural Takeaway**:  
    > *Test set, 40 held-out wafers, wafer-cluster bootstrap: PR-AUC +0.034 [+0.023, +0.043]; Fail F1 +0.001 [-0.008, +0.012], not distinguishable from zero. Ablation = 5 repeated wafer splits (validation).*
    """)

    col_tbl1, col_tbl2 = st.columns([1, 1])
    with col_tbl1:
        st.markdown("#### Holdout Test Set Performance (`outputs/comparison_table.csv`)")
        # Format comparison table cleanly
        clean_comp = pd.DataFrame({
            "Model": ["Model A (Parametric + Spatial + Anom)", "Model B (Model A + Block Summary & Anom)"],
            "Fail F1": [f"{comp_df.iloc[0]['fail_f1']:.4f}", f"{comp_df.iloc[1]['fail_f1']:.4f}"],
            "PR-AUC": [f"{comp_df.iloc[0]['pr_auc']:.4f}", f"{comp_df.iloc[1]['pr_auc']:.4f}"],
            "Fail Recall": [f"{comp_df.iloc[0]['fail_recall']*100:.2f}%", f"{comp_df.iloc[1]['fail_recall']*100:.2f}%"],
            "Fail Precision": [f"{comp_df.iloc[0]['fail_precision']*100:.2f}%", f"{comp_df.iloc[1]['fail_precision']*100:.2f}%"],
            "Overall Accuracy": [f"{comp_df.iloc[0]['overall_accuracy']*100:.2f}%", f"{comp_df.iloc[1]['overall_accuracy']*100:.2f}%"],
            "Tuned Threshold": [f"{meta_a['threshold']:.4f}", f"{meta_b['threshold']:.4f}"],
        })
        st.dataframe(clean_comp, use_container_width=True, hide_index=True)

    with col_tbl2:
        st.markdown("#### 5-Seed Validation Ablation (`outputs/ablation_table_multiseed.csv`)")
        clean_abl = pd.DataFrame({
            "Feature Configuration": abl_df["Feature Set"],
            "PR-AUC (Mean ± Std)": [f"{m:.4f} ± {s:.4f}" for m, s in zip(abl_df["PR-AUC mean"], abl_df["PR-AUC std"])],
            "Fail F1 (Mean ± Std)": [f"{m:.4f} ± {s:.4f}" for m, s in zip(abl_df["Fail F1 mean"], abl_df["Fail F1 std"])],
            "Fail Recall": [f"{m*100:.1f}%" for m in abl_df["Fail Rec mean"]],
            "Fail Precision": [f"{m*100:.1f}%" for m in abl_df["Fail Prec mean"]],
        })
        st.dataframe(clean_abl, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # 5.3 B-Only Catches Table (29 Gained Failures)
    # -------------------------------------------------------------------------
    st.markdown("<hr style='margin: 20px 0; border-color: #334155;'>", unsafe_allow_html=True)
    st.markdown("#### 🔍 Discrepancy Analysis: Gained Failures Uniquely Caught by Model B")
    st.markdown(
        "Across all 40 test wafers, Model B catches **29 additional true post-burn-in failures** that were missed by Model A, "
        "while missing 8 failures that Model A caught (**net gain of +21 caught defects**). "
        "At the fixed operating thresholds, Model B incurs extra false alarms (69 on Model B vs 15 on Model A), "
        "illustrating the classical operational trade-off between defect capture recall and false scrap cost."
    )

    el_all = (meta_df["old_label"] == 0)
    y_all = meta_df.loc[el_all, "label"].values if "label" in meta_df else np.zeros(el_all.sum())
    pa_all = prob_a[el_all.values]
    pb_all = prob_b[el_all.values]
    th_a = meta_a["threshold"]
    th_b = meta_b["threshold"]

    gained_mask = (y_all == 1) & (pa_all < th_a) & (pb_all >= th_b)
    gained_dies = meta_df[el_all][gained_mask][["wafer_id", "die_row", "die_col"]].copy()
    gained_dies["Model A Risk"] = [f"{p*100:.1f}%" for p in pa_all[gained_mask]]
    gained_dies["Model B Risk"] = [f"{p*100:.1f}%" for p in pb_all[gained_mask]]
    gained_dies["Risk Shift (B − A)"] = [f"{(b - a)*100:+.1f}%" for a, b in zip(pa_all[gained_mask], pb_all[gained_mask])]
    gained_dies["Status"] = "Caught by Model B Only (True Post-Test Fail)"

    st.markdown(f"**Unique Model B Catches List ({len(gained_dies)} dies)**:")
    st.dataframe(gained_dies.reset_index(drop=True), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
