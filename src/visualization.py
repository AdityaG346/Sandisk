"""
src/visualization.py
=====================
Wafer map visualizations and block strip plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
from typing import Optional, List


WAFER_CMAP = {0: "#DDDDDD", 1: "#2ECC71", 2: "#E74C3C"}  # no-die, pass, fail


# ---------------------------------------------------------------------------
# Per-wafer 4-panel visualization
# ---------------------------------------------------------------------------

def _wafer_grid(df_wafer: pd.DataFrame, value_col: str, fill: float = 0.0) -> np.ndarray:
    """Reconstruct dense 2D grid from wafer die DataFrame."""
    n_rows = int(df_wafer["die_row"].max()) + 1
    n_cols = int(df_wafer["die_col"].max()) + 1
    grid = np.full((n_rows, n_cols), np.nan)
    for _, row in df_wafer.iterrows():
        grid[int(row["die_row"]), int(row["die_col"])] = row[value_col]
    return grid


def plot_wafer_4panel(
    df_wafer: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    wafer_id: str,
    output_dir: str = "outputs/wafer_maps",
    model_name: str = "Model",
) -> None:
    """
    4-panel wafer visualization:
      Panel 1: Pre-test map (old_label)
      Panel 2: Ground-truth new fails (label, masked to old_label=0)
      Panel 3: Predicted-fail map
      Panel 4: Probability heatmap
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # We need to align y_pred and y_prob indices with df_wafer
    df = df_wafer.copy().reset_index(drop=True)
    df["_pred"] = y_pred
    df["_prob"] = y_prob

    n_rows = int(df["die_row"].max()) + 1
    n_cols = int(df["die_col"].max()) + 1

    def make_grid(col):
        grid = np.full((n_rows, n_cols), np.nan)
        for _, row in df.iterrows():
            grid[int(row["die_row"]), int(row["die_col"])] = row[col]
        return grid

    old_label_grid = make_grid("old_label")
    label_grid = make_grid("label")
    pred_grid = make_grid("_pred")
    prob_grid = make_grid("_prob")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Wafer {wafer_id} -- {model_name}", fontsize=13)

    # Panel 1: Pre-test (old_label)
    _plot_binary_map(axes[0], old_label_grid, "Pre-test (old_label)", cmap_pass="#2ECC71", cmap_fail="#E74C3C")

    # Panel 2: Ground truth new fails (eligible dies only)
    new_fail_grid = label_grid.copy()
    new_fail_grid[old_label_grid == 1] = np.nan  # mask old fails
    _plot_binary_map(axes[1], new_fail_grid, "Ground Truth (new fails only)", cmap_pass="#2ECC71", cmap_fail="#E74C3C")

    # Panel 3: Predicted map
    _plot_binary_map(axes[2], pred_grid, f"Predicted ({model_name})", cmap_pass="#2ECC71", cmap_fail="#E74C3C")

    # Panel 4: Probability heatmap
    im = axes[3].imshow(prob_grid, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    axes[3].set_title("Failure Probability")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fname = out / f"wafer_{wafer_id}_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved wafer map: {fname}")


def _plot_binary_map(ax, grid, title, cmap_pass="#2ECC71", cmap_fail="#E74C3C"):
    """Plot a 2D grid where NaN=no die, 0=pass, 1=fail."""
    display = np.full((*grid.shape, 3), 0.85)  # default grey (no die)
    pass_color = np.array(mcolors.to_rgb(cmap_pass))
    fail_color = np.array(mcolors.to_rgb(cmap_fail))

    display[grid == 0] = pass_color
    display[grid == 1] = fail_color
    # NaN stays grey

    ax.imshow(display, aspect="auto")
    ax.set_title(title, fontsize=10)
    legend_elements = [
        Patch(facecolor=cmap_pass, label="Pass"),
        Patch(facecolor=cmap_fail, label="Fail"),
        Patch(facecolor="#DDDDDD", label="No die / masked"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)


# ---------------------------------------------------------------------------
# Block strip plot
# ---------------------------------------------------------------------------

def plot_block_strip(
    readings: np.ndarray,
    die_row: int,
    die_col: int,
    wafer_id: str,
    output_dir: str = "outputs/plots",
    base_mean: float = 100.0,
    base_std: float = 15.0,
    anomaly_threshold_k: float = 3.0,
) -> None:
    """
    Strip plot of 2000 block readings (index vs. value) with anomalous
    positions highlighted in red.
    Note: index is just sequential -- no physical block coordinate implied.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = len(readings)
    indices = np.arange(n)

    lo = base_mean - anomaly_threshold_k * base_std
    hi = base_mean + anomaly_threshold_k * base_std
    anom_mask = (readings < lo) | (readings > hi)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(indices, readings, color="#7BAFD4", linewidth=0.4, alpha=0.8, label="Block reading")
    ax.scatter(
        indices[anom_mask], readings[anom_mask],
        color="#E74C3C", s=10, zorder=5, label=f"Anomalous (>{anomaly_threshold_k}σ from base)"
    )
    ax.axhline(lo, color="#999999", linestyle="--", linewidth=0.8, label=f"±{anomaly_threshold_k}σ threshold")
    ax.axhline(hi, color="#999999", linestyle="--", linewidth=0.8)
    ax.axhline(base_mean, color="#2ECC71", linestyle="-", linewidth=0.8, alpha=0.5, label="Base mean")

    ax.set_xlabel("Block index (sequential -- no physical coordinate)", fontsize=10)
    ax.set_ylabel("Block reading value")
    ax.set_title(
        f"Block readings -- Die ({die_row},{die_col}) on wafer {wafer_id}\n"
        f"n_anomalous = {anom_mask.sum()} / {n} ({100*anom_mask.mean():.1f}%)"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()

    fname = out / f"block_strip_{wafer_id}_r{die_row}_c{die_col}.png"
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved block strip: {fname}")


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_fail_rate_distribution(
    df: pd.DataFrame,
    output_dir: str = "outputs/plots",
) -> None:
    """Per-wafer fail rate histogram."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rates = df.groupby("wafer_id")["label"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rates, bins=30, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Per-wafer fail rate (post-test)")
    ax.set_ylabel("Number of wafers")
    ax.set_title("Distribution of Per-Wafer Fail Rates")
    plt.tight_layout()
    fig.savefig(out / "fail_rate_distribution.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_radial_vs_new_fail(
    df: pd.DataFrame,
    sp_df: pd.DataFrame,
    output_dir: str = "outputs/plots",
) -> None:
    """Bin radial distance and compute empirical new-fail rate."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eligible = df["old_label"] == 0
    radial = sp_df.loc[eligible.values, "sp_radial"].values
    new_fail = df.loc[eligible, "label"].values

    bins = np.linspace(0, 1, 11)
    bin_idx = np.digitize(radial, bins) - 1
    bin_rates = []
    bin_centers = []
    for b in range(len(bins) - 1):
        mask = bin_idx == b
        if mask.sum() > 0:
            bin_rates.append(new_fail[mask].mean())
            bin_centers.append((bins[b] + bins[b+1]) / 2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(bin_centers, bin_rates, width=0.08, color="#E74C3C", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Normalized radial distance from wafer center")
    ax.set_ylabel("New fail rate (old_label=0 dies)")
    ax.set_title("New Fail Rate vs Radial Position")
    plt.tight_layout()
    fig.savefig(out / "radial_vs_fail_rate.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
