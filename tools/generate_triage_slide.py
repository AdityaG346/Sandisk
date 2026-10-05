"""
tools/generate_triage_slide.py
==============================
Generates outputs/slides/triage_curve.png showing the budget-aware screening
gains curves for Model A, Model B, and Random baseline on the held-out test set.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, ".")
from src.triage import capture_curve, topk_stats

OUTPUTS = Path("outputs")
SLIDES = OUTPUTS / "slides"
SLIDES.mkdir(parents=True, exist_ok=True)

# Load test data
meta = pd.read_parquet(OUTPUTS / "cache" / "test_meta.parquet")
eligible = meta["old_label"] == 0
y = meta.loc[eligible, "label"].values

pa = np.load(OUTPUTS / "cache" / "test_probs_a.npy")[eligible.values]
pb = np.load(OUTPUTS / "cache" / "test_probs_b.npy")[eligible.values]

k_fracs_a, cap_a, _ = capture_curve(y, pa, n_points=500)
k_fracs_b, cap_b, _ = capture_curve(y, pb, n_points=500)

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200, facecolor="#0f172a")
ax.set_facecolor("#1e293b")

# Plot random baseline
ax.plot([0, 100], [0, 100], "--", color="#64748b", linewidth=1.5, label="Random Screening Baseline (Lift = 1.0x)")

# Plot Model A and Model B
ax.plot(k_fracs_a * 100, cap_a * 100, color="#38bdf8", linewidth=2.2, label="Model A (511 features: Parametric + Spatial + Anom)")
ax.plot(k_fracs_b * 100, cap_b * 100, color="#a855f7", linewidth=2.5, label="Model B (531 features: Model A + Sub-Die Blocks)")

# Focus on realistic screening budgets: 0% to 20%
ax.set_xlim(0, 20)
ax.set_ylim(0, 80)

# Highlight operating points
for k, col in [(0.02, "#f59e0b"), (0.05, "#10b981"), (0.10, "#ec4899")]:
    sa = topk_stats(y, pa, k)["capture_rate"] * 100
    sb = topk_stats(y, pb, k)["capture_rate"] * 100
    ax.axvline(k * 100, color=col, linestyle=":", alpha=0.6, linewidth=1.2)
    ax.scatter([k * 100], [sa], color="#38bdf8", s=40, zorder=5)
    ax.scatter([k * 100], [sb], color="#a855f7", s=40, zorder=5)
    ax.annotate(
        f"Top {int(k*100)}% budget:\nModel A: {sa:.1f}%\nModel B: {sb:.1f}% (+{sb-sa:.1f}%)",
        xy=(k * 100, sb),
        xytext=(k * 100 + 0.6, sb - (5 if k == 0.10 else -2)),
        color=col,
        fontsize=8.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=col, lw=1.0)
    )

ax.set_xlabel("Screening Budget (% of Eligible Dies Inspected)", color="#cbd5e1", fontsize=11, fontweight="medium")
ax.set_ylabel("Defect Capture Rate (% of All True Failures Caught)", color="#cbd5e1", fontsize=11, fontweight="medium")
ax.set_title("Budget-Aware Semiconductor Screening Triage (Test Set, 40 Wafers)", color="#f8fafc", fontsize=12, pad=12, fontweight="bold")

ax.tick_params(colors="#94a3b8", labelsize=9)
for spine in ax.spines.values():
    spine.set_color("#334155")

ax.grid(True, linestyle=":", alpha=0.4, color="#475569")
ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=9, loc="lower right")

plt.tight_layout()
out_fig = SLIDES / "triage_curve.png"
fig.savefig(out_fig, bbox_inches="tight")
plt.close(fig)
print(f"[triage-slide] Saved: {out_fig}")
