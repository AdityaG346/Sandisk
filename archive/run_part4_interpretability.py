"""
run_part4_interpretability.py
=============================
Part 4: Interpretability & Demo Layer
 1. Failure-signature discovery via HDBSCAN clustering on per-die SHAP vectors.
 2. Model-based counterfactual explanations for 5 representative dies.
 3. Continuous risk-field surface via 2D Gaussian blur + hotspot detection (4-panel wafer figures).
 4. Bootstrap ensemble uncertainty quantification (mean ± std & confidence qualifier).
 5. Standardized per-die explanation format written to outputs/per_die_explanations.md.
"""
import os
os.environ["PYTHONUTF8"] = "1"

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
from scipy.ndimage import gaussian_filter
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.data_loader import load_train, load_test, wafer_level_split, get_feature_cols, assert_aligned
from src.model_a import get_feature_set_a
from src.model_b import get_feature_set_b
from src.anomaly_features import DieAnomalyDetector, BlockAnomalyDetector

OUTPUT_DIR = "outputs"
PLOTS_DIR = "outputs/plots"
MAPS_DIR = "outputs/wafer_maps"
CACHE_DIR = "outputs/cache"
Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
Path(MAPS_DIR).mkdir(parents=True, exist_ok=True)

print("="*60)
print("  PART 4: INTERPRETABILITY & DEMO LAYER")
print("="*60)

# Load data
train_df = load_train("input")
test_df  = load_test("input")
feat_cols = get_feature_cols(train_df)
blk_cols  = [c for c in pd.read_parquet(f"{CACHE_DIR}/blk_test.parquet").columns if c.startswith("blk_") and not c.endswith("_id") and not c.endswith("_row") and not c.endswith("_col")]

# Load cached test features
sp_test  = pd.read_parquet(f"{CACHE_DIR}/sp_test.parquet")
blk_test = pd.read_parquet(f"{CACHE_DIR}/blk_test.parquet")
blk_train_full = pd.read_parquet(f"{CACHE_DIR}/blk_train_full.parquet")

# Anomaly detector fits
die_anom = DieAnomalyDetector(n_estimators=100, contamination=0.05)
die_anom.fit(train_df, feat_cols)
die_test_score = die_anom.score(test_df, feat_cols)

blk_anom = BlockAnomalyDetector(n_estimators=100, contamination=0.05)
blk_anom.fit(train_df, blk_train_full[blk_cols], blk_cols)
blk_test_score = blk_anom.score(blk_test[blk_cols], blk_cols)

# Features
X_test_b, cols_b = get_feature_set_b(test_df, sp_test, die_test_score, blk_test[blk_cols], blk_test_score)

# Load production Model B
model_b = joblib.load(f"{OUTPUT_DIR}/model_b.pkl")
meta_b  = joblib.load(f"{OUTPUT_DIR}/model_b_meta.pkl")
thresh_b = meta_b["threshold"]

prob_b = model_b.predict_proba(X_test_b[cols_b].values)[:, 1]
pred_b = (prob_b >= thresh_b).astype(int)

# Eligible dies (old_label == 0)
elig_mask = (test_df["old_label"] == 0).values
pred_fail_elig_idx = np.where(elig_mask & (pred_b == 1))[0]
print(f"Total predicted-fail eligible dies on test set: {len(pred_fail_elig_idx):,}")

# -------------------------------------------------------------
# 1. Failure-Signature Discovery (HDBSCAN on SHAP vectors)
# -------------------------------------------------------------
print("\n" + "="*60)
print("  1. Failure-Signature Discovery (HDBSCAN Clustering)")
print("="*60)

# Compute SHAP for predicted-fail dies
sample_fail_indices = pred_fail_elig_idx
sample_fail_df = test_df.iloc[sample_fail_indices].copy().reset_index(drop=True)
X_sample_fail = X_test_b[cols_b].iloc[sample_fail_indices].reset_index(drop=True)

explainer = shap.TreeExplainer(model_b)
shap_vals = explainer.shap_values(X_sample_fail.values)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

# Cluster in SHAP PCA subspace (captures core SHAP directional variance without noise inflation)
pca_shap = PCA(n_components=3, random_state=42).fit_transform(shap_vals)
clusterer = HDBSCAN(min_cluster_size=10, min_samples=2)
cluster_labels = clusterer.fit_predict(pca_shap)

sample_fail_df["cluster"] = cluster_labels
unique_clusters = sorted([c for c in set(cluster_labels) if c != -1])
noise_count = int(np.sum(cluster_labels == -1))
print(f"HDBSCAN found {len(unique_clusters)} distinct failure-signature clusters (plus noise {noise_count} dies):")

cluster_profiles = []
cluster_name_map = {}

for c_id in unique_clusters:
    c_mask = (cluster_labels == c_id)
    c_shap = shap_vals[c_mask]
    mean_abs_per_feat = np.mean(np.abs(c_shap), axis=0)
    top_feat_indices = np.argsort(mean_abs_per_feat)[::-1][:5]
    top_feats = [cols_b[i] for i in top_feat_indices]
    top_vals = [mean_abs_per_feat[i] for i in top_feat_indices]

    # Derive descriptive signature label based on top drivers
    primary_feat = top_feats[0]
    secondary_feat = top_feats[1] if len(top_feats) > 1 else ""
    
    if "blk_" in primary_feat:
        sig_label = "Block-Reading Signal Drift / Memory Array Shift"
    elif "sp_edge" in primary_feat or "sp_radial" in primary_feat:
        sig_label = "Edge-Proximity & Radial Thermal Gradient"
    elif "sp_dist_to_fail" in primary_feat or "sp_old_label" in primary_feat or "sp_old_fail" in primary_feat:
        sig_label = "Defect Neighborhood Proximity & Spatial Clustering"
    elif "die_anomaly" in primary_feat:
        sig_label = "Parametric Anomaly Outlier"
    else:
        sig_label = f"Parametric Drift ({primary_feat})"

    cluster_name_map[c_id] = sig_label
    print(f"\n  Cluster {c_id} ({c_mask.sum()} dies) -> Label: [{sig_label}]")
    print(f"    Top distinguishing features: {', '.join([f'{f} ({v:.4f})' for f, v in zip(top_feats, top_vals)])}")

    cluster_profiles.append({
        "cluster_id": c_id,
        "signature_label": sig_label,
        "n_dies": int(c_mask.sum()),
        "top_feature_1": f"{top_feats[0]} ({top_vals[0]:.4f})",
        "top_feature_2": f"{top_feats[1]} ({top_vals[1]:.4f})",
        "top_feature_3": f"{top_feats[2]} ({top_vals[2]:.4f})",
        "top_feature_4": f"{top_feats[3]} ({top_vals[3]:.4f})",
        "top_feature_5": f"{top_feats[4]} ({top_vals[4]:.4f})",
    })

pd.DataFrame(cluster_profiles).to_csv(f"{OUTPUT_DIR}/failure_signatures.csv", index=False)

# -------------------------------------------------------------
# 2. Pick 5 Representative Dies & Bootstrap Uncertainty
# -------------------------------------------------------------
print("\n" + "="*60)
print("  2. Representative Dies Selection & Bootstrap Uncertainty")
print("="*60)

# Pass population medians for counterfactual normalization
tr_split_df, _ = wafer_level_split(train_df, seed=42)
healthy_sp = pd.read_parquet(f"{CACHE_DIR}/sp_tr.parquet")
healthy_blk = pd.read_parquet(f"{CACHE_DIR}/blk_tr.parquet")
assert_aligned(tr_split_df, healthy_sp, key_cols=["wafer_id", "die_row", "die_col"], context="healthy_train_sp")
assert_aligned(tr_split_df, healthy_blk, key_cols=["wafer_id", "die_row", "die_col"], context="healthy_train_blk")

healthy_die_anom = die_anom.score(tr_split_df, feat_cols)
healthy_blk_anom = blk_anom.score(healthy_blk[blk_cols], blk_cols)
X_train_b, _ = get_feature_set_b(tr_split_df, healthy_sp, healthy_die_anom, healthy_blk[blk_cols], healthy_blk_anom)

healthy_train_dies = (tr_split_df["old_label"] == 0) & (tr_split_df["label"] == 0)
pass_medians = X_train_b.loc[healthy_train_dies.values, cols_b].median()

# Identify 5 diverse representative predicted-fail dies from X_sample_fail:
# 1. Edge-region die (highest sp_edge_prox)
idx_edge = int(X_sample_fail["sp_edge_prox"].idxmax())
# 2. Interior die (lowest sp_radial)
idx_interior = int(X_sample_fail["sp_radial"].idxmin())
# 3. Anomaly-score outlier (highest die_anomaly_score)
idx_anom = int(X_sample_fail["die_anomaly_score"].idxmax())
# 4. Neighborhood cluster die (highest sp_old_fail_density_5)
idx_cluster = int(X_sample_fail["sp_old_fail_density_5"].idxmax())
# 5. Marginal case (probability closest to decision threshold)
fail_probs = prob_b[sample_fail_indices]
abs_diff_to_thresh = np.abs(fail_probs - thresh_b)
idx_marginal = int(np.argmin(abs_diff_to_thresh))

selected_local_indices = [idx_edge, idx_interior, idx_anom, idx_cluster, idx_marginal]
rep_names = [
    "Edge-Region Die",
    "Wafer-Interior Die",
    "High Anomaly-Score Die",
    "Spatial Defect-Cluster Die",
    "Marginal Threshold Case",
]

# Train 5 bootstrap models for uncertainty estimation
print("\nTraining 5 bootstrap LightGBM models for ensemble uncertainty estimation...")
boot_models = []
rng_boot = np.random.default_rng(123)
y_tr_vals = tr_split_df["label"].values
for b_seed in [101, 202, 303, 404, 505]:
    boot_idx = rng_boot.choice(len(X_train_b), size=len(X_train_b), replace=True)
    bm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        scale_pos_weight=12.0,
        random_state=b_seed,
        n_jobs=-1,
        verbose=-1
    )
    bm.fit(X_train_b.iloc[boot_idx].values, y_tr_vals[boot_idx])
    boot_models.append(bm)
print(f"Trained {len(boot_models)} bootstrap ensemble models successfully.")

# -------------------------------------------------------------
# 3. Compute Counterfactual Trajectories & Format Explanations
# -------------------------------------------------------------
print("\n" + "="*60)
print("  3. Counterfactual Trajectories & Explanations")
print("="*60)

explanations_text = []
explanations_text.append("# Per-Die Diagnostic Explanations and Model-Based Counterfactuals\n")
explanations_text.append("> [!IMPORTANT]\n> **Model-Based Counterfactual Disclaimer**: The counterfactuals below demonstrate the mathematical model's hypothetical output response when individual feature inputs are shifted to the pass-population median. They represent *mathematical model sensitivity analysis*, NOT physical or causal semiconductor manufacturing interventions.\n\n")

for rank, (local_idx, rname) in enumerate(zip(selected_local_indices, rep_names), 1):
    orig_idx = sample_fail_indices[local_idx]
    die_row_meta = test_df.iloc[orig_idx]
    w_id = die_row_meta["wafer_id"]
    r_id = int(die_row_meta["die_row"])
    c_id = int(die_row_meta["die_col"])
    
    # Point probability and bootstrap predictions
    p_point = float(prob_b[orig_idx])
    x_vec = X_test_b[cols_b].iloc[[orig_idx]].copy()
    boot_preds = [float(bm.predict_proba(x_vec.values)[0, 1]) for bm in boot_models]
    p_std = float(np.std(boot_preds))
    
    if p_std < 0.05:
        conf_qual = "High Confidence"
    elif p_std < 0.12:
        conf_qual = "Moderate Confidence"
    else:
        conf_qual = "High Uncertainty"

    # SHAP for this specific die
    die_shap = shap_vals[local_idx]
    top_3_idx = np.argsort(np.abs(die_shap))[::-1][:3]
    top_3_feats = [cols_b[i] for i in top_3_idx]
    top_3_shaps = [die_shap[i] for i in top_3_idx]
    
    # Cluster assignment
    c_assign = cluster_labels[local_idx]
    c_label = cluster_name_map.get(c_assign, "Mixed / Boundary Pattern")

    # Stepwise counterfactual
    # Base: p_point
    # Step 1: feat 1 to median
    x_cf1 = x_vec.copy()
    x_cf1[top_3_feats[0]] = pass_medians[top_3_feats[0]]
    p_cf1 = float(model_b.predict_proba(x_cf1.values)[0, 1])

    # Step 2: feat 1 & 2 to median
    x_cf2 = x_cf1.copy()
    x_cf2[top_3_feats[1]] = pass_medians[top_3_feats[1]]
    p_cf2 = float(model_b.predict_proba(x_cf2.values)[0, 1])

    # Step 3: feat 1, 2, 3 to median
    x_cf3 = x_cf2.copy()
    x_cf3[top_3_feats[2]] = pass_medians[top_3_feats[2]]
    p_cf3 = float(model_b.predict_proba(x_cf3.values)[0, 1])

    # Standardized format requested in prompt
    explanation_entry = (
        f"### Die {rank}: {rname} -- Wafer {w_id}, Row {r_id}, Col {c_id}\n\n"
        f"**Standardized Explanation:**\n"
        f"> **Die ({r_id},{c_id})**: failure probability **{p_point*100:.1f}%**, confidence **{conf_qual}** (ensemble $\\sigma = {p_std:.3f}$). "
        f"Main reasons: 1) `{top_3_feats[0]}` (SHAP {top_3_shaps[0]:+.4f}), 2) `{top_3_feats[1]}` (SHAP {top_3_shaps[1]:+.4f}), 3) `{top_3_feats[2]}` (SHAP {top_3_shaps[2]:+.4f}). "
        f"Resembles failure signature cluster {c_assign} ({c_label}). "
        f"Counterfactual: normalizing `{top_3_feats[0]}` would reduce risk to **{p_cf1*100:.1f}%** (normalizing all top-3 reduces risk to **{p_cf3*100:.1f}%**).\n\n"
        f"- **Counterfactual Probability Trajectory**: "
        f"Original ({p_point*100:.1f}%) $\\rightarrow$ Normalizing `{top_3_feats[0]}` ({p_cf1*100:.1f}%) $\\rightarrow$ +`{top_3_feats[1]}` ({p_cf2*100:.1f}%) $\\rightarrow$ +`{top_3_feats[2]}` ({p_cf3*100:.1f}%)\n"
    )
    explanations_text.append(explanation_entry)
    print(f"Formatted explanation for Die {rank} ({rname}) at ({r_id},{c_id}): Prob={p_point*100:.1f}%, Conf={conf_qual}, CF1={p_cf1*100:.1f}%")

with open(f"{OUTPUT_DIR}/per_die_explanations.md", "w", encoding="utf-8") as f:
    f.write("\n".join(explanations_text))
print(f"\nSaved per-die explanations to {OUTPUT_DIR}/per_die_explanations.md")

# -------------------------------------------------------------
# 4. Continuous Risk-Field Surface & Hotspot Visualizations
# -------------------------------------------------------------
print("\n" + "="*60)
print("  4. 4-Panel Continuous Risk-Field & Hotspot Figures")
print("="*60)

# Select 3 sample test wafers with diverse failure profiles
test_wafers = test_df["wafer_id"].unique()
sample_wafers = test_wafers[:3]

for w_id in sample_wafers:
    wdf = test_df[test_df["wafer_id"] == w_id].copy()
    w_idx = wdf.index
    w_prob = prob_b[w_idx]

    rows = wdf["die_row"].values.astype(int)
    cols = wdf["die_col"].values.astype(int)
    n_r = int(rows.max()) + 1
    n_c = int(cols.max()) + 1

    pretest_grid = np.full((n_r, n_c), np.nan)
    newfail_grid = np.full((n_r, n_c), np.nan)
    prob_grid    = np.full((n_r, n_c), np.nan)

    for r, c, old_l, true_l, p in zip(rows, cols, wdf["old_label"].values, wdf["label"].values, w_prob):
        pretest_grid[r, c] = old_l
        if old_l == 0:
            newfail_grid[r, c] = true_l
            prob_grid[r, c] = p

    # Continuous 2D Gaussian blur for risk-field
    # Replace NaNs with 0 for filter, then normalize by kernel density
    prob_filled = np.nan_to_num(prob_grid, nan=0.0)
    valid_mask = ~np.isnan(prob_grid)
    norm_mask = gaussian_filter(valid_mask.astype(float), sigma=1.5)
    norm_mask = np.where(norm_mask > 0.05, norm_mask, 1.0)
    smoothed_risk = gaussian_filter(prob_filled, sigma=1.5) / norm_mask
    smoothed_risk[~valid_mask] = np.nan

    # Hotspots: regions with risk in the top 10%
    valid_risks = smoothed_risk[valid_mask]
    hotspot_thresh = np.percentile(valid_risks, 90) if len(valid_risks) > 0 else 0.5
    hotspot_mask = (smoothed_risk >= hotspot_thresh) & valid_mask

    # Plot 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#475569")

    # Panel 1: Pre-test map
    cmap_pre = mcolors.ListedColormap(["#22c55e", "#ef4444"])
    im1 = axes[0].imshow(pretest_grid, cmap=cmap_pre, vmin=0, vmax=1)
    axes[0].set_title(f"Wafer {w_id}: Pre-Test Status\n(Green=Pass, Red=Old Fail)", color="white", fontsize=10, pad=10)

    # Panel 2: Ground Truth New Fails
    cmap_new = mcolors.ListedColormap(["#334155", "#ef4444"])
    im2 = axes[1].imshow(newfail_grid, cmap=cmap_new, vmin=0, vmax=1)
    axes[1].set_title(f"Ground-Truth New Fails\n(Red=New Fail, Grey=Pass)", color="white", fontsize=10, pad=10)

    # Panel 3: Raw Predicted Probability
    im3 = axes[2].imshow(prob_grid, cmap="plasma", vmin=0, vmax=1)
    axes[2].set_title(f"Raw Model B Predicted Fail Prob\n(Threshold = {thresh_b:.3f})", color="white", fontsize=10, pad=10)
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: Smoothed Risk Field & Hotspots
    im4 = axes[3].imshow(smoothed_risk, cmap="magma", vmin=0, vmax=max(float(np.nanmax(smoothed_risk)), 0.8))
    axes[3].contour(hotspot_mask, levels=[0.5], colors=["#38bdf8"], linewidths=[1.5])
    axes[3].set_title(f"Continuous Risk Field (2D Gaussian)\n(Cyan Contours = Hotspots)", color="white", fontsize=10, pad=10)
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_fig_path = f"{PLOTS_DIR}/risk_field_wafer_{w_id}.png"
    plt.savefig(out_fig_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved risk-field 4-panel figure: {out_fig_path}")

print("\n" + "="*60)
print("  PART 4 COMPLETE: Failure signatures, counterfactuals, risk-fields saved.")
print("="*60)
