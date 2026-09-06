"""
finalize.py
------------
Regenerate results_summary.md from saved models and outputs.
All models, plots, and SHAP outputs were already produced by run_all.py.
This script re-runs just the results writing step.
"""
import os
os.environ["PYTHONUTF8"] = "1"

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.data_loader import load_train, load_test, wafer_level_split, get_feature_cols
from src.spatial_features import compute_spatial_features
from src.block_features import compute_block_features_from_series, get_block_feature_cols
from src.anomaly_features import DieAnomalyDetector, BlockAnomalyDetector
from src.model_a import predict_model_a, get_feature_set_a
from src.model_b import predict_model_b, get_feature_set_b
from src.evaluation import evaluate, comparison_table, ablation_table

OUTPUT_DIR = "outputs"

# ---- Reload everything that was saved ----
print("Loading saved models...")
model_a = joblib.load("outputs/model_a.pkl")
meta_a  = joblib.load("outputs/model_a_meta.pkl")
model_b = joblib.load("outputs/model_b.pkl")
meta_b  = joblib.load("outputs/model_b_meta.pkl")
thresh_a, feat_cols_a = meta_a["threshold"], meta_a["feat_cols"]
thresh_b, feat_cols_b = meta_b["threshold"], meta_b["feat_cols"]

print("Loading data...")
train_df = load_train("input")
test_df  = load_test("input")

# Rebuild features for test set
feat_cols = get_feature_cols(train_df)

print("Computing test spatial features...")
sp_test = compute_spatial_features(test_df)

print("Computing test block features...")
blk_test = compute_block_features_from_series(test_df["block_readings"])
blk_cols = get_block_feature_cols()

# Fit anomaly detectors on train (old_label=0 only)
print("Fitting anomaly detectors on train...")
die_anom = DieAnomalyDetector(n_estimators=200, contamination=0.05)
die_anom.fit(train_df, feat_cols)
die_anom_test = die_anom.score(test_df, feat_cols)

blk_train_full = compute_block_features_from_series(train_df["block_readings"])
blk_anom = BlockAnomalyDetector(n_estimators=200, contamination=0.05)
blk_anom.fit(train_df, blk_train_full, blk_cols)
blk_anom_test = blk_anom.score(blk_test, blk_cols)

# ---- Test predictions ----
print("\n--- Model A Test Evaluation ---")
y_pred_a, y_prob_a = predict_model_a(model_a, test_df, sp_test, die_anom_test, feat_cols_a, thresh_a)
test_metrics_a = evaluate(test_df, y_pred_a, y_prob_a, threshold=thresh_a, verbose=True)

print("\n--- Model B Test Evaluation ---")
y_pred_b, y_prob_b = predict_model_b(model_b, test_df, sp_test, die_anom_test,
                                      blk_test, blk_anom_test, feat_cols_b, thresh_b)
test_metrics_b = evaluate(test_df, y_pred_b, y_prob_b, threshold=thresh_b, verbose=True)

# ---- Ablation results (from log) ----
ablation_metrics = {
    "Die only":              {"pr_auc": 0.0616, "fail_f1": 0.0999, "fail_recall": 0.1651, "fail_precision": 0.0716, "overall_accuracy": 0.9129},
    "Die + Spatial":         {"pr_auc": 0.0587, "fail_f1": 0.0981, "fail_recall": 0.2099, "fail_precision": 0.0640, "overall_accuracy": 0.8870},
    "Die + Block":           {"pr_auc": 0.0660, "fail_f1": 0.1015, "fail_recall": 0.1604, "fail_precision": 0.0742, "overall_accuracy": 0.9168},
    "Die + Spatial + Block": {"pr_auc": 0.0545, "fail_f1": 0.0888, "fail_recall": 0.1922, "fail_precision": 0.0577, "overall_accuracy": 0.8845},
}

# ---- Comparison table ----
print("\n--- Comparison Table ---")
comp = comparison_table({
    "Model A (Die+Spatial+DieAnom)": test_metrics_a,
    "Model B (Model A + Block)":     test_metrics_b,
})
print(comp.to_string(index=False))
comp.to_csv(f"{OUTPUT_DIR}/comparison_table.csv", index=False)

print("\n--- Ablation Table ---")
abl = ablation_table(ablation_metrics)
print(abl.to_string(index=False))
abl.to_csv(f"{OUTPUT_DIR}/ablation_table.csv", index=False)

# ---- Per-die explanations (from log, preserved verbatim) ----
die_explanations = [
    "Die (2,25) on wafer W_F_0014: failure probability 17.9%\n"
    "   SHAP group contributions -> die-params: 0.1503 | spatial: 0.0026 | block: 0.0000 | anomaly: 0.0019\n"
    "   Main reasons:\n"
    "   1) feature_245 = 3.63e+03 (SHAP up 0.0381)\n"
    "   2) feature_3 = 0.133 (SHAP up 0.0381)\n"
    "   3) feature_89 = 974 (SHAP up 0.0149)",

    "Die (3,10) on wafer W_F_0014: failure probability 18.5%\n"
    "   SHAP group contributions -> die-params: 0.1141 | spatial: 0.0022 | block: 0.0000 | anomaly: 0.0620\n"
    "   Main reasons:\n"
    "   1) die_anomaly_score = 0.445 (SHAP up 0.0620)\n"
    "   2) feature_284 = 0.183 (SHAP up 0.0189)\n"
    "   3) feature_100 = 28.9 (SHAP up 0.0120)",

    "Die (3,13) on wafer W_F_0014: failure probability 18.5%\n"
    "   SHAP group contributions -> die-params: 0.1319 | spatial: 0.0017 | block: 0.0000 | anomaly: 0.0430\n"
    "   Main reasons:\n"
    "   1) die_anomaly_score = 0.46 (SHAP up 0.0430)\n"
    "   2) feature_460 = -28.8 (SHAP up 0.0422)\n"
    "   3) feature_342 = 1.86 (SHAP up 0.0103)",

    "Die (3,22) on wafer W_F_0014: failure probability 17.9%\n"
    "   SHAP group contributions -> die-params: 0.1352 | spatial: 0.0019 | block: 0.0000 | anomaly: 0.0044\n"
    "   Main reasons:\n"
    "   1) feature_245 = 3.46e+03 (SHAP up 0.0366)\n"
    "   2) feature_3 = 0.132 (SHAP up 0.0215)\n"
    "   3) feature_89 = 897 (SHAP up 0.0147)",

    "Die (3,29) on wafer W_F_0014: failure probability 18.0%\n"
    "   SHAP group contributions -> die-params: 0.1198 | spatial: 0.0027 | block: 0.0000 | anomaly: 0.0526\n"
    "   Main reasons:\n"
    "   1) die_anomaly_score = 0.442 (SHAP up 0.0526)\n"
    "   2) feature_285 = 392 (SHAP up 0.0328)\n"
    "   3) feature_286 = 28.7 (SHAP up 0.0197)",
]

# ---- Write results_summary.md ----
def write_results_summary(test_metrics_a, test_metrics_b, comp_df, abl_df,
                           die_explanations, train_df):
    out = Path(OUTPUT_DIR)
    lines = []

    lines.append("# Die Yield Prediction -- Results Summary\n")

    lines.append("## Step 1 -- Data Generation Verification\n")
    lines.append("| Item | Config Target | Actual |")
    lines.append("|------|--------------|--------|")
    lines.append("| train rows | ~100K-200K | 173,099 |")
    lines.append("| train wafers | 160 | 160 |")
    lines.append("| test wafers | 40 | 40 |")
    lines.append("| parametric features | 500 | 500 |")
    lines.append("| feature scale ratio | multi-order | 49,033x |")
    n_old = train_df["old_label"].sum()
    n_t = len(train_df)
    el = train_df[train_df["old_label"]==0]
    n_new = el["label"].sum()
    lines.append(f"| old_label=1 fail rate | ~3% | {n_old/n_t*100:.2f}% |")
    lines.append(f"| new fail rate (eligible) | ~2% | {n_new/len(el)*100:.2f}% |")
    lines.append(f"| overall fail rate (label) | ~5% | {train_df['label'].mean()*100:.2f}% |")
    lines.append("\n> **MISMATCH FLAGGED**: overall fail rate is 14.78% vs config target of ~5%.")
    lines.append("> Root cause: the WM-811K labeled-failure wafers selected for this seed have higher")
    lines.append("> internal fail rates (~14%) than the assumed 12% in select_wafers(). The new-fail")
    lines.append("> rate among eligible (old_label=0) dies is 4.23%, consistent with exponential(0.02).\n")

    lines.append("## Step 2 -- EDA Findings\n")
    lines.append("**Spatial predictiveness (new fail rate by zone):**")
    lines.append("- Inner third (r < 0.33): 3.24% new-fail rate")
    lines.append("- Middle third: 4.43%")
    lines.append("- Outer third (r > 0.67): 5.38%")
    lines.append("- **Conclusion: radial position is modestly predictive (+65% fail rate at edge vs center)**\n")
    lines.append("**Old-fail neighborhood density:**")
    lines.append("- Low density (<5%): 4.11% new-fail rate")
    lines.append("- High density (>=5%): 4.35% new-fail rate")
    lines.append("- **Conclusion: neighborhood density adds marginal predictive value in this dataset**\n")
    lines.append("**Block reading signal strength (FLAGGED -- subtle):**")
    lines.append("- Pass dies: block mean=100.00, std~10.5 (smoothing kernel reduces effective std from 15 to 10.5)")
    lines.append("- Fail dies: block mean~100.18 (shift of 4.5 units on 5% of blocks => average shift ~0.225 units)")
    lines.append("- Fixed threshold [55,145] cannot detect this shift; empirical 3*std threshold is needed")
    lines.append("- **This is the key difficulty: block signal is extremely sparse and weak**\n")

    lines.append("## Step 3 -- Leakage Check\n")
    lines.append("- **PASSED**: Spatial features computed identically with and without `label` column present.")
    lines.append("- All spatial features derive exclusively from `old_label`, `die_row`, `die_col`, `wafer_id`.")
    lines.append("- Unit test `test_no_leakage()` in `src/spatial_features.py` confirms this programmatically.\n")

    lines.append("## Step 5 -- Threshold Selection\n")
    lines.append(f"- **Model A**: threshold = {thresh_a:.4f}, chosen by maximizing Fail F1 on validation set")
    lines.append(f"- **Model B**: threshold = {thresh_b:.4f}, same criterion")
    lines.append("- Both models use class weighting (scale_pos_weight ~5.7) instead of oversampling\n")

    lines.append("## Step 6 -- Model Comparison (TEST set, old_label=0 eligible dies only)\n")
    lines.append(comp_df[["Model","pr_auc","fail_f1","fail_recall","fail_precision",
                           "overall_accuracy"]].to_markdown(index=False))
    lines.append("")
    lines.append("**Absolute improvements (Model B vs Model A):**")
    r = comp_df.iloc[1]
    lines.append(f"- PR-AUC:         {r['pr_auc_delta']:+.4f} ({r['pr_auc_pct']:+.2f}%)")
    lines.append(f"- Fail F1:        {r['fail_f1_delta']:+.4f} ({r['fail_f1_pct']:+.2f}%)")
    lines.append(f"- Fail Recall:    {r['fail_recall_delta']:+.4f} ({r['fail_recall_pct']:+.2f}%)")
    lines.append(f"- Fail Precision: {r['fail_precision_delta']:+.4f} ({r['fail_precision_pct']:+.2f}%)")
    lines.append(f"- Accuracy:       {r['overall_accuracy_delta']:+.4f} ({r['overall_accuracy_pct']:+.2f}%)\n")

    lines.append("## Ablation Table (VALIDATION set)\n")
    lines.append(abl_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Step 7 -- Per-die Explanations (Model B, 5 examples)\n")
    for expl in die_explanations:
        lines.append("```")
        lines.append(expl)
        lines.append("```")
        lines.append("")

    lines.append("## Conclusion\n")
    delta_f1  = test_metrics_b["fail_f1"]  - test_metrics_a["fail_f1"]
    delta_auc = test_metrics_b["pr_auc"]   - test_metrics_a["pr_auc"]
    delta_rec = test_metrics_b["fail_recall"] - test_metrics_a["fail_recall"]

    lines.append(f"**Block features change vs Model A (on TEST eligible dies):**")
    lines.append(f"- Fail F1: {delta_f1:+.4f}  ({delta_f1/max(test_metrics_a['fail_f1'],1e-9)*100:+.1f}%)")
    lines.append(f"- PR-AUC:  {delta_auc:+.4f}")
    lines.append(f"- Fail Recall: {delta_rec:+.4f}")
    lines.append("")
    if delta_f1 > 0.005:
        verdict = "Block-level information provides **meaningful additional predictive power** beyond die-level + spatial features."
    elif delta_f1 > 0:
        verdict = ("Block-level information provides only **marginal improvement** (+{:.1f}% Fail F1). "
                   "The extremely sparse, weak block signal (4.5-unit shift on 5% of blocks, "
                   "smoothed to ~0.43 actual sigma) is partially captured by aggregate stats "
                   "(blk_mean, blk_frac_anom_mad) but SHAP contributions are near zero. "
                   "Block features add no meaningfully separable signal beyond die parametric features.").format(
                       delta_f1/max(test_metrics_a['fail_f1'],1e-9)*100)
    else:
        verdict = ("Block-level information **does NOT improve** predictive power. "
                   "The block fail shift of 4.5 units on 5% of blocks is too small relative to "
                   "the noise floor to provide separable aggregate features beyond die parametric data.")

    lines.append(f"> **Verdict**: {verdict}")
    lines.append("")
    lines.append("**Why performance is generally low (Fail F1 ~0.12):**")
    lines.append("1. 65% of fails are 'marginal' (nearly indistinguishable from passes by design)")
    lines.append("2. Block signal is too weak to detect at individual-die level")
    lines.append("3. Spatial predictors are modest (neighborhood density effect: +0.24pp only)")
    lines.append("4. The die anomaly score (Isolation Forest) is the strongest single engineered feature")

    summary = "\n".join(lines)
    p = out / "results_summary.md"
    p.write_text(summary, encoding="utf-8")
    print(f"\nResults summary written: {p}")


write_results_summary(test_metrics_a, test_metrics_b, comp, abl,
                      die_explanations, train_df)
print("\nDone.")
