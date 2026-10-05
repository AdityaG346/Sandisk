"""
calibrate.py
============
Phase 3 Calibration Script for SanDisk Die Yield Prediction Hackathon.

1. GATE: Rebuild validation split (seed=42), fit anomaly detectors (100 trees),
   verify validation thresholds match model_*_meta.pkl within 1e-6 and val_fail_f1
   matches meta within 1e-4.
2. If GATE passes:
   Fit Platt scaling (logistic regression on logit of raw score) on ELIGIBLE
   VALIDATION DIES ONLY (old_label=0) for Model A and Model B.
3. Evaluate on test set eligible dies:
   - Check if test ECE (10 equal-frequency bins) falls >= 50%
   - Check if Brier score does not worsen (cal_brier <= raw_brier)
   - Check if PR-AUC differs from raw < 1e-4
   - Check if top-k capture is identical
4. KEEP IF conditions pass:
   Save outputs/calibration/{calibrator_a,calibrator_b}.pkl,
   calibration_meta.json, reliability.png, outputs/cache/test_probs_{a,b}_cal.npy.
5. REVERT IF:
   If gate fails or calibration worsens test ECE, delete calibration outputs.
"""

from __future__ import annotations
import os
import sys
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

from src.data_loader import load_train, load_test, wafer_level_split, get_feature_cols, assert_aligned, assert_split_disjoint
from src.block_features import get_block_feature_cols
from src.anomaly_features import DieAnomalyDetector, BlockAnomalyDetector
from src.model_a import get_feature_set_a
from src.model_b import get_feature_set_b
from src.evaluation import select_threshold
from src.calibration import fit_platt, apply, ece, brier
from src.triage import topk_stats


OUTPUT_DIR = Path("outputs")
CALIB_DIR = OUTPUT_DIR / "calibration"
CACHE_DIR = OUTPUT_DIR / "cache"


def run_gate(train_df: pd.DataFrame) -> tuple[bool, pd.DataFrame, np.ndarray, np.ndarray, dict, dict]:
    """Rebuild validation split and verify exact threshold and F1 match."""
    print("\n" + "=" * 65)
    print("  PHASE 3 GATE: Rebuilding Validation Pipeline")
    print("=" * 65)

    feat_cols = get_feature_cols(train_df)
    blk_cols = get_block_feature_cols()

    tr_df, val_df = wafer_level_split(train_df, val_fraction=0.2, seed=42)
    tr_df = tr_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    sp_val = pd.read_parquet(CACHE_DIR / "sp_val.parquet")
    blk_tr = pd.read_parquet(CACHE_DIR / "blk_tr.parquet")
    blk_val = pd.read_parquet(CACHE_DIR / "blk_val.parquet")

    assert_aligned(val_df, sp_val, context="sp_val alignment")
    assert_aligned(val_df, blk_val, context="blk_val alignment")

    print("  Fitting 100-tree anomaly detectors on healthy training dies...")
    die_anom = DieAnomalyDetector(n_estimators=100, contamination=0.05)
    die_anom.fit(tr_df, feat_cols)
    die_anom_val = die_anom.score(val_df, feat_cols)

    blk_anom = BlockAnomalyDetector(n_estimators=100, contamination=0.05)
    blk_anom.fit(tr_df, blk_tr, blk_cols)
    blk_anom_val = blk_anom.score(blk_val, blk_cols)

    X_val_a, feat_cols_a = get_feature_set_a(val_df, sp_val, die_anom_val)
    X_val_b, feat_cols_b = get_feature_set_b(val_df, sp_val, die_anom_val, blk_val, blk_anom_val)

    model_a = joblib.load(OUTPUT_DIR / "model_a.pkl")
    model_b = joblib.load(OUTPUT_DIR / "model_b.pkl")
    meta_a = joblib.load(OUTPUT_DIR / "model_a_meta.pkl")
    meta_b = joblib.load(OUTPUT_DIR / "model_b_meta.pkl")

    p_val_a = model_a.predict_proba(X_val_a[meta_a["feat_cols"]].values)[:, 1]
    p_val_b = model_b.predict_proba(X_val_b[meta_b["feat_cols"]].values)[:, 1]

    thresh_a, val_f1_a = select_threshold(val_df, p_val_a, n_thresholds=300)
    thresh_b, val_f1_b = select_threshold(val_df, p_val_b, n_thresholds=300)

    print(f"  Model A Threshold: {thresh_a:.6f} (expected {meta_a['threshold']:.6f})")
    print(f"  Model A Val F1:    {val_f1_a:.6f} (expected {meta_a['val_fail_f1']:.6f})")
    print(f"  Model B Threshold: {thresh_b:.6f} (expected {meta_b['threshold']:.6f})")
    print(f"  Model B Val F1:    {val_f1_b:.6f} (expected {meta_b['val_fail_f1']:.6f})")

    diff_th_a = abs(thresh_a - meta_a["threshold"])
    diff_f1_a = abs(val_f1_a - meta_a["val_fail_f1"])
    diff_th_b = abs(thresh_b - meta_b["threshold"])
    diff_f1_b = abs(val_f1_b - meta_b["val_fail_f1"])

    gate_ok = (diff_th_a < 1e-6 and diff_f1_a < 1e-4 and
               diff_th_b < 1e-6 and diff_f1_b < 1e-4)

    if gate_ok:
        print("  >>> GATE CHECK: PASSED (Exact match with frozen metadata) <<<")
    else:
        print("  >>> GATE CHECK: FAILED <<<")

    return gate_ok, val_df, p_val_a, p_val_b, meta_a, meta_b


def plot_reliability(
    y_test: np.ndarray,
    p_raw_a: np.ndarray,
    p_cal_a: np.ndarray,
    p_raw_b: np.ndarray,
    p_cal_b: np.ndarray,
    out_path: Path
) -> None:
    """Plot reliability diagrams (calibration curves) for Model A and B on test set."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    for ax, p_raw, p_cal, name in [
        (axes[0], p_raw_a, p_cal_a, "Model A"),
        (axes[1], p_raw_b, p_cal_b, "Model B")
    ]:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

        # 10 equal-frequency quantile bins for reliability curve
        for p, label, color in [
            (p_raw, f"Raw (ECE={ece(y_test, p_raw):.4f})", "#ef4444"),
            (p_cal, f"Calibrated (ECE={ece(y_test, p_cal):.4f})", "#3b82f6")
        ]:
            quantiles = np.linspace(0, 1, 11)
            edges = np.unique(np.percentile(p, quantiles * 100))
            if len(edges) > 1:
                b_ids = np.digitize(p, edges[1:-1])
                mean_conf = []
                mean_acc = []
                for b in range(len(edges) - 1):
                    m = (b_ids == b)
                    if m.sum() > 0:
                        mean_conf.append(np.mean(p[m]))
                        mean_acc.append(np.mean(y_test[m]))
                ax.plot(mean_conf, mean_acc, "s-", color=color, linewidth=1.5, label=label)

        ax.set_xlabel("Mean Predicted Confidence", fontsize=10)
        ax.set_ylabel("Empirical Failure Rate", fontsize=10)
        ax.set_title(f"{name} — Test Set Reliability Diagram (10-bin Quantile)", fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved reliability diagram: {out_path}")


def main():
    train_path = Path("input/train.csv")
    if not train_path.exists():
        print("[calibration] DATA_PRESENT=False. Skipping calibration to fallback.")
        return

    train_df = load_train("input")
    gate_ok, val_df, p_val_a, p_val_b, meta_a, meta_b = run_gate(train_df)

    if not gate_ok:
        print("[calibration] Gate failed. Reverting / aborting calibration.")
        if CALIB_DIR.exists():
            shutil.rmtree(CALIB_DIR)
        return

    print("\n" + "=" * 65)
    print("  FITTING PLATT SCALING ON ELIGIBLE VALIDATION DIES ONLY")
    print("=" * 65)

    val_eligible = val_df["old_label"] == 0
    y_val_el = val_df.loc[val_eligible, "label"].values
    p_val_a_el = p_val_a[val_eligible.values]
    p_val_b_el = p_val_b[val_eligible.values]

    cal_a = fit_platt(y_val_el, p_val_a_el)
    cal_b = fit_platt(y_val_el, p_val_b_el)

    print(f"  Calibrator A: intercept={cal_a.intercept_[0]:.4f}, coef={cal_a.coef_[0][0]:.4f}")
    print(f"  Calibrator B: intercept={cal_b.intercept_[0]:.4f}, coef={cal_b.coef_[0][0]:.4f}")

    # Load test set
    test_meta = pd.read_parquet(CACHE_DIR / "test_meta.parquet")
    assert_split_disjoint(val_df, test_meta)
    
    test_eligible = test_meta["old_label"] == 0
    y_test_el = test_meta.loc[test_eligible, "label"].values

    p_test_a_all = np.load(CACHE_DIR / "test_probs_a.npy")
    p_test_b_all = np.load(CACHE_DIR / "test_probs_b.npy")

    p_test_a_el = p_test_a_all[test_eligible.values]
    p_test_b_el = p_test_b_all[test_eligible.values]

    # Apply calibration
    p_test_a_cal_all = apply(cal_a, p_test_a_all)
    p_test_b_cal_all = apply(cal_b, p_test_b_all)

    p_test_a_cal_el = p_test_a_cal_all[test_eligible.values]
    p_test_b_cal_el = p_test_b_cal_all[test_eligible.values]

    # Metrics computation
    raw_ece_a = ece(y_test_el, p_test_a_el)
    cal_ece_a = ece(y_test_el, p_test_a_cal_el)
    raw_brier_a = brier(y_test_el, p_test_a_el)
    cal_brier_a = brier(y_test_el, p_test_a_cal_el)

    raw_ece_b = ece(y_test_el, p_test_b_el)
    cal_ece_b = ece(y_test_el, p_test_b_cal_el)
    raw_brier_b = brier(y_test_el, p_test_b_el)
    cal_brier_b = brier(y_test_el, p_test_b_cal_el)

    raw_prauc_a = average_precision_score(y_test_el, p_test_a_el)
    cal_prauc_a = average_precision_score(y_test_el, p_test_a_cal_el)
    raw_prauc_b = average_precision_score(y_test_el, p_test_b_el)
    cal_prauc_b = average_precision_score(y_test_el, p_test_b_cal_el)

    print("\n  Test Calibration Evaluation (10-bin quantile ECE & Brier score):")
    print(f"  Model A: Raw ECE={raw_ece_a:.4f} -> Cal ECE={cal_ece_a:.4f} (drop: {(raw_ece_a-cal_ece_a)/raw_ece_a*100:.1f}%)")
    print(f"           Raw Brier={raw_brier_a:.4f} -> Cal Brier={cal_brier_a:.4f}")
    print(f"           Raw PR-AUC={raw_prauc_a:.5f} -> Cal PR-AUC={cal_prauc_a:.5f} (diff: {abs(cal_prauc_a - raw_prauc_a):.2e})")
    print(f"  Model B: Raw ECE={raw_ece_b:.4f} -> Cal ECE={cal_ece_b:.4f} (drop: {(raw_ece_b-cal_ece_b)/raw_ece_b*100:.1f}%)")
    print(f"           Raw Brier={raw_brier_b:.4f} -> Cal Brier={cal_brier_b:.4f}")
    print(f"           Raw PR-AUC={raw_prauc_b:.5f} -> Cal PR-AUC={cal_prauc_b:.5f} (diff: {abs(cal_prauc_b - raw_prauc_b):.2e})")

    # Check top-k capture
    topk_match = True
    for k in [0.02, 0.05, 0.10]:
        t_raw_a = topk_stats(y_test_el, p_test_a_el, k)["capture_rate"]
        t_cal_a = topk_stats(y_test_el, p_test_a_cal_el, k)["capture_rate"]
        t_raw_b = topk_stats(y_test_el, p_test_b_el, k)["capture_rate"]
        t_cal_b = topk_stats(y_test_el, p_test_b_cal_el, k)["capture_rate"]
        if abs(t_raw_a - t_cal_a) > 1e-6 or abs(t_raw_b - t_cal_b) > 1e-6:
            topk_match = False
            print(f"  Top-k mismatch at k={k}")

    # Check KEEP IF criteria
    ece_drop_a_pct = (raw_ece_a - cal_ece_a) / raw_ece_a
    ece_drop_b_pct = (raw_ece_b - cal_ece_b) / raw_ece_b
    brier_ok_a = (cal_brier_a <= raw_brier_a + 1e-5)
    brier_ok_b = (cal_brier_b <= raw_brier_b + 1e-5)
    prauc_diff_ok = (abs(cal_prauc_a - raw_prauc_a) < 1e-4 and abs(cal_prauc_b - raw_prauc_b) < 1e-4)

    keep_condition = (ece_drop_a_pct >= 0.50 and ece_drop_b_pct >= 0.50 and
                      brier_ok_a and brier_ok_b and prauc_diff_ok and topk_match)

    print(f"\n  KEEP IF criteria met: {keep_condition}")
    print(f"    - ECE drop >= 50%: A={ece_drop_a_pct*100:.1f}%, B={ece_drop_b_pct*100:.1f}% -> {ece_drop_a_pct >= 0.50 and ece_drop_b_pct >= 0.50}")
    print(f"    - Brier not worsened: A={brier_ok_a}, B={brier_ok_b}")
    print(f"    - PR-AUC diff < 1e-4: {prauc_diff_ok}")
    print(f"    - Top-k capture identical: {topk_match}")

    if keep_condition:
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(cal_a, CALIB_DIR / "calibrator_a.pkl")
        joblib.dump(cal_b, CALIB_DIR / "calibrator_b.pkl")
        np.save(CACHE_DIR / "test_probs_a_cal.npy", p_test_a_cal_all)
        np.save(CACHE_DIR / "test_probs_b_cal.npy", p_test_b_cal_all)

        thresh_a_cal = float(apply(cal_a, np.array([meta_a["threshold"]]))[0])
        thresh_b_cal = float(apply(cal_b, np.array([meta_b["threshold"]]))[0])

        meta_json = {
            "fit_split": "val",
            "val_wafers": sorted(list(val_df["wafer_id"].unique())),
            "disjoint_assertion": True,
            "mapped_threshold_a": thresh_a_cal,
            "mapped_threshold_b": thresh_b_cal,
            "raw_test_ece_a": raw_ece_a,
            "cal_test_ece_a": cal_ece_a,
            "raw_test_brier_a": raw_brier_a,
            "cal_test_brier_a": cal_brier_a,
            "raw_test_ece_b": raw_ece_b,
            "cal_test_ece_b": cal_ece_b,
            "raw_test_brier_b": raw_brier_b,
            "cal_test_brier_b": cal_brier_b,
            "ece_drop_a_pct": ece_drop_a_pct,
            "ece_drop_b_pct": ece_drop_b_pct,
        }
        with open(CALIB_DIR / "calibration_meta.json", "w") as f:
            json.dump(meta_json, f, indent=2)

        plot_reliability(
            y_test_el, p_test_a_el, p_test_a_cal_el,
            p_test_b_el, p_test_b_cal_el,
            CALIB_DIR / "reliability.png"
        )
        print("  [calibration] Successfully saved all calibration artifacts!")
    else:
        print("  [calibration] REVERT IF triggered: calibration failed criteria.")
        if CALIB_DIR.exists():
            shutil.rmtree(CALIB_DIR)
        for p in [CACHE_DIR / "test_probs_a_cal.npy", CACHE_DIR / "test_probs_b_cal.npy"]:
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    main()
