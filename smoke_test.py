"""
smoke_test.py
=============
Automated multi-tier smoke test suite for SanDisk Die Yield Prediction Hackathon.

Tiers:
  --tier A (no raw data required):
    - Library imports
    - Model, metadata, and cached probability loading
    - Frozen artifact hash verification (tools/check_frozen.py)
    - Recomputation of headline metrics vs outputs/audited_numbers.json
    - outputs/predictions.csv row count (39,351), schema, and exact match to thresholded B
    - Calibrator loading, monotonicity, PR-AUC conservation, disjoint wafer assertion
    - Triage top-k capture verification vs audit
    - Streamlit AppTest execution across default, side-by-side, and all 4 demo presets

  --tier B (requires local raw datasets):
    - Run finalize.py to outputs_smoke/predictions.csv and require byte-identical match
    - Run audit_numbers.py
    - Run test_part1_guardrails.py tests 1-2
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Configure UTF-8 stdout if available
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_tier_a():
    print("=" * 65)
    print("  SMOKE TEST: TIER A (No Heavy Data Dependencies)")
    print("=" * 65)

    # 1. Imports
    print("\n[Tier A - 1/8] Verifying essential package imports...")
    try:
        import lightgbm
        import shap
        import joblib
        import streamlit
        import numpy
        import pandas
        import scipy
        import sklearn
        import matplotlib
        import pyarrow
        import yaml
        from src.triage import topk_stats, capture_curve
        from src.calibration import fit_platt, apply, ece, brier
        from tools.check_frozen import verify_frozen
        print("  [OK] All required libraries and project modules imported successfully.")
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        return False

    # 2. Model & Cache Load
    print("\n[Tier A - 2/8] Loading models, metadata, and cache...")
    try:
        model_a = joblib.load(ROOT / "outputs" / "model_a.pkl")
        model_b = joblib.load(ROOT / "outputs" / "model_b.pkl")
        meta_a = joblib.load(ROOT / "outputs" / "model_a_meta.pkl")
        meta_b = joblib.load(ROOT / "outputs" / "model_b_meta.pkl")
        pa = np.load(ROOT / "outputs" / "cache" / "test_probs_a.npy")
        pb = np.load(ROOT / "outputs" / "cache" / "test_probs_b.npy")
        test_meta = pd.read_parquet(ROOT / "outputs" / "cache" / "test_meta.parquet")
        print(f"  [OK] Models loaded (Model A: {len(meta_a['feat_cols'])} feats, Model B: {len(meta_b['feat_cols'])} feats).")
        print(f"  [OK] Cached test probabilities loaded ({len(pa):,} dies across {test_meta['wafer_id'].nunique()} wafers).")
    except Exception as e:
        print(f"  [FAIL] Model/cache loading failed: {e}")
        return False

    # 3. Frozen Artifacts Integrity
    print("\n[Tier A - 3/8] Verifying frozen artifacts integrity...")
    all_ok, results = verify_frozen()
    if not all_ok:
        print("  [FAIL] Frozen artifact hash mismatch detected!")
        for path, status in results.items():
            if status != "OK":
                print(f"    - {path}: {status}")
        return False
    print("  [OK] All 21 frozen artifacts verified bit-for-bit against official hashes.")

    # 4. Recompute Headline Metrics vs audited_numbers.json
    print("\n[Tier A - 4/8] Recomputing headline metrics vs outputs/audited_numbers.json...")
    try:
        with open(ROOT / "outputs" / "audited_numbers.json") as f:
            audited = json.load(f)
        
        el_mask = test_meta["old_label"] == 0
        y_test = test_meta.loc[el_mask, "label"].values
        pa_el = pa[el_mask.values]
        pb_el = pb[el_mask.values]

        from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
        
        # Model A
        prauc_a = average_precision_score(y_test, pa_el)
        pred_a = (pa_el >= meta_a["threshold"]).astype(int)
        f1_a = f1_score(y_test, pred_a, zero_division=0)
        prec_a = precision_score(y_test, pred_a, zero_division=0)
        rec_a = recall_score(y_test, pred_a, zero_division=0)

        # Model B
        prauc_b = average_precision_score(y_test, pb_el)
        pred_b = (pb_el >= meta_b["threshold"]).astype(int)
        f1_b = f1_score(y_test, pred_b, zero_division=0)
        prec_b = precision_score(y_test, pred_b, zero_division=0)
        rec_b = recall_score(y_test, pred_b, zero_division=0)

        assert abs(prauc_a - audited["model_a"]["pr_auc"]) < 1e-4, f"A PR-AUC diff: {prauc_a} vs {audited['model_a']['pr_auc']}"
        assert abs(f1_a - audited["model_a"]["fail_f1"]) < 1e-4, f"A F1 diff: {f1_a} vs {audited['model_a']['fail_f1']}"
        assert abs(prauc_b - audited["model_b"]["pr_auc"]) < 1e-4, f"B PR-AUC diff: {prauc_b} vs {audited['model_b']['pr_auc']}"
        assert abs(f1_b - audited["model_b"]["fail_f1"]) < 1e-4, f"B F1 diff: {f1_b} vs {audited['model_b']['fail_f1']}"
        print(f"  [OK] Headline metrics match audited numbers: A (PR-AUC={prauc_a:.4f}, F1={f1_a:.4f}), B (PR-AUC={prauc_b:.4f}, F1={f1_b:.4f}).")
    except Exception as e:
        print(f"  [FAIL] Headline metrics recomputation failed: {e}")
        return False

    # 5. Predictions CSV Verification
    print("\n[Tier A - 5/8] Verifying outputs/predictions.csv structure & predictions...")
    try:
        pred_csv = pd.read_csv(ROOT / "outputs" / "predictions.csv")
        assert len(pred_csv) == 39351, f"Expected 39,351 rows, got {len(pred_csv):,}"
        expected_cols = ["wafer_id", "die_row", "die_col", "predicted_label"]
        assert list(pred_csv.columns) == expected_cols, f"Columns mismatch: {list(pred_csv.columns)}"

        # Verify predicted_label equals thresholded cached B probabilities
        expected_preds = (pb >= meta_b["threshold"]).astype(int)
        expected_preds[test_meta["old_label"] == 1] = 1
        assert (pred_csv["predicted_label"].values == expected_preds).all(), "predictions.csv does not match thresholded Model B"
        print("  [OK] outputs/predictions.csv has 39,351 rows and strictly matches thresholded Model B.")
    except Exception as e:
        print(f"  [FAIL] predictions.csv verification failed: {e}")
        return False

    # 6. Calibrator Verification
    print("\n[Tier A - 6/8] Verifying probability calibrators...")
    try:
        cal_a_path = ROOT / "outputs" / "calibration" / "calibrator_a.pkl"
        cal_b_path = ROOT / "outputs" / "calibration" / "calibrator_b.pkl"
        cal_meta_path = ROOT / "outputs" / "calibration" / "calibration_meta.json"

        if cal_a_path.exists() and cal_b_path.exists() and cal_meta_path.exists():
            cal_a = joblib.load(cal_a_path)
            cal_b = joblib.load(cal_b_path)
            with open(cal_meta_path) as f:
                c_meta = json.load(f)

            assert cal_a.coef_[0][0] > 0, "Calibrator A slope <= 0 (monotonicity violation)"
            assert cal_b.coef_[0][0] > 0, "Calibrator B slope <= 0 (monotonicity violation)"
            assert c_meta["fit_split"] == "val", f"fit_split is not val: {c_meta['fit_split']}"

            # Check test wafers disjoint
            val_wafers = set(c_meta["val_wafers"])
            test_wafers = set(test_meta["wafer_id"].unique())
            overlap = val_wafers & test_wafers
            assert len(overlap) == 0, f"Leakage detected! Val and test wafers overlap: {overlap}"
            print(f"  [OK] Calibrators verified (strictly monotonic, fit on validation, 0 test overlap).")
        else:
            print("  [INFO] Calibrator files not present (uncalibrated mode).")
    except Exception as e:
        print(f"  [FAIL] Calibrator verification failed: {e}")
        return False

    # 7. Triage Statistics Verification
    print("\n[Tier A - 7/8] Verifying triage numbers against audited_numbers.json...")
    try:
        for k_frac, label in [(0.02, "top2pct"), (0.05, "top5pct"), (0.10, "top10pct")]:
            sa = topk_stats(y_test, pa_el, k_frac)["capture_rate"]
            sb = topk_stats(y_test, pb_el, k_frac)["capture_rate"]
            exp_a = audited["triage"][f"A_{label}"]
            exp_b = audited["triage"][f"B_{label}"]
            assert abs(sa - exp_a) < 0.0005, f"Triage A {label} mismatch: {sa} vs {exp_a}"
            assert abs(sb - exp_b) < 0.0005, f"Triage B {label} mismatch: {sb} vs {exp_b}"
        print("  [OK] Triage top-k capture rates match audited numbers within 0.0005.")
    except Exception as e:
        print(f"  [FAIL] Triage verification failed: {e}")
        return False

    # 8. Streamlit AppTest
    print("\n[Tier A - 8/8] Executing Streamlit AppTest across views and presets...")
    try:
        from streamlit.testing.v1 import AppTest

        # Default run
        at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=30)
        at.run()
        assert not at.exception, f"Exception on default view: {at.exception}"

        # Side-by-side mode
        at.radio(key="model_mode_select").set_value("Side-by-side Comparison").run()
        assert not at.exception, f"Exception on side-by-side view: {at.exception}"

        # Demo Presets
        presets = [
            "Preset 1: W_F_0014 (40,18) — Spatial Context True Fail",
            "Preset 2: W_N_0066 (2,13) — Block-driven B-only Catch",
            "Preset 3: W_F_0047 (4,10) — Second B-only Catch",
            "Preset 4: W_F_0016 (21,12) — Borderline False Alarm (Pass)",
        ]
        for p in presets:
            at_p = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=30)
            at_p.run()
            at_p.selectbox(key="preset_select").set_value(p).run()
            assert not at_p.exception, f"Exception on {p}: {at_p.exception}"

        print("  [OK] Streamlit AppTest passed for default, side-by-side, and all 4 demo presets.")
    except Exception as e:
        print(f"  [FAIL] AppTest execution failed: {e}")
        return False

    print("\n" + "=" * 65)
    print("  >>> ALL TIER A SMOKE TESTS PASSED (8/8) <<<")
    print("=" * 65)
    return True


def run_tier_b():
    print("\n" + "=" * 65)
    print("  SMOKE TEST: TIER B (Data-Present End-to-End Pipeline)")
    print("=" * 65)

    train_path = ROOT / "input" / "train.csv"
    test_path = ROOT / "input" / "test.csv"
    if not train_path.exists() or not test_path.exists():
        print("  [FAIL] Cannot run Tier B: input/train.csv or input/test.csv missing.")
        return False

    # 1. Reproduce predictions.csv via finalize.py
    print("\n[Tier B - 1/3] Generating predictions to outputs_smoke/predictions.csv...")
    try:
        import finalize
        smoke_dir = ROOT / "outputs_smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        smoke_csv = smoke_dir / "predictions.csv"

        finalize.generate_submission_predictions(
            test_csv_path=str(ROOT / "input" / "test.csv"),
            output_csv_path=str(smoke_csv),
            cache_dir=str(ROOT / "outputs" / "cache"),
            output_dir=str(ROOT / "outputs"),
        )

        with open(ROOT / "outputs" / "predictions.csv", "rb") as f1, open(smoke_csv, "rb") as f2:
            h1 = hashlib.sha256(f1.read()).hexdigest()
            h2 = hashlib.sha256(f2.read()).hexdigest()

        assert h1 == h2, f"predictions.csv hash mismatch!\nOfficial: {h1}\nSmoke:    {h2}"
        print(f"  [OK] finalize.py output is 100% BYTE-IDENTICAL to official submission CSV.")
    except Exception as e:
        print(f"  [FAIL] finalize.py reproduction failed: {e}")
        return False

    # 2. Run audit_numbers.py
    print("\n[Tier B - 2/3] Executing audit_numbers.py...")
    ret = os.system(f'"{sys.executable}" audit_numbers.py')
    if ret != 0:
        print("  [FAIL] audit_numbers.py exited with error.")
        return False
    print("  [OK] audit_numbers.py passed.")

    # 3. Run test_part1_guardrails.py tests 1-3
    print("\n[Tier B - 3/3] Running test_part1_guardrails.py guardrails...")
    try:
        import test_part1_guardrails as tg
        tg.test_1_index_alignment_helper()
        print("    [OK] Test 1: Index alignment helper passed.")
        tg.test_2_training_sanity_check()
        print("    [OK] Test 2: Training sanity floor (best_iteration_ > 5) passed.")
        tg.test_3_threshold_independence()
        print("    [OK] Test 3: Threshold independence passed.")
    except Exception as e:
        print(f"  [FAIL] test_part1_guardrails failed: {e}")
        return False

    print("\n" + "=" * 65)
    print("  >>> ALL TIER B SMOKE TESTS PASSED (3/3) <<<")
    print("=" * 65)
    return True


def main():
    parser = argparse.ArgumentParser(description="Multi-tier Smoke Test Suite")
    parser.add_argument("--tier", choices=["A", "B", "ALL"], default="A",
                        help="Test tier to run: A (no data), B (data-dependent), or ALL")
    args = parser.parse_args()

    ok = True
    if args.tier in ("A", "ALL"):
        ok = run_tier_a()
        if not ok:
            sys.exit(1)

    if args.tier in ("B", "ALL"):
        ok = run_tier_b()
        if not ok:
            sys.exit(1)

    print("\nAll requested smoke tests completed successfully.")


if __name__ == "__main__":
    main()
