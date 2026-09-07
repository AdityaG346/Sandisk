"""
finalize.py
-----------
Generate the final predictions CSV for hackathon submission.

Workflow:
1. Loads holdout test data (input/test.csv) and production Model B checkpoint.
2. Extracts and aligns parametric, spatial, sub-die block, and anomaly features.
3. Scores every die in input/test.csv (all 39,351 dies, not just eligible ones).
4. Applies competition eligibility logic:
   - old_label == 1: predicted_label = 1 directly (trivially failed pre-test).
   - old_label == 0: predicted_label = 1 if prob >= tuned_threshold else 0.
5. Saves outputs/predictions.csv matching README.md format:
   wafer_id,die_row,die_col,predicted_label
6. Runs sanity checks on row counts, column names/types, and fail rates.
"""

from __future__ import annotations

import os
os.environ["PYTHONUTF8"] = "1"

import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_loader import load_test, load_train, wafer_level_split, get_feature_cols
from src.spatial_features import compute_spatial_features
from src.block_features import compute_block_features_from_series, get_block_feature_cols
from src.anomaly_features import DieAnomalyDetector, BlockAnomalyDetector
from src.model_b import get_feature_set_b


def generate_submission_predictions(
    test_csv_path: str = "input/test.csv",
    output_csv_path: str = "outputs/predictions.csv",
    cache_dir: str = "outputs/cache",
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """
    Score test dies and write outputs/predictions.csv.
    Returns the submission DataFrame.
    """
    print("=" * 65)
    print("  GENERATING FINAL HACKATHON SUBMISSION PREDICTIONS")
    print("=" * 65)

    # 1. Load Model B and its tuned threshold
    model_path = Path(output_dir) / "model_b.pkl"
    meta_path = Path(output_dir) / "model_b_meta.pkl"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing Model B artifacts: {model_path} or {meta_path}")

    print(f"\n[1/6] Loading Model B checkpoint from {model_path}...")
    model_b = joblib.load(model_path)
    meta_b = joblib.load(meta_path)
    threshold_b = float(meta_b["threshold"])
    feat_cols_b = meta_b["feat_cols"]
    print(f"      Model B loaded successfully.")
    print(f"      Feature count: {len(feat_cols_b)}")
    print(f"      Tuned threshold: {threshold_b:.6f}")

    # 2. Load test data
    print(f"\n[2/6] Loading test dataset from {test_csv_path}...")
    test_df = load_test(str(Path(test_csv_path).parent))
    n_total = len(test_df)
    n_wafers = test_df["wafer_id"].nunique()
    n_old_fails = int((test_df["old_label"] == 1).sum())
    n_eligible = int((test_df["old_label"] == 0).sum())
    print(f"      Total test dies: {n_total:,} across {n_wafers} wafers")
    print(f"      Pre-test failed dies (old_label=1): {n_old_fails:,} ({n_old_fails/n_total*100:.2f}%)")
    print(f"      Eligible dies (old_label=0):        {n_eligible:,} ({n_eligible/n_total*100:.2f}%)")

    # 3. Load or compute spatial and block features
    print("\n[3/6] Preparing feature matrices...")
    sp_test_path = Path(cache_dir) / "sp_test.parquet"
    if sp_test_path.exists():
        print(f"      Loading spatial features from {sp_test_path}...")
        sp_test = pd.read_parquet(sp_test_path)
    else:
        print("      Computing test spatial features from scratch...")
        sp_test = compute_spatial_features(test_df)

    blk_test_path = Path(cache_dir) / "blk_test.parquet"
    if blk_test_path.exists():
        print(f"      Loading block features from {blk_test_path}...")
        blk_test = pd.read_parquet(blk_test_path)
    else:
        print("      Computing test block features from raw block_readings...")
        blk_test = compute_block_features_from_series(test_df["block_readings"])

    # Fit anomaly detectors on healthy training dies (seed=42)
    print("      Fitting anomaly detectors on healthy training dies (old_label=0)...")
    train_df = load_train(str(Path(test_csv_path).parent))
    tr_df, _ = wafer_level_split(train_df, val_fraction=0.2, seed=42)
    feat_cols = get_feature_cols(test_df)
    blk_cols = get_block_feature_cols()

    blk_tr_path = Path(cache_dir) / "blk_tr.parquet"
    if blk_tr_path.exists():
        blk_tr = pd.read_parquet(blk_tr_path)
    else:
        blk_tr = compute_block_features_from_series(tr_df["block_readings"])

    die_anom = DieAnomalyDetector(n_estimators=100, contamination=0.05)
    die_anom.fit(tr_df, feat_cols)
    die_anom_test = die_anom.score(test_df, feat_cols)

    blk_anom = BlockAnomalyDetector(n_estimators=100, contamination=0.05)
    blk_anom.fit(tr_df, blk_tr, blk_cols)
    blk_anom_test = blk_anom.score(blk_test, blk_cols)

    # Assemble Model B feature matrix
    X_test_b, _ = get_feature_set_b(test_df, sp_test, die_anom_test, blk_test, blk_anom_test)

    # 4. Predict probabilities for all dies
    print("\n[4/6] Scoring all dies through Model B...")
    y_prob_b = model_b.predict_proba(X_test_b[feat_cols_b].values)[:, 1]

    # 5. Apply eligibility assignment rules
    print("\n[5/6] Applying competition label assignment rules...")
    predicted_label = np.zeros(n_total, dtype=np.int64)

    eligible_mask = (test_df["old_label"].values == 0)
    old_fail_mask = (test_df["old_label"].values == 1)

    # Rule A: Eligible dies use tuned threshold
    predicted_label[eligible_mask] = (y_prob_b[eligible_mask] >= threshold_b).astype(np.int64)

    # Rule B: Pre-test failed dies trivially remain failed
    predicted_label[old_fail_mask] = 1

    # Assemble final submission DataFrame
    pred_df = pd.DataFrame({
        "wafer_id": test_df["wafer_id"].astype(str),
        "die_row": test_df["die_row"].astype(np.int64),
        "die_col": test_df["die_col"].astype(np.int64),
        "predicted_label": predicted_label,
    })

    # Save to disk
    out_path = Path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    print(f"      Saved predictions CSV to: {out_path}")

    # 6. Sanity checks and validation
    print("\n[6/6] Running automated sanity checks on predictions file...")
    # Check A: Row count
    assert len(pred_df) == 39351, f"Expected 39,351 rows, got {len(pred_df)}"
    print(f"      [PASS] Row count matches input exactly: {len(pred_df):,} rows")

    # Check B: Columns and order
    expected_cols = ["wafer_id", "die_row", "die_col", "predicted_label"]
    assert list(pred_df.columns) == expected_cols, f"Columns mismatch: {list(pred_df.columns)} vs {expected_cols}"
    print(f"      [PASS] Column names and order match specification: {expected_cols}")

    # Check C: Null values
    assert not pred_df.isnull().any().any(), "Found null/NaN values in predictions!"
    print("      [PASS] Zero null/NaN values detected")

    # Check D: Pre-test fails all predicted as 1
    assert (pred_df.loc[old_fail_mask, "predicted_label"] == 1).all(), "Some pre-test failed dies have predicted_label != 1!"
    print(f"      [PASS] All {n_old_fails:,} pre-test failed dies (old_label=1) set to predicted_label=1")

    # Check E: Label set
    unique_labels = set(pred_df["predicted_label"].unique())
    assert unique_labels.issubset({0, 1}), f"Unexpected labels found: {unique_labels}"
    print(f"      [PASS] Predicted labels are binary: {sorted(list(unique_labels))}")

    # Check F: Fail rates
    total_fails = int(pred_df["predicted_label"].sum())
    total_fail_rate = total_fails / n_total * 100
    eligible_fails = int(pred_df.loc[eligible_mask, "predicted_label"].sum())
    eligible_fail_rate = eligible_fails / n_eligible * 100

    print("\n" + "=" * 65)
    print("  FINAL PREDICTION SUMMARY & METRICS")
    print("=" * 65)
    print(f"  Model Selected:               Model B (Die + Spatial + Block + Anomaly)")
    print(f"  Decision Threshold:           {threshold_b:.6f}")
    print(f"  Total Dies Evaluated:         {n_total:,}")
    print(f"  Pre-test Failed Dies:         {n_old_fails:,} (100.00% predicted fail)")
    print(f"  Eligible Dies (old_label=0):  {n_eligible:,}")
    print(f"  Eligible Predicted Fails:     {eligible_fails:,} ({eligible_fail_rate:.2f}% of eligible)")
    print(f"  Total Predicted Fails:        {total_fails:,}")
    print(f"  Final Overall Fail Rate:      {total_fail_rate:.4f}% ({total_fails:,} / {n_total:,})")
    print("=" * 65)

    assert 10.0 < total_fail_rate < 30.0, f"Overall fail rate {total_fail_rate:.2f}% is outside plausible range [10%, 30%]!"
    assert 0.5 < eligible_fail_rate < 10.0, f"Eligible fail rate {eligible_fail_rate:.2f}% is outside plausible range [0.5%, 10%]!"
    print("\n[SUCCESS] Final fail rate is verified in a healthy, plausible range!")

    return pred_df


if __name__ == "__main__":
    generate_submission_predictions()
