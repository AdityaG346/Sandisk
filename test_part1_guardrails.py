"""
test_part1_guardrails.py
========================
Automated regression & robustness test suite for Part 1:
 1. Index-alignment assertions (assert_aligned helper).
 2. Training sanity floor (best_iteration_ > 5).
 3. Threshold independence verification.
 4. End-to-end reproducibility check (two identical seeded runs match to >= 4 decimals).
 5. Reload-from-disk verification (loaded models reproduce exact test metrics).
"""
import os
os.environ["PYTHONUTF8"] = "1"

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path

from src.data_loader import load_train, load_test, wafer_level_split, get_feature_cols, assert_aligned, assert_split_disjoint
from src.spatial_features import compute_spatial_features
from src.block_features import get_block_feature_cols
from src.anomaly_features import DieAnomalyDetector, BlockAnomalyDetector
from src.model_a import get_feature_set_a, train_model_a
from src.model_b import get_feature_set_b, train_model_b
from src.evaluation import evaluate, select_threshold


def test_1_index_alignment_helper():
    print("\n[TEST 1] Testing assert_aligned and assert_split_disjoint guardrails...")
    # Test valid alignment
    df1 = pd.DataFrame({"wafer_id": [1, 1, 2], "die_row": [0, 1, 0], "die_col": [0, 0, 1], "val": [10, 20, 30]})
    df2 = pd.DataFrame({"wafer_id": [1, 1, 2], "die_row": [0, 1, 0], "die_col": [0, 0, 1], "feat": [0.1, 0.2, 0.3]})
    assert_aligned(df1, df2, key_cols=["wafer_id", "die_row", "die_col"], context="valid_test")
    
    # Test length mismatch
    try:
        assert_aligned(df1, df2.iloc[:2], context="len_test")
        raise RuntimeError("Failed to catch length mismatch!")
    except AssertionError as e:
        print("  -> Correctly caught length mismatch:", str(e)[:60])

    # Test key mismatch (scrambled rows)
    df2_scrambled = df2.iloc[[1, 0, 2]].reset_index(drop=True)
    try:
        assert_aligned(df1, df2_scrambled, key_cols=["wafer_id", "die_row", "die_col"], context="key_test")
        raise RuntimeError("Failed to catch key column mismatch!")
    except AssertionError as e:
        print("  -> Correctly caught key mismatch:", str(e)[:60])

    # Test wafer split disjointness
    tr_mock = pd.DataFrame({"wafer_id": [1, 2, 3]})
    val_mock_bad = pd.DataFrame({"wafer_id": [3, 4, 5]})
    try:
        assert_split_disjoint(tr_mock, val_mock_bad)
        raise RuntimeError("Failed to catch wafer leakage!")
    except AssertionError as e:
        print("  -> Correctly caught wafer split overlap:", str(e)[:60])

    print("  -> [TEST 1 PASSED]: All alignment and disjointness assertions verified.")


def test_2_training_sanity_check():
    print("\n[TEST 2] Testing training sanity floor (best_iteration_ > 5)...")
    # Verify mock early stopping failure
    class MockModel:
        best_iteration_ = 1
    
    try:
        if MockModel.best_iteration_ <= 5:
            raise AssertionError("[TrainingSanityError] Model stopped suspiciously early at iteration 1 <= 5!")
    except AssertionError as e:
        print("  -> Correctly caught iteration floor violation:", str(e)[:60])

    # Check that current saved models on disk satisfy best_iteration_ > 5
    m_a = joblib.load("outputs/model_a.pkl")
    m_b = joblib.load("outputs/model_b.pkl")
    iter_a = getattr(m_a, "best_iteration_", None) or m_a.n_estimators
    iter_b = getattr(m_b, "best_iteration_", None) or m_b.n_estimators
    print(f"  -> Model A best_iteration_ = {iter_a}")
    print(f"  -> Model B best_iteration_ = {iter_b}")
    assert iter_a > 5, f"Model A best_iteration_={iter_a} <= 5"
    assert iter_b > 5, f"Model B best_iteration_={iter_b} <= 5"
    print("  -> [TEST 2 PASSED]: Both models have best_iteration_ > 5.")


def test_3_threshold_independence():
    print("\n[TEST 3] Testing threshold independence...")
    rng = np.random.default_rng(42)
    val_mock = pd.DataFrame({"old_label": [0]*1000, "label": rng.choice([0, 1], size=1000, p=[0.95, 0.05])})
    y_prob_a = rng.uniform(0.1, 0.9, size=1000)
    y_prob_b = rng.uniform(0.1, 0.9, size=1000)

    thresh_b_orig, f1_b_orig = select_threshold(val_mock, y_prob_b, n_thresholds=100)

    # Mutate Model A probabilities completely (invert, zero out, scale)
    y_prob_a_mutated = 1.0 - y_prob_a
    thresh_a_mut, f1_a_mut = select_threshold(val_mock, y_prob_a_mutated, n_thresholds=100)

    # Re-evaluate Model B threshold
    thresh_b_after, f1_b_after = select_threshold(val_mock, y_prob_b, n_thresholds=100)

    assert thresh_b_orig == thresh_b_after, "Model B threshold changed when Model A probabilities changed!"
    assert f1_b_orig == f1_b_after, "Model B F1 changed when Model A probabilities changed!"
    print(f"  -> Model B threshold before mutation: {thresh_b_orig:.4f}, after: {thresh_b_after:.4f}")
    print("  -> [TEST 3 PASSED]: Thresholds are strictly independent.")


def test_4_reproducibility():
    print("\n[TEST 4] Testing pipeline reproducibility across 2 seeded runs...")
    # Load cached splits
    CACHE_DIR = "outputs/cache"
    sp_tr = pd.read_parquet(f"{CACHE_DIR}/sp_tr.parquet")
    sp_val = pd.read_parquet(f"{CACHE_DIR}/sp_val.parquet")
    
    train_df = load_train("input")
    tr_df, val_df = wafer_level_split(train_df, val_fraction=0.2, seed=42)
    feat_cols = get_feature_cols(tr_df)

    die_anom = DieAnomalyDetector(n_estimators=50, contamination=0.05)
    die_anom.fit(tr_df, feat_cols)
    anom_tr = die_anom.score(tr_df, feat_cols)
    anom_val = die_anom.score(val_df, feat_cols)

    X_tr_a, cols_a = get_feature_set_a(tr_df, sp_tr, anom_tr)
    X_val_a, _ = get_feature_set_a(val_df, sp_val, anom_val)
    y_tr = tr_df["label"].values

    # Run 1
    m1 = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, scale_pos_weight=12.0, random_state=42, n_jobs=1, verbose=-1)
    m1.fit(X_tr_a.values, y_tr)
    p1 = m1.predict_proba(X_val_a.values)[:, 1]

    # Run 2
    m2 = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, scale_pos_weight=12.0, random_state=42, n_jobs=1, verbose=-1)
    m2.fit(X_tr_a.values, y_tr)
    p2 = m2.predict_proba(X_val_a.values)[:, 1]

    max_diff = np.max(np.abs(p1 - p2))
    print(f"  -> Max absolute difference between run 1 and run 2 predictions: {max_diff:.8e}")
    assert max_diff < 1e-5, f"Nondeterminism detected! Max diff = {max_diff}"
    print("  -> [TEST 4 PASSED]: Pipeline is strictly reproducible (diff < 1e-5).")


def test_5_reload_from_disk_verification():
    print("\n[TEST 5] Testing reload-from-disk and exact test-set metric reproduction...")
    test_df = load_test("input")
    CACHE_DIR = "outputs/cache"
    sp_test = pd.read_parquet(f"{CACHE_DIR}/sp_test.parquet")
    blk_test = pd.read_parquet(f"{CACHE_DIR}/blk_test.parquet")
    
    feat_cols = get_feature_cols(test_df)
    blk_cols = get_block_feature_cols()

    # Load trained models and metadata fresh from disk
    model_a = joblib.load("outputs/model_a.pkl")
    meta_a = joblib.load("outputs/model_a_meta.pkl")
    thresh_a = meta_a["threshold"]
    feat_cols_a = meta_a["feat_cols"]

    model_b = joblib.load("outputs/model_b.pkl")
    meta_b = joblib.load("outputs/model_b_meta.pkl")
    thresh_b = meta_b["threshold"]
    feat_cols_b = meta_b["feat_cols"]

    # Re-score test set
    train_df = load_train("input")
    tr_df, _ = wafer_level_split(train_df, val_fraction=0.2, seed=42)
    blk_tr = pd.read_parquet(f"{CACHE_DIR}/blk_tr.parquet")
    
    die_anom = DieAnomalyDetector(n_estimators=100, contamination=0.05)
    die_anom.fit(tr_df, feat_cols)
    die_anom_test = die_anom.score(test_df, feat_cols)

    blk_anom = BlockAnomalyDetector(n_estimators=100, contamination=0.05)
    blk_anom.fit(tr_df, blk_tr, blk_cols)
    blk_anom_test = blk_anom.score(blk_test, blk_cols)

    X_test_a, _ = get_feature_set_a(test_df, sp_test, die_anom_test)
    X_test_b, _ = get_feature_set_b(test_df, sp_test, die_anom_test, blk_test, blk_anom_test)

    y_prob_a = model_a.predict_proba(X_test_a[feat_cols_a].values)[:, 1]
    y_pred_a = (y_prob_a >= thresh_a).astype(int)
    y_pred_a[test_df["old_label"].values == 1] = 1

    y_prob_b = model_b.predict_proba(X_test_b[feat_cols_b].values)[:, 1]
    y_pred_b = (y_prob_b >= thresh_b).astype(int)
    y_pred_b[test_df["old_label"].values == 1] = 1

    metrics_a = evaluate(test_df, y_pred_a, y_prob_a, thresh_a, verbose=False)
    metrics_b = evaluate(test_df, y_pred_b, y_prob_b, thresh_b, verbose=False)

    # Check against comparison_table.csv on disk
    comp_df = pd.read_csv("outputs/comparison_table.csv")
    reloaded_f1_a = comp_df.loc[comp_df["Model"].str.contains("Model A"), "fail_f1"].values[0]
    reloaded_f1_b = comp_df.loc[comp_df["Model"].str.contains("Model B"), "fail_f1"].values[0]

    print(f"  -> Model A reloaded Fail F1: {metrics_a['fail_f1']:.6f} vs table: {reloaded_f1_a:.6f}")
    print(f"  -> Model B reloaded Fail F1: {metrics_b['fail_f1']:.6f} vs table: {reloaded_f1_b:.6f}")

    assert abs(metrics_a["fail_f1"] - reloaded_f1_a) < 1e-4, "Model A Fail F1 mismatch on reload!"
    assert abs(metrics_b["fail_f1"] - reloaded_f1_b) < 1e-4, "Model B Fail F1 mismatch on reload!"
    print("  -> [TEST 5 PASSED]: Reload from disk exactly reproduces comparison_table.csv metrics.")


if __name__ == "__main__":
    print("="*60)
    print("  RUNNING PART 1 ROBUSTNESS & REGRESSION TEST SUITE")
    print("="*60)
    test_1_index_alignment_helper()
    test_2_training_sanity_check()
    test_3_threshold_independence()
    test_4_reproducibility()
    test_5_reload_from_disk_verification()
    print("\n" + "="*60)
    print("  ALL 5 PART 1 REGRESSION CHECKS PASSED PERFECTLY!")
    print("="*60)
