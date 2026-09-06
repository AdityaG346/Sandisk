import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
from src.data_loader import load_train, wafer_level_split, get_feature_cols
from src.spatial_features import compute_spatial_features
from src.anomaly_features import DieAnomalyDetector
from src.model_a import get_feature_set_a

print("Loading data...")
df = load_train("input")
feat_cols = get_feature_cols(df)

tr_df, val_df = wafer_level_split(df, val_fraction=0.2, seed=42)
tr_df = tr_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

sp_tr = compute_spatial_features(tr_df)
sp_val = compute_spatial_features(val_df)

die_anom = DieAnomalyDetector(n_estimators=50, contamination=0.05)
die_anom.fit(tr_df, feat_cols)
die_anom_tr = die_anom.score(tr_df, feat_cols)
die_anom_val = die_anom.score(val_df, feat_cols)

X_tr, cols = get_feature_set_a(tr_df, sp_tr, die_anom_tr)
X_val, _ = get_feature_set_a(val_df, sp_val, die_anom_val)
y_tr = tr_df["label"].values
y_val = val_df["label"].values

print("\n--- Test 1: with early stopping (as in retune.py line 150) ---")
m1 = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=12.0, random_state=42, n_jobs=-1, verbose=-1
)
m1.fit(X_tr.values, y_tr, eval_set=[(X_val.values, y_val)], callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(10)])
print("m1 best_iteration_:", m1.best_iteration_)
p1 = m1.predict_proba(X_val.values)[:, 1]
print("p1 min, max, mean:", p1.min(), p1.max(), p1.mean())

print("\n--- Test 2: WITHOUT early stopping, fixed 300 trees (as in grid search line 104) ---")
m2 = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=12.0, random_state=42, n_jobs=-1, verbose=-1
)
m2.fit(X_tr.values, y_tr)
p2 = m2.predict_proba(X_val.values)[:, 1]
print("p2 min, max, mean:", p2.min(), p2.max(), p2.mean())

# Check threshold and Fail F1 on eligible val dies for both:
for name, p in [("m1 (early stopped)", p1), ("m2 (fixed 300)", p2)]:
    elig = val_df["old_label"] == 0
    y_true_e = val_df.loc[elig, "label"].values
    p_e = p[elig.values]
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 200):
        f = f1_score(y_true_e, (p_e >= t).astype(int), pos_label=1, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    print(f"{name}: best val Fail F1 = {best_f1:.5f} at threshold = {best_t:.4f}")
