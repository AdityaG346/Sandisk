# Die Yield Prediction: Audited Results Summary

> **Data note**: Real WM-811K wafer geometry and pre-test maps combined with synthetic die-level parametric measurements, synthetic sub-die block readings, and controlled new-failure labels.

**PR-AUC convention**: trapezoidal (sklearn `average_precision_score`).  
**Models**: Model A (500 Parametric + 10 Spatial + 1 Die Anomaly = 511 features)  
vs. Model B (Model A + 19 Block Statistics + 1 Block Anomaly = 531 features).  
**Algorithm**: LightGBM (`num_leaves=63`, `learning_rate=0.05`, `scale_pos_weight=12.0`, `n_estimators=500`).  

---

## Test Set Performance (40 Held-Out Wafers, 32,598 Eligible Dies, 1,380 True Fails)

| Metric | Model A | Model B | Delta (B-A) |
|--------|---------|---------|-------------|
| Threshold | 0.5574 | 0.5180 | -0.0393 |
| PR-AUC (trapezoidal) | 0.5025 | 0.5362 | +0.0337 |
| Fail F1 | 0.5207 | 0.5222 | +0.0015 |
| Fail Precision | 97.0% | 88.1% | -8.9% |
| Fail Recall | 35.6% | 37.1% | +1.5% |
| TP / FP / FN | 491/15/889 | 512/69/868 | +21/+54/-21 |
| Overall Accuracy | 97.23% | 97.13% | -0.10% |

### Confusion Matrices

**Model A** (threshold 0.5574):
```
                  Pred Fail  Pred Pass
  Actual Fail          491        889
  Actual Pass           15      31203
```

**Model B** (threshold 0.5180):
```
                  Pred Fail  Pred Pass
  Actual Fail          512        868
  Actual Pass           69      31149
```

---

## Statistical Comparison

Test set, 40 held-out wafers, wafer-cluster bootstrap (1000 iterations):  
- PR-AUC B-A: +0.034 [+0.023, +0.043] (strictly excludes zero)  
- Fail F1 B-A: +0.001 [-0.008, +0.012] (not distinguishable from zero)  

Ablation = 5 repeated wafer splits (validation set only).  
The paired t-test p-value (1.33e-5) and Wilcoxon results are from the validation ablation, NOT the test set.

---

## Budget-Aware Triage (Top-K Screening)

Fraction of all 1,380 true failures captured when screening the top K% of eligible dies:

| Screen Top K% | Model A Capture | Model B Capture | Delta (B − A) |
|--------------|-----------------|-----------------|---------------|
| 2% | 38.1% (526 fails) | 38.8% (535 fails) | +9 fails (+0.7%) |
| 5% | 46.2% (637 fails) | 49.7% (686 fails) | +49 fails (+3.6%) |
| 10% | 55.7% (768 fails) | 60.4% (834 fails) | +66 fails (+4.8%) |

At the same 10% inspection budget, Model B captures 66 more failures than Model A (834 vs. 768 failures, or 60.4% vs. 55.7%).

---

## Probability Calibration (Platt Scaling)

Evaluated on 40 held-out test wafers (32,598 eligible dies, 10 equal-frequency quantile bins):

| Metric | Model A (Raw -> Calibrated) | Model B (Raw -> Calibrated) | Impact |
|--------|-----------------------------|-----------------------------|--------|
| Quantile ECE (10 bins) | 0.0336 -> 0.0084 | 0.0293 -> 0.0085 | -74.9% (A) / -71.0% (B) |
| Brier Score | 0.0287 -> 0.0265 | 0.0280 -> 0.0260 | Improved |
| Test PR-AUC | 0.5025 -> 0.5025 | 0.5362 -> 0.5362 | Identical (monotonic) |
| Spearman Rank Corr | 1.0000 | 1.0000 | Identical ranking |

---

*Numbers audited from frozen cached artifacts. Source: `audit_numbers.py --write`.*