# Semiconductor Die Yield Prediction (Model A vs. Model B)

## Project Overview
This repository contains a complete, production-ready machine learning solution for the semiconductor die-yield prediction hackathon. The objective is to identify which fabricated silicon dies that passed pre-test inspection (`old_label = 0`) will fail post-stress burn-in testing (`label = 1`), preventing defective chips from advancing to expensive packaging. The repository includes end-to-end data ingestion pipelines, leakage-free spatial and block feature engineering, trained gradient-boosted tree models (Model A and Model B), automated regression guardrails, multi-seed statistical significance evaluations, and a full diagnostic interpretability layer.

---

## What Was Built
- **Model A (Baseline + Spatial + Anomaly)**: Combines 500 die-level parametric electrical test measurements, 24 leakage-safe spatial wafer context features (radial distance, edge proximity, neighborhood defect clustering, and zone yields derived strictly from pre-test data), and an Isolation Forest anomaly score fit exclusively on healthy dies.
- **Model B (Full Diagnostic Model)**: Extends Model A with summary statistics, tail distributions, and anomaly metrics derived from ~2,000 sub-die high-frequency block readings per die, paired with a dedicated block-level anomaly detector.
- **Comparison & Interpretability Layer**: Features five-seed validation ablations, paired hypothesis tests (t-test and Wilcoxon), a 1,000-sample test set bootstrap, exact TreeSHAP attribution, HDBSCAN failure-signature discovery, model-based counterfactual trajectories, and four-panel continuous 2D Gaussian wafer risk fields with hotspot detection.
- **Plain-Language Documentation**: For a non-technical overview written in plain prose for teammates and stakeholders, see [docs/project_explanation.md](docs/project_explanation.md).

---

## Key Results
- **Headline Test Performance (32,598 eligible dies, 1,380 true fails)**: Model A achieves Fail F1 = **0.5209**, PR-AUC = **0.5014**, Fail Precision = **96.66%**, Fail Recall = **35.65%** (threshold 0.5574). Model B achieves Fail F1 = **0.5217**, PR-AUC = **0.5357**, Fail Precision = **88.26%**, Fail Recall = **37.03%** (threshold 0.5180). Both models prioritize high precision to avoid costly false silicon scrap.
- **Statistically Significant PR-AUC Gain (+0.0343 to +0.0359)**: Adding sub-die block features yields a verified, statistically significant improvement in continuous probability ranking ($p = 1.33 \times 10^{-5}$ paired t-test; 95% test bootstrap confidence interval $[+0.0223, +0.0466]$ strictly excludes zero).
- **Fail F1 Invariance (+0.0008 to +0.0090)**: At the discrete binary decision threshold, block features do not meaningfully improve Fail F1 (95% bootstrap CI $[−0.0087, +0.0108]$ overlaps zero) because 65% of new failures are marginal defects whose sub-die electrical signals overlap heavily with passing dies.
- **Block Feature Attribution & Full Audit**: Sub-die block readings contribute **5.19%** of total TreeSHAP attribution mass (led by `blk_mean` at 0.3806), explaining why block data enhances risk ranking even with subtle signal. For comprehensive audit details, bug autopsies, and class-weight tuning curves, see [outputs/results_summary.md](outputs/results_summary.md).

---

## Live Interactive Dashboard

An interactive Streamlit application is available for exploring the results, inspecting continuous 2D Gaussian wafer risk fields, and examining per-die SHAP diagnostics:

- **Shareable Cloud Link**: `https://die-yield-prediction.streamlit.app/`
- **Local Launch**: `streamlit run dashboard/app.py`
- **Read-Only Architecture**: The dashboard operates strictly as a fast, read-only presentation layer over the already-audited models and cached artifacts — no retraining or threshold re-tuning occurs inside the dashboard.

### 5-Line Live Presentation Demo Script
1. **Overview & Wafer Risk Field**: Open the dashboard with default wafer `W_F_0014` selected, observe the 4-panel view comparing the binary pre-test map to the continuous 2D Gaussian risk field, and highlight the prominent cyan hotspot contour encircling the outer defect cluster.
2. **Side-by-Side Model Comparison**: Switch the sidebar toggle to *Side-by-side Comparison* to show how Model B's sub-die block readings sharpen probability separation along defect boundaries compared to Model A's spatial-only field.
3. **Spatial Cluster Die Inspection**: In the Die Selector, inspect benchmark Die `(40, 18)` on `W_F_0014` (90.5% fail probability); note that `sp_dist_to_fail` and `blk_mean` dominate the SHAP attribution, classifying it into *Cluster 2 (Defect Neighborhood Proximity)*.
4. **Sub-Die Block Signal Profile**: Scroll down to the *Sub-Die Block Signal Profile* to reveal the 2,000-reading sequential strip plot for Die `(40, 18)` with anomalous readings (>2×MAD from median, matching `src/block_features.py`) highlighted in red (360 anomalous blocks, 18.0%), clearly noting the index-position limitation.
5. **Boundary Sensitivity & Benchmark Audit**: Select wafer `W_F_0016`, Die `(21, 12)` to demonstrate the marginal threshold case (52.1% probability) where normalizing `blk_mean` drops failure risk to 18.3% (crossing below the 0.518 threshold into Pass), concluding with the bottom audit table proving block features deliver statistically significant PR-AUC gains (+0.034, p < 0.001) while F1 remains constrained by marginal defect overlap.

---

## Input Structure

Files are generated in `input/` directory:

| File | Description |
|------|-------------|
| `train.csv` / `test.csv` |Labeled train/test data for model development |
| `validation.csv` | Unlabeled data for model validation |

## Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| `wafer_id` | Identifier | Unique wafer identifier |
| `die_row`, `die_col` | Identifier | Die position on wafer grid |
| `feature_1` ... `feature_n` | Feature | Parametric test measurements |
| `block_readings` | Feature | Space-separated array of k float values (sub-die block test signal, default k=2000) |
| `old_label` | Feature | **Pre-test** die status: 0=pass, 1=fail (from WM-811K map) |
| `label` | Label (target) | **Post-test** die status: 0=pass, 1=fail (old fails + new fails) |

## Evaluation Logic

The evaluation should **only consider eligible dies** (those with `old_label=0`). Dies that were already failed (`old_label=1`) should be excluded because they trivially remain failed and would inflate accuracy numbers.

**Metrics:**
- **Overall Accuracy**: Correct predictions / eligible dies
- **Pass Accuracy (Recall)**: Of dies that stayed pass, fraction correctly predicted as pass
- **Fail Accuracy (Recall)**: Of dies that newly failed, fraction correctly predicted as fail
- **Pass Precision**: Of dies predicted as pass, fraction that actually passed
- **Fail Precision**: Of dies predicted as fail, fraction that actually failed
- **Pass F1 Score**: Harmonic mean of pass precision and pass accuracy (Recall)
- **Fail F1 Score**: Harmonic mean of fail precision and fail accuracy (Recall)

**Expected Confusion Matrix Output:**
```
                      Pred Fail  Pred Pass       Metric               Value
  Actual Fail            1330       1164       Fail Accuracy        0.533280
  Actual Pass             118      23917       Pass Accuracy        0.995090
```

## Prediction Submission Format

Predictions file must be a CSV with columns:

```csv
wafer_id,die_row,die_col,predicted_label
W_F_0001,5,12,0
W_F_0001,5,13,1
...
```

- `predicted_label`: 0=pass, 1=fail (for ALL dies, including old fails)

---

## Repository Structure
- **`src/`**: Modular Python library containing data loaders, spatial and block feature engineering, anomaly detectors, Model A/B definitions, evaluation routines, and visualization tools.
- **`dashboard/`**: Interactive Streamlit presentation application (`app.py`), cloud dependencies (`requirements.txt`), and live demo guide.
- **`input/`**: Directory holding raw and generated dataset CSVs (`train.csv`, `test.csv`, and `validation.csv`).
- **`outputs/`**: Final production model checkpoints (`model_a.pkl`, `model_b.pkl`), evaluation tables, exact SHAP statistics, wafer heatmaps, and comprehensive documentation.
- **`archive/`**: Archived intermediate tuning scripts, experimental runs, and scratch files preserved for reproducibility.
- **`docs/`**: Non-technical plain-language project documentation and deliverable guide ([`project_explanation.md`](docs/project_explanation.md)).

---

## How to Reproduce
All steps can be reproduced from the project root using standard Python 3.10+:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated Part 1 regression guardrails (verifies data alignment and assertions)
python test_part1_guardrails.py

# 3. (Optional) Regenerate synthetic data from LSWMD.pkl using local config if input/*.csv are missing
python generate_data.py --config config_local.yaml

# 4. Execute end-to-end training, evaluation, statistical tests, SHAP, and report generation
python run_all.py --config config_local.yaml

# 5. Generate submission predictions CSV
python finalize.py

# 6. Launch interactive Streamlit presentation dashboard
streamlit run dashboard/app.py
```
