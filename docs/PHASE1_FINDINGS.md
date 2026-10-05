# PHASE 1 FINDINGS — Repository Inspection Report

**Inspection date**: 2026-10-05
**Inspector**: Antigravity (senior ML/software engineer, hackathon final mode)
**Baseline commit**: f68d8d4
**Branch created**: `final-work`
**Tag created**: `pre-final-freeze` → f68d8d4

---

## 1. Data Presence

| File | Exists? | Size |
|------|---------|------|
| `input/train.csv` | **YES** | 3.88 GB |
| `input/test.csv` | **YES** | 882 MB |
| `input/validation.csv` | YES (unlabeled) | 882 MB |
| `LSWMD.pkl` | **YES** | 2.00 GB |

**DATA_PRESENT = YES**. All three CSVs and the source WM-811K pickle are present locally.

---

## 2. Environment

| Item | Value |
|------|-------|
| Python | 3.13.1 |
| pip | 26.2 |
| lightgbm | 4.7.0 |
| shap | 0.52.0 |
| joblib | 1.5.1 |
| streamlit | 1.63.0 |
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| scipy | 1.16.1 |
| scikit-learn | 1.5.2 |
| matplotlib | 3.9.2 |
| pyarrow | 25.0.1 |
| PyYAML | 6.0.2 |

All packages required by `dashboard/requirements.txt` are installed and within the stated ranges.

---

## 3. Git History

```
f68d8d4  Update README with live Streamlit Cloud deployment URL   <- HEAD / baseline
2ce9cc2  Update cloud link in README.md
0c9a5ff  Add Streamlit presentation dashboard, predictions CSV, and cloud deployment cache
b0fe67f  Initial commit: Die yield prediction pipeline with Model A/B, statistical audit
```

4 commits total. 8 untracked files (documents not yet added to git, not frozen artifacts).

---

## 4. Frozen Artifact Verification (Pre-Code)

| Artifact | Size | Status |
|----------|------|--------|
| `outputs/model_a.pkl` | 3.57 MB | EXISTS |
| `outputs/model_b.pkl` | 3.57 MB | EXISTS |
| `outputs/model_a_meta.pkl` | 7.2 KB | EXISTS |
| `outputs/model_b_meta.pkl` | 7.5 KB | EXISTS |
| `outputs/predictions.csv` | 691 KB | EXISTS |
| `outputs/comparison_table.csv` | 684 B | EXISTS |
| `outputs/per_die_explanations.md` | 4.0 KB | EXISTS |
| `outputs/cache/test_probs_a.npy` | 315 KB | EXISTS |
| `outputs/cache/test_probs_b.npy` | 315 KB | EXISTS |

Both models have `n_estimators=500` and `best_iteration_=500` (never reached early stopping, as specified).

---

## 5. Known-Good Number Verification (PASS — No STOP required)

Numbers recomputed from cached artifacts using `average_precision_score` (trapezoidal PR-AUC):

| Metric | Prompt Spec | Computed | Match? |
|--------|-------------|----------|--------|
| Eligible dies | 32,598 | **32,598** | YES |
| True fails | 1,380 | **1,380** | YES |
| A PR-AUC | 0.5025 | **0.5025** | YES |
| A F1 | 0.5207 | **0.5207** | YES |
| A Precision | 97.0% | **97.04%** | YES |
| A Recall | 35.6% | **35.58%** | YES |
| A TP/FP/FN | 491/15/889 | **491/15/889** | YES |
| B PR-AUC | 0.5362 | **0.5362** | YES |
| B F1 | 0.5222 | **0.5222** | YES |
| B Precision | 88.1% | **88.12%** | YES |
| B Recall | 37.1% | **37.10%** | YES |
| B TP/FP/FN | 512/69/868 | **512/69/868** | YES |
| Threshold A | 0.5574 | **0.557358** | YES |
| Threshold B | 0.5180 | **0.518027** | YES |

**No STOP condition triggered. All known-good numbers reproduce exactly.**

### Meta File Consistency

| Meta field | Model A | Model B | Expected |
|-----------|---------|---------|----------|
| threshold | 0.5574 | 0.5180 | matches prompt |
| spw | 12.0 | 12.0 | matches prompt |
| val_fail_f1 | 0.5299 | 0.5310 | matches prompt |
| Feature count | 511 (500+10+1) | 531 (511+19+1) | matches prompt |

Spatial columns in meta: `sp_row_norm, sp_col_norm, sp_radial, sp_edge_prox,
sp_old_fail_density_3, sp_old_fail_density_5, sp_old_fail_density_7,
sp_zone_yield, sp_dist_to_fail, sp_old_label` = **10 spatial features**
(NOT "24" as stated in README line 9 and results_summary.md — documentation error to fix P0).

Block columns beyond Model A: 19 block statistics + `blk_anomaly_score` = 20 block features (total 531). Matches prompt.

---

## 6. Script-by-Script Analysis

### `archive/retrain_and_evaluate.py` (source of committed artifacts)
- Anomaly detectors: n_estimators=100 (correct)
- Threshold sweep: n_thresholds=300 (correct)
- Split seed=42 (correct)
- Writes directly to `outputs/` with NO guard — OVERWRITE RISK (P0.2 fix)
- Also writes `model_a_retune.pkl` / `model_b_retune.pkl` duplicates to `outputs/`

### `run_all.py`
- Anomaly detectors: **n_estimators=200** — DIFFERENT from archive/retrain_and_evaluate.py (200 vs 100)
- `OUTPUT_DIR`, `PLOTS_DIR`, `SHAP_DIR`, `MAPS_DIR` hardcoded as `outputs/` — no `--out-dir` flag (P0.3 fix)
- No `--overwrite-official` guard
- Will produce different anomaly scores than committed artifacts

### `finalize.py`
- Correctly uses DieAnomalyDetector(n_estimators=100) and BlockAnomalyDetector(n_estimators=100)
- Reads cached sp_test.parquet, blk_test.parquet, blk_tr.parquet from outputs/cache/ if present
- No docstring stating it is the official submission path (P0.4)
- No self-check against outputs/cache/test_probs_b.npy (P0.4)

### `src/*.py`
- 10 modules present, all with assert_aligned/assert_split_disjoint guards
- evaluation.py uses average_precision_score (trapezoidal)
- spatial_features.py: test_no_leakage present, features from old_label only

### `dashboard/app.py`
- Line 788: Key Architectural Takeaway block — two errors:
  1. "p = 1.33e-5" cited as test-set evidence but is a VALIDATION paired t-test p-value (P0.8)
  2. "+0.0343 to +0.0359" mixes test point estimate with validation mean
- No semi-synthetic banner (P0.9)
- PRECOMPUTED_REPRESENTATIVES: 5 entries present
- Trajectory for W_F_0014(40,18) includes "Normalizing sp_dist_to_fail" step (P5 cleanup)
- Domain attribution table already shows "Spatial Neighborhood (10)" correctly

### `requirements.txt` (root)
Missing: lightgbm, shap, joblib, streamlit vs. dashboard/requirements.txt (P0.6 fix)

### `test_part1_guardrails.py`
- Tests 1-2: no data needed (mock + model load from outputs/)
- Tests 3-5: require DATA_PRESENT, refit anomaly detectors (stochastic)
- Test 5: checks F1 match vs comparison_table.csv within 1e-4

---

## 7. Contradictions vs. Prompt

| Location | Issue | Severity | Phase |
|----------|-------|----------|-------|
| README.md L9 | "24 leakage-safe spatial features" (actual: 10) | HIGH | P0.7 |
| README.md L17 | A F1=0.5209, PR-AUC=0.5014 (actual: 0.5207, 0.5025) | LOW (rounding) | P0.7 |
| README.md L18 | "p = 1.33e-5" as test evidence (it is validation t-test) | HIGH | P0.8 |
| README.md L37 | "p < 0.001" as test evidence | HIGH | P0.8 |
| results_summary.md L3 | "24 Spatial" in feature count | HIGH | P0.7 |
| results_summary.md L76-79 | TP=492, FP=17, FN=888 (cache: TP=491, FP=15, FN=889) | MEDIUM | P0.7 |
| dashboard/app.py L788 | "p = 1.33e-5" as test-set evidence | HIGH | P0.8 |
| docs/project_explanation.md | Deliverable 4 conflates validation tests with test-set | MEDIUM | P0.8 |
| run_all.py | n_estimators=200 anomaly detectors vs 100 in official | RISK | P0.3 guard |
| requirements.txt (root) | Missing lightgbm, shap, joblib, streamlit | HIGH | P0.6 |
| audit_numbers.py | Does NOT exist in repo | BLOCKER | P0.7 |

---

## 8. Key Risk Summary

| Risk | Impact | Phase |
|------|--------|-------|
| run_all.py writes to outputs/ with 200-tree detectors | Overwrite frozen artifacts | P0.3 |
| archive/retrain_and_evaluate.py writes to outputs/ | Overwrite frozen artifacts | P0.2 |
| README/results_summary say "24 spatial" (wrong) | Credibility | P0.7 |
| README/dashboard cite p=1.33e-5 as test evidence | Statistical credibility | P0.8 |
| No semi-synthetic disclosure | Transparency | P0.9 |
| requirements.txt (root) incomplete | Reproducibility | P0.6 |
| audit_numbers.py does not exist | No automated number lock | P0.7 |
| results_summary.md TP/FP/FN numbers differ from cache | Wrong numbers in report | P0.7 |

---

## 9. audit_numbers.py Status

`audit_numbers.py` does NOT exist in the repo root. Because cached
test_probs_a.npy, test_probs_b.npy, and test_meta.parquet reproduce the
known-good numbers exactly (verified by direct computation above), there is
NO STOP condition. Proceed to create audit_numbers.py in P0.7.

---

## 10. Confirmed No-STOP

Both STOP conditions are cleared:
1. Frozen artifact hashes: cached probabilities reproduce known-good numbers exactly.
2. audit_numbers.py: does not yet exist but manual recomputation confirms all numbers match.

---

## 11. Phase Plan (Ordered by Risk)

| Step | Action | Risk if skipped |
|------|--------|-----------------|
| P0.1 | tools/freeze_artifacts.py + tools/check_frozen.py | No overwrite detection |
| P0.2 | train_final.py (copy archive, add --out-dir + guard) | Accidental retrain overwrites |
| P0.3 | run_all.py: add --out-dir guard | Accidental overwrite during demo |
| P0.4 | finalize.py: docstring + self-check vs cached probs | Silent regression |
| P0.5 | PROVENANCE.md | No audit trail |
| P0.6 | Align requirements.txt files + produce requirements-lock.txt | Install failures |
| P0.7 | Create audit_numbers.py + fix "24 spatial" + fix TP/FP/FN | Wrong numbers in submission |
| P0.8 | Fix statistical language (p-value, Wilcoxon) in README, dashboard, docs | Credibility risk |
| P0.9 | Add semi-synthetic banner | Transparency |

---

*This file was created as the first step of the hackathon final pipeline.
No code was modified before this file was committed.*
