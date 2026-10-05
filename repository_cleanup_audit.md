# Repository Cleanup and Structural Audit

**Project:** SanDisk Semiconductor Die Yield Prediction (Model A vs. Model B)  
**Date:** October 6, 2026  
**Status:** Audit Complete & Verified  

---

## 1. Executive Summary

This repository cleanup audit was performed prior to executing file operations to ensure that:
1. **No ML/model logic was altered** (Model A and Model B code, hyperparameters, thresholds, features remain 100% frozen).
2. **All 21 frozen artifacts** in `outputs/ARTIFACT_HASHES.json` remain bit-for-bit identical (verified by `tools/check_frozen.py`).
3. **No working functionality was broken** (both Tier A and Tier B smoke tests pass cleanly; the Streamlit dashboard loads all presets and views without error).
4. **No secrets or credentials exist** in the repository.
5. **Obsolete temporary caches and duplicate reproduction folders** (`outputs_repro/`, `outputs_smoke/`, `__pycache__/`) are safely purged.
6. **A standardized `tests/` directory** is established to provide clear separation for automated regression and guardrail tests while maintaining 100% backward-compatible execution for existing scripts and documentation.

---

## 2. File Classification Matrix

Every file and directory in the repository was audited and classified into the standardized categories:

| Category Code | Description | Count |
|:---:|:---|:---:|
| **A** | Required for Application (Dashboard runtime) | 12 |
| **B** | Required for ML/Model Pipeline (Training, feature extraction, evaluation) | 28 |
| **C** | Required for Demo / Presentation (Slides, reports, presentations) | 10 |
| **D** | Required Configuration (Config files, package dependencies, gitignore) | 6 |
| **E** | Test / Verification (Smoke tests, guardrails, hash checkers, audit scripts) | 5 |
| **F** | Documentation (README, technical docs, design system, provenance) | 6 |
| **G** | Historical / Reference (Preserved research scripts, milestone experiments) | 7 |
| **H** | Duplicate | 0 |
| **I** | Unused | 0 |
| **J** | Generated / Temporary (Safe to remove from local workspace) | 4 dirs |
| **K** | Unknown — Do Not Delete | 0 |

---

## 3. Files and Directories Kept

### 3.1 Root Configuration & Top-Level Deliverables
- **`README.md`** `[F. DOCUMENTATION]`: Primary landing page, setup guide, architecture overview, and quick-start instructions.
- **`design.md`** `[F. DOCUMENTATION]`: Complete SanDisk-inspired design tokens, CSS architecture, and UI specifications for the dashboard.
- **`PROVENANCE.md`** `[F. DOCUMENTATION]`: Explicit provenance ledger documenting commit histories, data generation parameters, and artifact origins.
- **`sandisk_project_walkthrough.md`** `[F. DOCUMENTATION]`: Comprehensive architectural deep-dive document covering every pipeline stage and formula.
- **`requirements.txt`** `[D. REQUIRED CONFIGURATION]`: Top-level Python dependency specification.
- **`requirements-lock.txt`** `[D. REQUIRED CONFIGURATION]`: Exact pinned dependencies for deterministic environment replication.
- **`config.yaml`** `[D. REQUIRED CONFIGURATION]`: Pipeline configuration file with feature definitions, model hyperparameters, and paths.
- **`config_local.yaml`** `[D. REQUIRED CONFIGURATION]`: Local development configuration specifying local path for `LSWMD.pkl`.
- **`.gitignore`** `[D. REQUIRED CONFIGURATION]`: Rules ignoring bytecode, heavy CSVs, virtual environments, and temporary test directories.
- **`LSWMD.pkl`** `[B. REQUIRED FOR ML PIPELINE / DATASET]`: Raw 2.09 GB WM-811K semiconductor wafer map pickle (ignored by git, kept locally for `generate_data.py`).

### 3.2 Root Pipeline & Verification Entry Points
- **`smoke_test.py`** `[E. TEST/VERIFICATION]`: Central automated verification suite supporting `--tier A` (no data dependencies) and `--tier B` (full pipeline).
- **`finalize.py`** `[B. REQUIRED FOR ML PIPELINE]`: Official competition submission script generating `outputs/predictions.csv`.
- **`train_final.py`** `[B. REQUIRED FOR ML PIPELINE]`: Official reproduction training script for Model A and Model B checkpoints.
- **`calibrate.py`** `[B. REQUIRED FOR ML PIPELINE]`: Platt scaling sigmoid calibrator script.
- **`generate_data.py`** `[B. REQUIRED FOR ML PIPELINE]`: Semi-synthetic data generator combining WM-811K real wafer maps with parametric signals.
- **`run_all.py`** `[B. REQUIRED FOR ML PIPELINE]`: Master pipeline orchestrator script.
- **`audit_numbers.py`** `[E. TEST/VERIFICATION]`: Verifies all headline, calibration, and operational triage numbers against official audited values.
- **`test_part1_guardrails.py`** `[E. TEST/VERIFICATION]`: Root-level wrapper/entry point for regression guardrails, maintaining 100% backward compatibility.
- **`create_presentation.py`** `[C. REQUIRED FOR DEMO/PRESENTATION]`: Programmatic PowerPoint generator for `outputs/pitch_deck.pptx`.
- **`generate_submission_report.py`** `[C. REQUIRED FOR DEMO/PRESENTATION]`: Programmatic ReportLab PDF generator for `outputs/submission_report.pdf`.

### 3.3 Application (`dashboard/`)
- **`dashboard/app.py`** `[A. REQUIRED FOR APPLICATION]`: Streamlit web application featuring 4-panel wafer inspection, side-by-side model comparison, sub-die block signal strips, and SHAP explainability.
- **`dashboard/README.md`** `[F. DOCUMENTATION]`: Dashboard user guide and demo script.
- **`dashboard/requirements.txt`** `[D. REQUIRED CONFIGURATION]`: Minimal cloud dependencies for Streamlit Community Cloud deployment.

### 3.4 Core Source Library (`src/`)
- **`src/__init__.py`**: Package initialization.
- **`src/data_loader.py`**: Robust CSV loader, wafer-level disjoint splitting, and key alignment assertions.
- **`src/spatial_features.py`**: Spatial wafer context engineering (radial distance, edge distance, ring sector, 2D continuous Gaussian risk field).
- **`src/block_features.py`**: Sub-die block signal aggregation and tail risk statistics.
- **`src/anomaly_features.py`**: Isolation Forest anomaly scoring on healthy dies.
- **`src/model_a.py`**: LightGBM Model A training pipeline and feature sets (511 features).
- **`src/model_b.py`**: LightGBM Model B training pipeline and feature sets (531 features).
- **`src/evaluation.py`**: PR-AUC, ROC-AUC, threshold optimization, and confusion matrix computation.
- **`src/calibration.py`**: Platt scaling sigmoid calibration model and quantile ECE evaluation.
- **`src/triage.py`**: Operational screening triage simulation and cumulative capture curves.
- **`src/visualization.py`**: Wafer map rendering, risk field heatmaps, and block profile strip plots.
- **`src/explainability.py`**: TreeSHAP feature attributions and natural language explanation generation.

### 3.5 Tools (`tools/`)
- **`tools/check_frozen.py`** `[E. TEST/VERIFICATION]`: Cryptographic SHA-256 hash verifier for the 21 frozen artifacts.
- **`tools/freeze_artifacts.py`** `[B. REQUIRED FOR ML PIPELINE]`: Hashes production artifacts and updates `outputs/ARTIFACT_HASHES.json`.
- **`tools/generate_triage_slide.py`** `[C. REQUIRED FOR DEMO/PRESENTATION]`: Generates high-resolution triage curve graphic for presentation slides.

### 3.6 Tests (`tests/`)
- **`tests/__init__.py`**: Package initialization.
- **`tests/test_guardrails.py`** `[E. TEST/VERIFICATION]`: Part 1 guardrail regression suite (5 automated checks: alignment, training sanity floor, threshold independence, reproducibility, disk reload).

### 3.7 Documentation (`docs/`)
- **`docs/technical_documentation.md`** `[F. DOCUMENTATION]`: Complete 90+ KB technical specification of all algorithms, equations, data splits, and model architectures.
- **`docs/project_explanation.md`** `[F. DOCUMENTATION]`: Plain-language project guide for non-technical stakeholders.
- **`docs/PHASE1_FINDINGS.md`** `[F. DOCUMENTATION / G. HISTORICAL]`: Phase 1 baseline exploration and exploratory analysis.

### 3.8 Production & Frozen Artifacts (`outputs/`)
All 21 frozen artifacts verified bit-for-bit:
1. `outputs/model_a.pkl`: Model A LightGBM checkpoint.
2. `outputs/model_b.pkl`: Model B LightGBM checkpoint.
3. `outputs/model_a_meta.pkl`: Model A metadata (threshold=0.5574, features, metrics).
4. `outputs/model_b_meta.pkl`: Model B metadata (threshold=0.5180, features, metrics).
5. `outputs/predictions.csv`: Official competition test submission predictions (39,351 rows).
6. `outputs/comparison_table.csv`: Authoritative Model A vs Model B evaluation table.
7. `outputs/per_die_explanations.md`: Natural language explanations for benchmark dies.
8. `outputs/cache/test_probs_a.npy`: Test set predicted probabilities for Model A.
9. `outputs/cache/test_probs_b.npy`: Test set predicted probabilities for Model B.
10. `outputs/cache/blk_anom_test.npy`: Test set block anomaly scores.
11. `outputs/cache/blk_test.parquet`: Test set block features.
12. `outputs/cache/blk_tr.parquet`: Training set block features.
13. `outputs/cache/blk_train_full.parquet`: Full training split block features.
14. `outputs/cache/blk_val.parquet`: Validation set block features.
15. `outputs/cache/die_anom_test.npy`: Test set die anomaly scores.
16. `outputs/cache/pass_medians_a.parquet`: Median feature values for Model A counterfactuals.
17. `outputs/cache/pass_medians_b.parquet`: Median feature values for Model B counterfactuals.
18. `outputs/cache/sp_test.parquet`: Test set spatial features.
19. `outputs/cache/sp_tr.parquet`: Training set spatial features.
20. `outputs/cache/sp_val.parquet`: Validation set spatial features.
21. `outputs/cache/test_meta.parquet`: Test set die coordinates and ground-truth metadata.

Additionally preserved in `outputs/`:
- `outputs/audited_numbers.json`: Authoritative numerical truth consumed by dashboard and smoke test.
- `outputs/calibration/`: Platt calibrator checkpoints (`calibrator_a.pkl`, `calibrator_b.pkl`, `calibration_meta.json`, `reliability.png`).
- `outputs/cache/wafer_blocks/` (30 parquets): Precomputed block statistics for instant dashboard drilldowns.
- `outputs/cache/wafer_features/` (30 parquets): Precomputed die features for instant wafer map rendering.
- `outputs/cache/test_probs_a_cal.npy` & `outputs/cache/test_probs_b_cal.npy`: Calibrated probability arrays.
- `outputs/pitch_deck.pptx`: Official PowerPoint slide presentation.
- `outputs/submission_report.pdf`: Official 9-page executive PDF technical report.
- `outputs/pitch_script.md` & `outputs/pitch_script.pdf`: Spoken presentation script.
- `outputs/shap/`: Precomputed SHAP feature importance and summary plots.
- `outputs/plots/`: High-resolution figures (risk fields, block strips, distributions).
- `outputs/wafer_maps/`: Comparison wafer maps.
- `outputs/slides/`: Standalone slide figures (`triage_curve.png`, `reliability.png`).
- Supporting CSVs: `ablation_table_multiseed.csv`, `multiseed_statistical_tests.csv`, `multiseed_per_seed_runs.csv`, `failure_signatures.csv`, `shap_importance_exact.csv`, `shap_mass_breakdown.csv`, `test_bootstrap_ci.csv`, `spw_grid_search_corrected.csv`.

---

## 4. Files Kept as Historical / Reference (`archive/`)

The following files are preserved strictly in `archive/` because they represent historical research milestones and one-off experiments documented in `PROVENANCE.md` and `sandisk_project_walkthrough.md`:

| File | Purpose | Reason Preserved |
|:---|:---|:---|
| **`archive/retrain_and_evaluate.py`** | Original training script | Source of official frozen model weights (`outputs/model_a.pkl`, `model_b.pkl`), cited in `PROVENANCE.md`. |
| **`archive/retune.py`** | SPW grid search & tuning | Produced `outputs/spw_grid_search_corrected.csv`, cited in technical documentation. |
| **`archive/run_part2_statistical_rigor.py`** | Multi-seed ablation & statistical tests | Produced `ablation_table_multiseed.csv`, `multiseed_statistical_tests.csv`, and SHAP breakdown tables. |
| **`archive/run_part3_sanity_checks.py`** | Early sanity checks | Validated target permutations and feature integrity. |
| **`archive/run_part4_interpretability.py`** | Interpretability research | Produced HDBSCAN failure signatures (`failure_signatures.csv`), counterfactual baselines, and bootstrap CIs. |
| **`archive/test_fit.py`** | Early-stopping vs fixed trees diagnostic | Documents the technical decision to use fixed 500 trees over early stopping. |
| **`archive/fix_unicode.py`** | Character encoding cleanup | Historic utility for cleaning legacy non-ASCII characters in source files. |

---

## 5. Files and Directories Removed

The following temporary and generated directories were audited and identified as completely safe to remove:

| Path | Category | Reason for Removal | Safety Justification |
|:---|:---:|:---|:---|
| **`outputs_repro/`** | `J. GENERATED/TEMPORARY` | Temporary output folder created during past reproduction test runs. | Gitignored, untracked, contains redundant duplicate model checkpoints and caches (~18 MB). |
| **`outputs_smoke/`** | `J. GENERATED/TEMPORARY` | Temporary scratch folder created during `smoke_test.py --tier B`. | Gitignored, untracked, automatically recreated when Tier B smoke tests are executed. |
| **`data/`** | `J. GENERATED/TEMPORARY` | Empty directory. | Gitignored, untracked, contains no files. |
| **`__pycache__/`** | `J. GENERATED/TEMPORARY` | Python compiled bytecode (`.pyc`). | Gitignored, automatically generated by Python interpreter. |
| **`src/__pycache__/`** | `J. GENERATED/TEMPORARY` | Python compiled bytecode (`.pyc`). | Gitignored, automatically generated by Python interpreter. |
| **`tools/__pycache__/`** | `J. GENERATED/TEMPORARY` | Python compiled bytecode (`.pyc`). | Gitignored, automatically generated by Python interpreter. |

---

## 6. Files Moved / Reorganized

| Original Location | New Location | Reason | Backward Compatibility |
|:---|:---|:---|:---|
| `test_part1_guardrails.py` (implementation) | `tests/test_guardrails.py` | Consolidates regression test suite into standard `tests/` directory. | `test_part1_guardrails.py` in root remains as a transparent forwarding shim so existing documentation and commands (`python test_part1_guardrails.py`) continue to function identically. |

---

## 7. Secrets and Security Audit

- **Regex Secret Scan**: Performed across all files for patterns matching `api_key`, `secret`, `token`, `password`, `bearer`, `auth_key`.
- **Result**: Zero secrets, credentials, API keys, or private tokens found.
- **Matches**: Only standard Python library packages (`asttokens`, `tokenizers`) and UI "Design Token" terminology in `design.md` and `dashboard/app.py`.
- **File Name Check**: No `.env`, `*.pem`, `*.key`, `id_rsa`, or credential files present in the repository.

---

## 8. Final Repository Structure

```
sandisk/
│
├── dashboard/                      # Interactive Streamlit Web Application
│   ├── app.py                      # Main dashboard application (adheres to design.md)
│   ├── README.md                   # Dashboard guide & live demo script
│   └── requirements.txt            # Streamlit Community Cloud dependencies
│
├── src/                            # Core Feature Engineering & ML Library
│   ├── __init__.py
│   ├── anomaly_features.py         # Isolation Forest & LOF anomaly detectors
│   ├── block_features.py           # Sub-die block signal statistics
│   ├── calibration.py              # Platt scaling calibrator & ECE evaluation
│   ├── data_loader.py              # Robust data ingestion & wafer splitting
│   ├── evaluation.py               # Metrics, threshold selection, evaluation
│   ├── explainability.py           # TreeSHAP feature attributions
│   ├── model_a.py                  # Model A (511 features) pipeline
│   ├── model_b.py                  # Model B (531 features) pipeline
│   ├── spatial_features.py         # Spatial context & 2D Gaussian risk fields
│   ├── triage.py                   # Operational factory screening triage
│   └── visualization.py            # Wafer maps, risk fields, block strips
│
├── tests/                          # Automated Regression & Test Suites
│   ├── __init__.py
│   └── test_guardrails.py          # 5-test Part 1 regression & guardrail suite
│
├── tools/                          # Verification & Utility Scripts
│   ├── check_frozen.py             # SHA-256 integrity verifier for 21 frozen artifacts
│   ├── freeze_artifacts.py         # Hashes production artifacts -> ARTIFACT_HASHES.json
│   └── generate_triage_slide.py    # Generates standalone triage curve figure
│
├── docs/                           # Technical Specifications & Plain-Language Guides
│   ├── technical_documentation.md  # Comprehensive 90+ KB technical specifications
│   ├── project_explanation.md      # Plain-language explanation for stakeholders
│   └── PHASE1_FINDINGS.md          # Exploratory baseline findings
│
├── archive/                        # Preserved Historical Research & Milestones
│   ├── fix_unicode.py              # Character encoding cleanup
│   ├── retrain_and_evaluate.py     # Source script for frozen models (PROVENANCE.md)
│   ├── retune.py                   # SPW grid search & tuning
│   ├── run_part2_statistical_rigor.py # Multi-seed ablation & statistical tests
│   ├── run_part3_sanity_checks.py  # Data sanity checks
│   ├── run_part4_interpretability.py  # HDBSCAN clustering & counterfactuals
│   └── test_fit.py                 # Diagnostic fitting script
│
├── outputs/                        # Production & Frozen Artifacts
│   ├── ARTIFACT_HASHES.json        # Authoritative SHA-256 hashes (21 files)
│   ├── model_a.pkl                 # Trained Model A checkpoint
│   ├── model_b.pkl                 # Trained Model B checkpoint
│   ├── model_a_meta.pkl            # Model A metadata (threshold=0.5574)
│   ├── model_b_meta.pkl            # Model B metadata (threshold=0.5180)
│   ├── predictions.csv             # Official test set predictions (39,351 rows)
│   ├── comparison_table.csv        # Official Model A vs B comparison table
│   ├── audited_numbers.json        # Authoritative audited metrics
│   ├── pitch_deck.pptx             # PowerPoint executive presentation
│   ├── submission_report.pdf       # 9-page executive PDF report
│   ├── calibration/                # Fitted Platt calibrators & reliability curves
│   ├── cache/                      # Precomputed feature caches & wafer parquets
│   ├── plots/                      # High-res risk fields & distributions
│   ├── shap/                       # SHAP summary & importance plots
│   ├── slides/                     # Triage curves & reliability graphics
│   └── wafer_maps/                 # Precomputed comparison wafer maps
│
├── input/                          # Input Data Directory (Ignored in Git)
│   ├── train.csv                   # Full training dataset (local)
│   ├── validation.csv              # Validation dataset (local)
│   └── test.csv                    # Test dataset (local)
│
├── audit_numbers.py                # Standalone metrics verifier vs audited_numbers.json
├── calibrate.py                    # Platt scaling probability calibration pipeline
├── config.yaml                     # Master pipeline configuration
├── config_local.yaml               # Local development overrides
├── create_presentation.py          # Programmatic generator for outputs/pitch_deck.pptx
├── design.md                       # SanDisk UI redesign design system & specs
├── finalize.py                     # Competition submission prediction generator
├── generate_data.py                # Semi-synthetic data generator from WM-811K
├── generate_submission_report.py   # Programmatic generator for outputs/submission_report.pdf
├── PROVENANCE.md                   # Programmatic data & model provenance ledger
├── README.md                       # Main repository overview & quick-start guide
├── requirements.txt                # Production environment dependencies
├── requirements-lock.txt           # Pinned dependency lockfile
├── run_all.py                      # Master end-to-end retraining orchestrator
├── sandisk_project_walkthrough.md  # Architectural deep-dive & code walkthrough
├── smoke_test.py                   # Primary multi-tier automated test suite (--tier A/B)
├── test_part1_guardrails.py        # Backward-compatible guardrail test wrapper
├── train_final.py                  # Production model training reproduction script
└── .gitignore                      # Git ignore patterns
```
