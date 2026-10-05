# Model Provenance

This document records which scripts produced the official model artifacts,
how to reproduce them, and the status of each script.

---

## Official Submission Artifacts

The following artifacts in `outputs/` were produced by a single run of
`archive/retrain_and_evaluate.py` at commit f68d8d4:

| Artifact | Script | Notes |
|----------|--------|-------|
| `outputs/model_a.pkl` | `archive/retrain_and_evaluate.py` | 500 trees, spw=12, no early stop |
| `outputs/model_b.pkl` | `archive/retrain_and_evaluate.py` | 500 trees, spw=12, no early stop |
| `outputs/model_a_meta.pkl` | `archive/retrain_and_evaluate.py` | threshold=0.5574, val_fail_f1=0.5299 |
| `outputs/model_b_meta.pkl` | `archive/retrain_and_evaluate.py` | threshold=0.5180, val_fail_f1=0.5310 |
| `outputs/predictions.csv` | `finalize.py` | 39,351 rows, Model B, threshold=0.5180 |
| `outputs/comparison_table.csv` | `archive/retrain_and_evaluate.py` | trapezoidal PR-AUC |
| `outputs/cache/test_probs_a.npy` | `archive/retrain_and_evaluate.py` | shape (39351,) |
| `outputs/cache/test_probs_b.npy` | `archive/retrain_and_evaluate.py` | shape (39351,) |

All hashes are recorded in `outputs/ARTIFACT_HASHES.json` (SHA-256).
Run `python tools/check_frozen.py` to verify integrity at any time.

---

## Official Reproduction Script

`train_final.py` is a direct copy of `archive/retrain_and_evaluate.py` with
two additions:

1. `--out-dir` (default: `outputs_repro/`) — redirects all output away from `outputs/`.
2. Refuses to write to `outputs/` unless `--overwrite-official` is passed.

**To reproduce the models from scratch (requires DATA_PRESENT):**

```bash
python train_final.py          # writes to outputs_repro/
# Then run finalize.py to score the test set:
python finalize.py
```

### Key parameters (must not change):
- `n_estimators=500`, `learning_rate=0.05`, `num_leaves=63`
- `scale_pos_weight=12.0`, `random_state=42`
- `DieAnomalyDetector(n_estimators=100, contamination=0.05)`
- `BlockAnomalyDetector(n_estimators=100, contamination=0.05)`
- `wafer_level_split(val_fraction=0.2, seed=42)`
- `select_threshold(val_df, y_prob, n_thresholds=300)`

---

## `run_all.py` Status: EXPERIMENTAL

`run_all.py` is **NOT** the official submission pipeline. Differences:

| Property | Official (`archive/retrain_and_evaluate.py`) | Experimental (`run_all.py`) |
|----------|----------------------------------------------|----------------------------|
| Anomaly detector trees | 100 | **200** |
| Default output dir | `outputs/` (guarded by `train_final.py`) | `outputs_runall/` (guarded) |
| Purpose | Official training run | End-to-end demo / ablation |

`run_all.py` will produce **different anomaly scores** from the committed
artifacts (different detector random state). It is safe to run with
`--out-dir outputs_runall` (the default).

---

## Data Generation

`generate_data.py` produces the semi-synthetic dataset from `LSWMD.pkl`.

**Data description**: Real WM-811K wafer geometry and pre-test maps combined
with synthetic die-level parametric measurements, synthetic sub-die block
readings, and controlled new-failure labels.

See `config_local.yaml` for generator parameters. Never modify `input/*.csv`
directly — regenerate via `generate_data.py` if needed.

---

## Phase Freeze

After the freeze tag `pre-final-freeze` (commit f68d8d4), no frozen artifacts
were modified. All subsequent changes are documentation, tooling, and dashboard
updates only.
