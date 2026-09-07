# Die Yield Prediction & Diagnostics Streamlit Dashboard

## How to Run

To run the interactive presentation dashboard, install dependencies from `dashboard/requirements.txt` (or verify `streamlit` is installed) and launch the application from the repository root using:

```bash
streamlit run dashboard/app.py
```

The application will launch on `http://localhost:8501`. It loads all precomputed models, probability fields, and evaluation benchmarks directly from `outputs/` without triggering retraining.

---

## 5-Line Live Presentation Demo Script

1. **Overview & Wafer Risk Field**: Open the dashboard with default wafer `W_F_0014` selected, observe the 4-panel view comparing the binary pre-test map to the continuous 2D Gaussian risk field, and highlight the prominent cyan hotspot contour encircling the outer defect cluster.
2. **Side-by-Side Model Comparison**: Switch the sidebar toggle to *Side-by-side Comparison* to show how Model B's sub-die block readings sharpen probability separation along defect boundaries compared to Model A's spatial-only field.
3. **Spatial Cluster Die Inspection**: In the Die Selector, inspect benchmark Die `(40, 18)` on `W_F_0014` (90.5% fail probability); note that `sp_dist_to_fail` and `blk_mean` dominate the SHAP attribution, classifying it into *Cluster 2 (Defect Neighborhood Proximity)*.
4. **Sub-Die Block Signal Profile**: Scroll down to the *Sub-Die Block Signal Profile* to reveal the 2,000-reading sequential strip plot for Die `(40, 18)` with anomalous readings (>2×MAD from median, matching `src/block_features.py`) highlighted in red (360 anomalous blocks, 18.0%), clearly noting the index-position limitation.
5. **Boundary Sensitivity & Benchmark Audit**: Select wafer `W_F_0016`, Die `(21, 12)` to demonstrate the marginal threshold case (52.1% probability) where normalizing `blk_mean` drops failure risk to 18.3% (crossing below the 0.518 threshold into Pass), concluding with the bottom audit table proving block features deliver statistically significant PR-AUC gains (+0.034, p < 0.001) while F1 remains constrained by marginal defect overlap.
