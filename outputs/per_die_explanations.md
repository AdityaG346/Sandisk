# Per-Die Diagnostic Explanations and Model-Based Counterfactuals

> [!IMPORTANT]
> **Model-Based Counterfactual Disclaimer**: The counterfactuals below demonstrate the mathematical model's hypothetical output response when individual feature inputs are shifted to the pass-population median. They represent *mathematical model sensitivity analysis*, NOT physical or causal semiconductor manufacturing interventions.


### Die 1: Edge-Region Die -- Wafer W_F_0014, Row 40, Col 18

**Standardized Explanation:**
> **Die (40,18)**: failure probability **90.5%**, confidence **Moderate Confidence** (ensemble $\sigma = 0.101$). Main reasons: 1) `sp_dist_to_fail` (SHAP -1.0208), 2) `blk_mean` (SHAP +0.7047), 3) `blk_q75` (SHAP +0.3758). Resembles failure signature cluster 2 (Defect Neighborhood Proximity & Spatial Clustering). Counterfactual: normalizing `sp_dist_to_fail` would reduce risk to **90.5%** (normalizing all top-3 reduces risk to **76.2%**).

- **Counterfactual Probability Trajectory**: Original (90.5%) $\rightarrow$ Normalizing `sp_dist_to_fail` (90.5%) $\rightarrow$ +`blk_mean` (81.8%) $\rightarrow$ +`blk_q75` (76.2%)

### Die 2: Wafer-Interior Die -- Wafer W_F_0019, Row 22, Col 22

**Standardized Explanation:**
> **Die (22,22)**: failure probability **74.0%**, confidence **Moderate Confidence** (ensemble $\sigma = 0.095$). Main reasons: 1) `sp_dist_to_fail` (SHAP -0.9181), 2) `blk_mean` (SHAP +0.2001), 3) `blk_std` (SHAP -0.1873). Resembles failure signature cluster 2 (Defect Neighborhood Proximity & Spatial Clustering). Counterfactual: normalizing `sp_dist_to_fail` would reduce risk to **70.4%** (normalizing all top-3 reduces risk to **69.7%**).

- **Counterfactual Probability Trajectory**: Original (74.0%) $\rightarrow$ Normalizing `sp_dist_to_fail` (70.4%) $\rightarrow$ +`blk_mean` (63.0%) $\rightarrow$ +`blk_std` (69.7%)

### Die 3: High Anomaly-Score Die -- Wafer W_F_0009, Row 9, Col 39

**Standardized Explanation:**
> **Die (9,39)**: failure probability **88.3%**, confidence **High Confidence** (ensemble $\sigma = 0.029$). Main reasons: 1) `sp_dist_to_fail` (SHAP -0.9906), 2) `blk_mean` (SHAP +0.9579), 3) `feature_291` (SHAP -0.2176). Resembles failure signature cluster 2 (Defect Neighborhood Proximity & Spatial Clustering). Counterfactual: normalizing `sp_dist_to_fail` would reduce risk to **88.3%** (normalizing all top-3 reduces risk to **75.0%**).

- **Counterfactual Probability Trajectory**: Original (88.3%) $\rightarrow$ Normalizing `sp_dist_to_fail` (88.3%) $\rightarrow$ +`blk_mean` (72.3%) $\rightarrow$ +`feature_291` (75.0%)

### Die 4: Spatial Defect-Cluster Die -- Wafer W_F_0010, Row 48, Col 6

**Standardized Explanation:**
> **Die (48,6)**: failure probability **98.4%**, confidence **High Confidence** (ensemble $\sigma = 0.012$). Main reasons: 1) `blk_mean` (SHAP +0.7140), 2) `sp_dist_to_fail` (SHAP -0.5571), 3) `feature_97` (SHAP -0.1885). Resembles failure signature cluster -1 (Mixed / Boundary Pattern). Counterfactual: normalizing `blk_mean` would reduce risk to **97.1%** (normalizing all top-3 reduces risk to **95.5%**).

- **Counterfactual Probability Trajectory**: Original (98.4%) $\rightarrow$ Normalizing `blk_mean` (97.1%) $\rightarrow$ +`sp_dist_to_fail` (94.2%) $\rightarrow$ +`feature_97` (95.5%)

### Die 5: Marginal Threshold Case -- Wafer W_F_0016, Row 21, Col 12

**Standardized Explanation:**
> **Die (21,12)**: failure probability **52.1%**, confidence **Moderate Confidence** (ensemble $\sigma = 0.079$). Main reasons: 1) `blk_mean` (SHAP +1.4180), 2) `sp_dist_to_fail` (SHAP -1.0596), 3) `blk_q75` (SHAP +0.2558). Resembles failure signature cluster -1 (Mixed / Boundary Pattern). Counterfactual: normalizing `blk_mean` would reduce risk to **18.3%** (normalizing all top-3 reduces risk to **13.5%**).

- **Counterfactual Probability Trajectory**: Original (52.1%) $\rightarrow$ Normalizing `blk_mean` (18.3%) $\rightarrow$ +`sp_dist_to_fail` (18.3%) $\rightarrow$ +`blk_q75` (13.5%)
