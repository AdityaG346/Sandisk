# SanDisk Cerebrum 2026 — Final Pitch Script & Live Demo Guide
## Team: Cookies of the Dark Web
**Members**: Shrey Sharma, Aditya Gupta, Prisha Kushwaha, Darsh Ramoliya  
**Project**: Semiconductor Die Yield Prediction  
**Pitch Structure**: Exactly 10 Minutes Total — 6 Minutes Slides (4 Speakers) + 4 Minutes Live Website Demo (Speaker 4)  
**Judging Criteria Alignment**: Innovation & Originality, Technical Execution, Design & UX, Business Value & Viability, Presentation & Pitch  
**Open Source Repo**: [https://github.com/AdityaG346/Sandisk](https://github.com/AdityaG346/Sandisk)  
**Live Production App**: [https://die-yield-prediction.streamlit.app/](https://die-yield-prediction.streamlit.app/)  

---

### Pitch Timing Architecture & Speaker Breakdown

| Section | Speaker | Role & Focus | Slide / Asset | Target Time | Word Count |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Part 1** | **Speaker 1** | Pitch Opening, Problem Definition & Dual Architecture | Slides 1 & 2 | **0:00 – 1:30 (1.5 min)** | ~205 words |
| **Part 2** | **Speaker 2** | Production Benchmark Results & Statistical Rigor Audit | Slide 3 | **1:30 – 3:00 (1.5 min)** | ~200 words |
| **Part 3** | **Speaker 3** | Interpretability, Wafer Risk Fields & Fab Economics | Slides 4 & 5 | **3:00 – 4:30 (1.5 min)** | ~205 words |
| **Part 4** | **Speaker 4** | Engineering Boundaries, Open Links & Slide Close | Slide 6 | **4:30 – 6:00 (1.5 min)** | ~185 words |
| **Part 5** | **Speaker 4** | Interactive Live Dashboard Demonstration | Live Web Demo | **6:00 – 10:00 (4.0 min)** | ~510 words |
| **Total** | **All 4** | **Unified Presentation Package** | **Deck + App** | **10:00 Exactly** | **~1,305 words** |

---

## DELIVERABLE 2: Slide Deck Speaker Script (0:00 – 6:00)

### Speaker 1: Problem Definition & Architectural Strategy (Slides 1 & 2)
**Allocated Time**: 0:00 – 1:30 (~90 seconds | Word Count: 206 words)  
**Slides Active**: Slide 1 (Title) $\rightarrow$ Slide 2 (Problem & Approach)  

*(Slide 1: Title)*  
**[0:00 – 0:25 | Slide 1]**  
"Good afternoon, judges and fellow engineers. We are Team Cookies of the Dark Web—Shrey Sharma, Aditya Gupta, Prisha Kushwaha, and Darsh Ramoliya. Today, we are presenting our end-to-end production framework for Semiconductor Die Yield Prediction at SanDisk Cerebrum 2026. 

In high-volume silicon manufacturing, dies undergo stressful post-burn-in testing to filter out latent physical flaws before packaging. But physical testing is slow, expensive, and factory-constrained. Our mission is to predict post-burn-in defects early, intercepting faulty silicon before it ever reaches assembly."

*(Advance to Slide 2: Problem & Approach)*  
**[0:25 – 1:30 | Slide 2]**  
"To solve this, we formulated the prediction task with zero tolerance for data leakage. Looking at forty holdout test wafers comprising thirty-nine thousand total dies, sixty-seven hundred are known pre-test dead chips—meaning they are excluded by definition. That leaves exactly thirty-two thousand five hundred and ninety-eight eligible candidates, with a true post-test failure rate of 4.23 percent. Critically, sixty-five percent of those failures are marginal defects whose electrical sensor readings heavily overlap with healthy chips.

To tackle this challenge, we developed two hierarchical models evaluated on wafer-disjoint splits. Model A deploys five hundred parametric probe sensors, combined with ten leakage-free spatial features—such as distance to defect clusters and wafer edge proximity—plus an Isolation Forest anomaly detector, totaling 511 features. Model B builds directly upon Model A, adding twenty sub-die block features including eight summary statistics and a robust MAD-based anomaly score extracted from two thousand sub-die memory block readings per chip, totaling 531 features.

Now, my teammate Speaker 2 will walk you through the rigorous benchmark audit comparing these two systems."

---

### Speaker 2: Production Benchmark Results & Statistical Rigor (Slide 3)
**Allocated Time**: 1:30 – 3:00 (~90 seconds | Word Count: 198 words)  
**Slide Active**: Slide 3 (Benchmark Results & Key Statistical Finding)  

*(Slide 3: Results)*  
**[1:30 – 3:00 | Slide 3]**  
"Thank you. Looking at Slide 3, we evaluated both production models across forty holdout test wafers and conducted a five-seed paired ablation alongside a wafer-cluster bootstrap to separate genuine engineering gains from split noise.

Here is the central scientific finding of our audit: Adding sub-die block telemetry yields a statistically real, significant improvement in continuous probability ranking. Model B boosts PR-AUC from 0.5025 to 0.5362—a 6.7 percent relative gain (+0.0337) with a p-value under 0.001, and a 95 percent bootstrap confidence interval strictly bounded above zero.

However, when thresholded for hard binary classification, Fail F1 remains essentially unchanged—0.5207 for Model A versus 0.5222 for Model B. The delta is only plus 0.0015, and the 95 percent wafer-cluster bootstrap confidence interval spans across zero.

Why? Because sixty-five percent of failures are marginal cases by design. As shown in our TreeSHAP summary on the right, spatial proximity to defect clusters dominates overall attribution, while block features provide smooth secondary ranking signals that refine risk separation without altering thresholded boundaries.

Next, Speaker 3 will explain how we make these predictions interpretable and operational in a production fab."

---

### Speaker 3: Interpretability, Risk Fields & Fab Economics (Slides 4 & 5)
**Allocated Time**: 3:00 – 4:30 (~90 seconds | Word Count: 204 words)  
**Slides Active**: Slide 4 (Spatial Risk Fields) $\rightarrow$ Slide 5 (Operational Triage & Reliability)  

*(Slide 4: Interpretability & Risk Fields)*  
**[3:00 – 3:45 | Slide 4]**  
"Thank you. In a commercial fabrication plant, engineers will never deploy a black-box model. Slide 4 illustrates our visual interpretability architecture. 

Rather than viewing raw dies as isolated discrete points, we apply 2D Gaussian kernel smoothing with sigma equal to 1.5 across the wafer grid. As shown on Wafer W_F_0014, this transforms discrete failure predictions into continuous thermal risk surfaces, automatically overlaying 90th-percentile cyan hazard contours that delineate physical defect boundaries. 

Crucially, our system enforces a strict distinction: spatial features provide context, not an intervention. Actionable signal lies in die parametric sensors and sub-die block readings. On our live dashboard, you can see benchmark dies like W_N_0066 die (2,13), where Model A predicted a safe 12.9% raw score (4.5% calibrated), but Model B caught a 62.9% raw score (45.2% calibrated) post-burn-in failure because of elevated sub-die block variance."

*(Advance to Slide 5: Operational Triage & Calibration)*  
**[3:45 – 4:30 | Slide 5]**  
"Now, let's translate this into operational triage and reliability on Slide 5. 

First: Bounded screening triage. In real fabs, inspection capacity is strictly limited. At the same 10 percent inspection budget, Model B captures 66 more failures than Model A—834 versus 768 failures, or 60.4 percent versus 55.7 percent of all true defects—without increasing the inspection budget.

Second: Reliable probability calibration. Using validation-only Platt scaling fitted strictly across 5 cross-validation folds with zero test data leakage, Model A's Expected Calibration Error drops from 0.0336 to 0.0084—a 75 percent reduction—and Model B drops from 0.0293 to 0.0085.

Third: High precision and scrap preservation. Model B achieves 88.12 percent precision, with only 69 false alarms out of 31,218 healthy dies across 40 test wafers—a false alarm rate under 0.22 percent, while Model A achieves 97.04 percent precision with just 15 false alarms.

Finally, sub-second inference: all thirty-nine thousand test dies are scored in under 0.8 seconds on a single CPU thread.

Now, Speaker 4 will present our engineering boundaries and launch the live interactive demonstration."

---

### Speaker 4: Engineering Boundaries, Links & Pitch Close (Slide 6)
**Allocated Time**: 4:30 – 6:00 (~90 seconds | Word Count: 182 words)  
**Slide Active**: Slide 6 (Limitations, Horizons & Links)  

*(Slide 6: Limitations & Links)*  
**[4:30 – 6:00 | Slide 6]**  
"Thank you. True engineering excellence demands honest transparency regarding boundary conditions.

On Slide 6, we highlight three core findings from our technical audit. First, synthetic data limitations: synthetic defect generators produce cleaner spatial boundaries than real sub-micron fab noise. Second, empirical marginal-failure limitations: a large fraction of failures (65% in this semi-synthetic design) are marginal and overlap healthy electrical distributions, making binary classification difficult; observed F1 remains around 0.52. Third, block coordinate representation: block telemetry was provided as flattened 1D arrays; preserving true 3D memory stack coordinates in future revisions will unlock 3D convolutional neural networks for word-line leakage detection.

Everything we built is completely open-source and fully reproducible in our GitHub repository at `github.com/AdityaG346/Sandisk`, featuring automated index-alignment assertions and a single-command master pipeline.

Most importantly, we have deployed our complete production model and diagnostic suite live on the web at `die-yield-prediction.streamlit.app`. 

I will now transition directly from slides to our live production web application to demonstrate our system in action."

---

## DELIVERABLE 3: Live Website Demonstration Script (6:00 – 10:00)

**Speaker**: Speaker 4  
**Duration**: Exactly 4 Minutes (240 Seconds | Word Count: 508 words)  
**URL**: [https://die-yield-prediction.streamlit.app/](https://die-yield-prediction.streamlit.app/)  
**Interactive Targets**:
1. Wafer Selector: Select `W_F_0014`
2. 4-Panel Wafer View & Risk-Field Hotspot Contours
3. Radio Toggle: `Side-by-side Comparison` (Model A vs Model B)
4. Die Selector: Inspect `Die (40, 18)` (Probability, SHAP, Failure Signature, Counterfactual)
5. Sub-Die Block Signal Strip Plot & Sequential Index Caveat
6. Final Wrap-Up & Sign-Off

---

### Phase 1: Application Launch & 4-Panel Wafer Overview (6:00 – 6:55)
* **Action**: Switch screen to browser displaying `https://die-yield-prediction.streamlit.app/`. Ensure sidebar dropdown has Wafer `W_F_0014` selected (the default).  
* **Time**: 0:00 – 0:55 of demo (55 seconds | Word Count: ~118 words)  

**Spoken Script**:  
"We are now live on our deployed production dashboard at `die-yield-prediction.streamlit.app`. In the sidebar, you can see all forty holdout test wafers available for real-time inspection. We have selected benchmark Wafer `W_F_0014`. 

Notice the five KPI summary cards across the top: this wafer contains 1,024 total dies, 108 known pre-test failures, 916 eligible candidate dies, and 38 actual post-burn-in failures.

Directly below is our primary four-panel spatial inspector. Panel 1 displays the pre-test status, with known defective chips in red. Panel 2 shows ground-truth post-burn-in failures. Panel 3 displays our LightGBM predicted failure probabilities. And Panel 4 renders our continuous 2D Gaussian risk field, where cyan contours automatically isolate 90th-percentile hazard hotspots encircling the true defect clusters."

---

### Phase 2: Side-by-Side Model A vs Model B Comparison (6:55 – 7:50)
* **Action**: In the sidebar under 'Model Selection', click the radio button for **'Side-by-side Comparison'**. Allow both 4-panel figures to render on screen.  
* **Time**: 0:55 – 1:50 of demo (55 seconds | Word Count: ~114 words)  

**Spoken Script**:  
"Now, let's observe the concrete difference between Model A and Model B in production. In the sidebar, I am toggling 'Side-by-side Comparison'.

On the top row is Model A, utilizing 511 macro electrical, spatial context, and die anomaly features. On the bottom row is Model B, augmented with 531 features including 20 sub-die block telemetry signals.

Look closely at Panel 3 and Panel 4: Model A relies heavily on radial distance and known defect clusters, creating broader, coarser probability boundaries. Model B incorporates internal block summary statistics and robust MAD anomaly signals. This sharpens probability separation along the wafer perimeter, pulling high-risk dies into sharper thermal focus and directly explaining the statistically significant PR-AUC gain (+0.0337) we proved earlier."

---

### Phase 3: Die (40, 18) Diagnostics, SHAP Breakdown & Counterfactuals (7:50 – 8:45)
* **Action**: In the sidebar, switch radio back to **'Model B (Full Diagnostic)'**. Scroll down to Section 2: 'Die-Level Diagnostic Inspector'. Select **'Row 40, Col 18'** from the dropdown (pre-selected by default as a benchmark die).  
* **Time**: 1:50 – 2:45 of demo (55 seconds | Word Count: ~122 words)  

**Spoken Script**:  
"Returning to Model B, let's zoom in from wafer-scale geography to an individual chip. We select Die Row 40, Column 18, located on the outer edge of Wafer W_F_0014.

As shown in our metric cards, the model flags this die with a 90.50 percent failure probability against our tuned 0.5180 threshold—and ground truth confirms it indeed failed post-test.

Looking at the SHAP attribution bar chart, the warning is driven by proximity to pre-test defects, elevated block mean, and high 75th-percentile voltage dispersion. Our clustering engine classifies this chip into Cluster 2: Defect Neighborhood Proximity and Spatial Clustering. 

Furthermore, our model counterfactual sensitivity trajectory shows that normalizing sub-die memory block deviations reduces failure probability from 90.5 percent down to 76.2 percent."

---

### Phase 4: Sub-Die Block Signal Profile & The Index Caveat (8:45 – 9:35)
* **Action**: Scroll down to Section 3: 'Sub-Die Block Signal Profile (Model B Feature View)'. Point cursor at the horizontal block strip plot.  
* **Time**: 2:45 – 3:35 of demo (50 seconds | Word Count: ~106 words)  

**Spoken Script**:  
"Scrolling down to Section 3, our dashboard opens the internal anatomy of Die (40, 18). 

This strip plot visualizes all two thousand sequential sub-die block voltage readings. The amber dashed lines indicate our robust threshold computed directly from the median plus or minus two times the Median Absolute Deviation. Every red dot marks a verified anomalous block exceeding this boundary. 

However, we provide a vital engineering caveat right on the x-axis: this plot reflects sequential stream index positions from zero to nineteen ninety-nine—not physical 2D or 3D coordinates in the silicon stack. True physical coordinates remain the key to future 3D convolutional models."

---

### Phase 5: Synthesis, Fab Impact & Final Pitch Sign-Off (9:35 – 10:00)
* **Action**: Scroll back up to the main dashboard header or comparison table. Stand up, face judges, and deliver final closing sentence.  
* **Time**: 3:35 – 4:00 of demo (25 seconds | Word Count: ~48 words)  

**Spoken Script**:  
"To summarize: by uniting leakage-free spatial risk fields, sub-die telemetry, and high-precision threshold calibration, our system intercepts 37.1 percent of silicon failures before packaging at 88.12 percent precision—saving factory bandwidth and protecting customer trust. 

Thank you, judges. We welcome your questions!"

---

### Technical Fact Verification & Citation Index
Every claim, metric, and finding in this pitch package traces directly to verified project artifacts:
1. **Eligible Die Count & Fail Rate**: Verified from raw data in `outputs/results_summary.md` Section 1 (32,598 eligible test dies, 1,380 fails = 4.2334%).
2. **Model A vs B Metrics**: Verified from `outputs/audited_numbers.json` (PR-AUC: 0.5025 vs 0.5362; F1: 0.5207 vs 0.5222; Recall: 35.58% vs 37.10%; Precision: 97.04% vs 88.12%).
3. **Statistical Significance**: Verified from `outputs/results_summary.md` and `outputs/test_bootstrap_ci.csv` (wafer-cluster bootstrap PR-AUC CI $[+0.023, +0.043]$ strictly positive; F1 CI $[−0.008, +0.012]$ crosses zero).
4. **Triage Capture**: Top 10% budget yields 55.7% (768 fails) for Model A vs 60.4% (834 fails) for Model B; at the same 10% inspection budget, Model B captures 66 more failures than Model A.
5. **Probability Calibration**: Validation Platt scaling reduces ECE by 74.9% (Model A: 0.0336 -> 0.0084) and 71.0% (Model B: 0.0293 -> 0.0085).
6. **Die (40, 18) Diagnostics**: Verified from `outputs/per_die_explanations.md` and `dashboard/app.py` (90.5% prob, $\sigma = 0.101$, Cluster 2, counterfactual 76.2%).
7. **Robust MAD Threshold**: Verified from `src/block_features.py` and `dashboard/app.py` ($|x - \text{median}| > 2.0 \times \text{MAD}$).
