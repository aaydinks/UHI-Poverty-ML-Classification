# 4.6 Tuned models and the soft-voting ensemble

## Purpose

Tests whether a fully tuned pipeline (scaling, PCA, SMOTE, variance and collinearity filters,
engineered terms, tuned XGBoost, LightGBM and CatBoost, a soft-voting ensemble of the three,
and threshold optimisation) raises accuracy and F1 in held-out cities, against targets of 0.90
for high SUHI and 0.75 for high poverty. Sections 4.1–4.5 use fixed settings and τ = 0.5,
declared in advance; the tuned models sit beside them and change none of their results.

## Method

`src/uhi_poverty/tuning.py` assembles the pipeline from its parts, and inner cross-validation
chooses every option:

| Step | Implementation | Search options |
|---|---|---|
| Scaling | `MinMaxScaler` / `StandardScaler` | none, min-max, standard |
| Low-variance filter | `VarianceThreshold(0.1)` | on, off |
| Collinearity filter | drop the later column of each pair with \|r\| > 0.9 | on, off |
| PCA | PCA after imputation and scaling | off, 95 % or 99 % of variance |
| Class rebalancing | `imblearn` SMOTE inside the training fold | on, off |
| Engineered terms | Task B: 9 seasonal, day-night, city-contrast, squared and interaction terms of UHI; Task A: the same terms on the vegetation difference (no temperature) | on, off |
| Models | `XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier`, plus logistic regression | model parameters (Optuna TPE) |
| Hyperparameter search | Optuna, 60 trials for LightGBM and XGBoost, 40 for CatBoost, 40 for logistic regression, per outer fold | objective: pooled inner ROC-AUC |
| Soft-voting ensemble | mean of the three tuned boosters' probabilities | equal weights; weights on a 0.05 grid chosen on inner predictions |
| Decision threshold | a 99-point grid on **inner** out-of-fold predictions | τ_acc (best accuracy), τ_F1 (best F1) |

The interaction terms multiply `UHI_annual_day` by `Area`, `DEM_rur` and `DEM_urb_CT`, the
first three numeric non-identifier columns of the SUHI file. Task A never sees an `LST_*` or
`UHI_*` column (section 3.2), so its engineered terms are built on `DelNDVI` instead; there
is no day/night vegetation, so the two day-night terms have no Task A analogue.

**Nesting.** For each of the five UA-grouped outer folds, an inner UA-grouped 5-fold over the
training cities scores every trial. The best trial's options and parameters, and the τ_acc and
τ_F1 that maximise accuracy and F1 on its inner out-of-fold predictions, are that fold's
settings. Under leave-one-city-out, each city is predicted by a model fitted on all other
cities with the settings of the outer fold that held it out, which were chosen without it.
Random 5-fold repeats the procedure with random folds, as a reference only.
`tests/test_tuning.py` checks that no tuning step sees an outer test row.

**SMOTE.** The median split leaves the classes balanced. The positive share of the training
cities ranges from 0.481 to 0.504 (Task A) and
0.496 to 0.505 (Task B) across leave-one-city-out
folds, so SMOTE has almost nothing to rebalance.

## Results

Leave-one-city-out, the headline (estimate [95 % UA cluster-bootstrap CI]). Tuned models are
scored at τ_acc for accuracy and ROC-AUC and at τ_F1 for F1; fixed models at τ = 0.5. Every
threshold and scheme is in [`results/tuned_panel.md`](results/tuned_panel.md):

| Task | Model | Accuracy | F1 | ROC-AUC | Accuracy, UA-grouped 5-fold | Accuracy, random 5-fold (reference) |
|---|---|---:|---:|---:|---:|---:|
| A | Logistic regression, fixed (4.4) | 0.830 [0.815, 0.845] | 0.834 [0.809, 0.853] | 0.899 [0.884, 0.914] | 0.829 | 0.838 |
| A | LightGBM, fixed (4.4) | 0.823 [0.780, 0.855] | 0.819 [0.768, 0.856] | 0.910 [0.888, 0.929] | 0.823 | 0.874 |
| A | Logistic regression, tuned | 0.838 [0.826, 0.849] | 0.843 [0.817, 0.861] | 0.902 [0.886, 0.915] | 0.835 | 0.841 |
| A | LightGBM, tuned | 0.834 [0.809, 0.855] | 0.840 [0.816, 0.859] | 0.910 [0.893, 0.928] | 0.834 | 0.875 |
| A | XGBoost, tuned | 0.834 [0.805, 0.856] | 0.846 [0.824, 0.863] | 0.911 [0.893, 0.928] | 0.828 | 0.875 |
| A | CatBoost, tuned | 0.833 [0.805, 0.855] | 0.840 [0.813, 0.862] | 0.911 [0.893, 0.929] | 0.840 | 0.874 |
| A | Soft-voting ensemble, equal weights | 0.833 [0.801, 0.857] | 0.844 [0.819, 0.864] | 0.913 [0.895, 0.930] | 0.838 | 0.877 |
| A | Soft-voting ensemble, tuned weights | 0.835 [0.808, 0.856] | 0.844 [0.820, 0.862] | 0.913 [0.895, 0.930] | 0.837 | 0.877 |
| B | Logistic regression, fixed (4.4) | 0.665 [0.636, 0.690] | 0.666 [0.620, 0.696] | 0.714 [0.670, 0.752] | 0.662 | 0.695 |
| B | LightGBM, fixed (4.4) | 0.705 [0.691, 0.720] | 0.725 [0.707, 0.742] | 0.777 [0.758, 0.796] | 0.698 | 0.757 |
| B | Logistic regression, tuned | 0.659 [0.633, 0.681] | 0.688 [0.651, 0.715] | 0.701 [0.655, 0.742] | 0.659 | 0.695 |
| B | LightGBM, tuned | 0.703 [0.684, 0.720] | 0.734 [0.719, 0.747] | 0.776 [0.758, 0.793] | 0.700 | 0.762 |
| B | XGBoost, tuned | 0.703 [0.687, 0.719] | 0.734 [0.719, 0.748] | 0.777 [0.758, 0.795] | 0.699 | 0.764 |
| B | CatBoost, tuned | 0.704 [0.687, 0.722] | 0.735 [0.720, 0.748] | 0.776 [0.760, 0.793] | 0.699 | 0.758 |
| B | Soft-voting ensemble, equal weights | 0.706 [0.691, 0.722] | 0.737 [0.722, 0.750] | 0.779 [0.762, 0.796] | 0.702 | 0.766 |
| B | Soft-voting ensemble, tuned weights | 0.706 [0.691, 0.723] | 0.736 [0.722, 0.750] | 0.779 [0.761, 0.796] | 0.701 | 0.766 |

**Targets.**

- **High SUHI (Task A), target 0.90.** Best leave-one-city-out accuracy 0.838 (Logistic regression, tuned): not met. Best F1 0.846 (XGBoost, tuned): not met. Under random 5-fold, which leaves almost every tract beside a training tract (section 4.2), the best accuracy is 0.877 (Soft-voting ensemble, equal weights).
- **High poverty (Task B), target 0.75.** Best leave-one-city-out accuracy 0.706 (Soft-voting ensemble, tuned weights): not met. Best F1 0.737 (Soft-voting ensemble, equal weights): not met. Under random 5-fold, which leaves almost every tract beside a training tract (section 4.2), the best accuracy is 0.766 (Soft-voting ensemble, tuned weights).

**Against the fixed LightGBM** (paired, leave-one-city-out;
[`results/paired_deltas.md`](results/paired_deltas.md)):

| Task | Tuned model | Δ accuracy | Δ ROC-AUC | Δ F1 (τ_F1) |
|---|---|---:|---:|---:|
| A | Logistic regression, tuned | 0.015 [-0.010, 0.056] | -0.008 [-0.020, 0.006] | 0.023 [-0.010, 0.076] |
| A | LightGBM, tuned | 0.011 [-0.002, 0.032] | 0.001 [-0.005, 0.006] | 0.021 [-0.001, 0.057] |
| A | XGBoost, tuned | 0.010 [-0.002, 0.029] | 0.001 [-0.004, 0.008] | 0.026 [0.002, 0.068] |
| A | CatBoost, tuned | 0.010 [-0.002, 0.028] | 0.002 [-0.003, 0.007] | 0.021 [0.002, 0.051] |
| A | Soft-voting ensemble, equal weights | 0.010 [0.000, 0.023] | 0.004 [-0.002, 0.009] | 0.025 [0.004, 0.060] |
| A | Soft-voting ensemble, tuned weights | 0.012 [-0.001, 0.031] | 0.003 [-0.002, 0.009] | 0.024 [0.001, 0.063] |
| B | Logistic regression, tuned | -0.046 [-0.079, -0.021] | -0.077 [-0.132, -0.030] | -0.037 [-0.086, -0.003] |
| B | LightGBM, tuned | -0.002 [-0.009, 0.005] | -0.001 [-0.005, 0.002] | 0.008 [0.001, 0.017] |
| B | XGBoost, tuned | -0.002 [-0.007, 0.003] | -0.001 [-0.003, 0.003] | 0.008 [0.002, 0.017] |
| B | CatBoost, tuned | -0.001 [-0.008, 0.006] | -0.001 [-0.007, 0.004] | 0.009 [0.001, 0.020] |
| B | Soft-voting ensemble, equal weights | 0.001 [-0.003, 0.006] | 0.002 [-0.001, 0.005] | 0.011 [0.003, 0.022] |
| B | Soft-voting ensemble, tuned weights | 0.001 [-0.004, 0.007] | 0.002 [-0.001, 0.005] | 0.011 [0.003, 0.021] |

Tuned models whose accuracy or ROC-AUC gain has a 95 % interval above zero: Task A,
none; Task B, none. Gains in F1 at τ_F1 do have intervals
above zero for Task A (XGBoost, tuned; CatBoost, tuned; Soft-voting ensemble, equal weights; Soft-voting ensemble, tuned weights) and Task B
(LightGBM, tuned; XGBoost, tuned; CatBoost, tuned; Soft-voting ensemble, equal weights; Soft-voting ensemble, tuned weights): most of them come from moving the threshold, which trades
specificity for recall, rather than from better ranking. The best leave-one-city-out accuracy
moves from 0.823 to 0.838 for Task A (Logistic regression, tuned) and from 0.705 to
0.706 for Task B (Soft-voting ensemble, tuned weights).

**The ensemble against its members** (Δ accuracy, leave-one-city-out):

| Task | Ensemble | − LightGBM | − XGBoost | − CatBoost |
|---|---|---|---|---|
| A | Soft-voting ensemble, equal weights | -0.001 [-0.009, 0.004] | -0.001 [-0.006, 0.002] | -0.000 [-0.005, 0.003] |
| A | Soft-voting ensemble, tuned weights | 0.001 [-0.002, 0.004] | 0.001 [-0.000, 0.003] | 0.002 [-0.001, 0.004] |
| B | Soft-voting ensemble, equal weights | 0.004 [0.000, 0.007] | 0.003 [0.001, 0.006] | 0.002 [-0.002, 0.007] |
| B | Soft-voting ensemble, tuned weights | 0.004 [0.000, 0.007] | 0.003 [0.001, 0.006] | 0.002 [-0.003, 0.008] |

An ensemble beats all three of its members with a 95 % interval above zero for Task A:
none; for Task B: none.

**What the inner cross-validation chose** ([`results/search_choices.md`](results/search_choices.md),
[`results/mechanisms.md`](results/mechanisms.md)). Each cell gives, for each option value, the
best inner ROC-AUC among trials using it (mean over boosters and folds), and in brackets how
many of the 15 booster × fold settings chose it:

| Task | Option | Value: best inner ROC-AUC (chosen) |
|---|---|---|
| A | Engineered terms | False: 0.9057 (3/15); True: 0.9128 (12/15) |
| A | Low-variance filter (0.1) | False: 0.9132 (15/15); True: 0.8890 (0/15) |
| A | Correlation filter (|r| > 0.9) | False: 0.9076 (7/15); True: 0.9102 (8/15) |
| A | Scaling | minmax: 0.9038 (6/15); none: 0.8856 (3/15); standard: 0.9084 (6/15) |
| A | PCA | 0.95: 0.8863 (0/15); 0.99: 0.8962 (4/15); none: 0.9084 (11/15) |
| A | SMOTE | False: 0.9088 (8/15); True: 0.9086 (7/15) |
| B | Engineered terms | False: 0.7620 (3/15); True: 0.7670 (12/15) |
| B | Low-variance filter (0.1) | False: 0.7723 (15/15); True: 0.7432 (0/15) |
| B | Correlation filter (|r| > 0.9) | False: 0.7685 (12/15); True: 0.7573 (3/15) |
| B | Scaling | minmax: 0.7606 (7/15); none: 0.7450 (3/15); standard: 0.7637 (5/15) |
| B | PCA | 0.95: 0.7360 (0/15); 0.99: 0.7472 (0/15); none: 0.7723 (15/15) |
| B | SMOTE | False: 0.7628 (5/15); True: 0.7697 (10/15) |

Across the 30 booster × fold settings of both tasks, PCA was chosen
4 times, SMOTE 17 times and the
engineered terms 24 times. The chosen τ_acc ranged from
0.41 to 0.61 and τ_F1 from
0.34 to 0.47.

**Why the targets are not reached.** Tuning moves the best leave-one-city-out accuracy by less than 0.02 on both tasks (0.015 for Task A, 0.001 for Task B). Leave-one-city-out ROC-AUC bounds what any threshold can reach, and none of the options raises it far above the fixed models.

Figure: [`results/tuned_vs_fixed.png`](results/tuned_vs_fixed.png).
