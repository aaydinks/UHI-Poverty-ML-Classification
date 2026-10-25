# 4 Model specifications

How well the two classifiers generalise, and how they are evaluated: spatially grouped
validation, held-out states and cities, confusion matrices with denominators, a full metric
panel with uncertainty, baselines, and nested-tuned models. Sections 4.1–4.5 read the same
cached out-of-fold predictions (`src/uhi_poverty/validation.py`), so they all score the same
fitted models.

| Section | Folder | Script | Main outputs |
|---|---|---|---|
| 4.1 Spatially grouped validation | [`4.1_Spatially_Grouped_Validation/`](4.1_Spatially_Grouped_Validation/) | `grouped_validation.py` | pooled metrics for six grouped schemes; per-UA, per-state, per-region results |
| 4.2 Held-out states and major UAs | [`4.2_Holdout_States_and_Major_Urban_Areas/`](4.2_Holdout_States_and_Major_Urban_Areas/) | `holdout_vs_random.py` | optimism of random folds with paired CIs; Moran's I; train-neighbour share |
| 4.3 Confusion matrices and denominators | [`4.3_Confusion_Matrices_and_Denominators/`](4.3_Confusion_Matrices_and_Denominators/) | `confusion_matrices.py` | [`DEFINITIONS.md`](4.3_Confusion_Matrices_and_Denominators/DEFINITIONS.md); counts and rates with denominators |
| 4.4 Metric panel with uncertainty | [`4.4_Metric_Panel_with_Uncertainty/`](4.4_Metric_Panel_with_Uncertainty/) | `metric_panel.py` | every metric × model × task × scheme with UA cluster-bootstrap CIs; forest plot |
| 4.5 Baselines | [`4.5_Baselines/`](4.5_Baselines/) | `baselines.py` | majority, logistic, LightGBM and reference baselines; paired differences |
| 4.6 Tuned models and ensemble | [`4.6_Tuned_Models_and_Ensemble/`](4.6_Tuned_Models_and_Ensemble/) | `tuned_models.py` | logistic regression, LightGBM, XGBoost and CatBoost tuned in nested UA-grouped cross-validation, and a soft-voting ensemble, with thresholds chosen on training cities |

## Shared specification

- **Tasks.** Task A predicts `High_UHI` from `A-noLST` (no LST or SUHI input; section 3.2).
  Task B predicts `High_Pov` (ACS poverty rate ≥ national median) from `B-env` (all SUHI
  columns plus tract geometry).
- **Models.** Majority class; logistic regression (median imputation → `log1p` on area and
  pixel counts → standardisation → `C = 1`); LightGBM with fixed settings
  (`n_estimators = 400`, `learning_rate = 0.05`, `num_leaves = 31`, `min_child_samples = 40`,
  row and column subsampling 0.8, `reg_lambda = 1`). No hyperparameter is tuned in 4.1–4.5;
  section 4.6 adds tuned models beside them, with every choice made in nested cross-validation.
- **No leakage through the pipeline.** Every learned step, imputation and scaling included, is
  fitted inside the training fold. The decision threshold is fixed at τ = 0.5 and never chosen
  on evaluation data; threshold-free metrics (ROC-AUC, PR-AUC) are reported alongside.
- **Validation.** Leave-one-city-out is the headline; county- and UA-grouped 5-fold,
  UA-buffered leave-one-state-out and leave-one-region-out, and the 20 largest UAs held out are
  reported too, with random 5-fold as a reference only.
- **Uncertainty.** 95 % percentile intervals from a bootstrap that resamples urbanized areas
  (B = 1,000, seed 37); paired differences reuse the same draws.

Run with `make engine` (fills the prediction cache, about 15 minutes on 15 cores) and then
`make models`. Section 4.6 runs with `make tune tuned`; `make tune` fills the tuned-model
cache (`data/interim/tuned/`, a few hours on 15 cores).
