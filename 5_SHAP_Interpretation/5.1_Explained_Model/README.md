# 5.1 The explained model

## Purpose

States which model the SHAP values explain, on what scale they are reported, and how each
attribution maps back to a named input column.

## Method

- The model explained is the model evaluated in sections 3 and 4: LightGBM with fixed settings
  (`n_estimators=400`, `learning_rate=0.05`,
  `num_leaves=31`, seed 37), fitted directly on the input
  columns, so every attribution belongs to a named input column.
- `src/uhi_poverty/interpret.py` refits that model on each training fold of the UA-grouped
  5-fold split and computes LightGBM's exact TreeSHAP values (`pred_contrib=True`) for the
  held-out fold only. No tract is explained by a model that saw its urbanized area.
- The values are in **log-odds** of the positive class. For each tract, the bias term plus the
  sum of its SHAP values equals the logit of its out-of-fold score. `explained_model.py` checks
  this against the out-of-fold predictions cached by `validation.oof()`, the same predictions
  behind the UA-grouped results in section 4
  ([`results/additivity.md`](results/additivity.md)).
- Columns linked by an exact identity (section 3.1: every `UHI_*` is an urban-minus-rural LST
  difference, every `DelNDVI_*` an NDVI difference, `DelDEM` a DEM difference) split credit
  arbitrarily among themselves. Attributions are therefore also reported for groups that keep
  each identity whole ([`results/attribution_groups.md`](results/attribution_groups.md)). The
  tract and city SUHI variants share their rural reference, so heat separates by season and
  time of day but not into tract versus city, or SUHI versus LST.

## Results

| Task | Configuration | Predictors | Tracts explained | Max error, probability | Max error, logit |
|---|---|---:|---:|---:|---:|
| A | `A-noLST` | 28 | 54,243 | 7e-15 | 3e-14 |
| B | `B-env` | 64 | 54,243 | 3e-15 | 2e-14 |

The attributions reproduce the evaluated model's predictions to floating-point precision.
UA-grouped 5-fold is used here because the stability checks in 5.2 need several fold models
for every tract; its LightGBM ROC-AUC (0.908 on Task A,
0.772 on Task B) is within 0.01 of leave-one-city-out
(0.910 and 0.777; section 4.5).

Attribution groups:

| Task | Group | Columns | Exact identities inside |
|---|---|---:|---:|
| A | Vegetation: annual | 6 | 2 |
| A | Vegetation: summer | 6 | 2 |
| A | Vegetation: winter | 6 | 2 |
| A | Elevation | 5 | 1 |
| A | Tract size | 4 | 0 |
| A | Water share | 1 | 0 |
| B | Heat: annual day | 6 | 2 |
| B | Heat: annual night | 6 | 2 |
| B | Heat: summer day | 6 | 2 |
| B | Heat: summer night | 6 | 2 |
| B | Heat: winter day | 6 | 2 |
| B | Heat: winter night | 6 | 2 |
| B | Vegetation: annual | 6 | 2 |
| B | Vegetation: summer | 6 | 2 |
| B | Vegetation: winter | 6 | 2 |
| B | Elevation | 5 | 1 |
| B | Tract size | 4 | 0 |
| B | Water share | 1 | 0 |
