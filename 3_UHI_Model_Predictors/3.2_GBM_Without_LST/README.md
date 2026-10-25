# 3.2 Gradient-boosted model without LST

## Purpose

This section fits the UHI model without the target's components or their close proxies, and
shows, through an ablation ladder and an LST-only baseline, what each family of inputs adds.

## Method

- `gbm_no_lst.py` trains LightGBM (and logistic regression) on **`A-noLST`**: vegetation
  (NDVI), urban–rural vegetation contrast (ΔNDVI), elevation (DEM), tract morphology and
  tract geometry. It contains **no `LST_*` column and no `UHI_*` column**, because every
  `UHI_*` column is an LST difference (section 3.1).
- Evaluation: leave-one-city-out (primary, shared with section 4), UA-grouped 5-fold, and random
  5-fold for comparison only. Fixed hyperparameters, τ = 0.5, every learned step fitted inside
  the training folds.
- Ablation ladder, each rung removing only what its name says (the two geometry columns are
  in every rung except the LST-only baseline): `A-full` → `A-narrow` → `A-noLST-keepUHI` →
  `A-noLST` → `A-noLST+pov`, plus `A-lst-only`.
- [`results/feature_lists.csv`](results/feature_lists.csv) lists every predictor of every
  configuration in this repository, with its family.

## Results

LightGBM, Task A (`High_UHI`), leave-one-city-out unless stated
([`results/ablation.md`](results/ablation.md) has logistic regression, UA-grouped 5-fold, F1,
MCC, PR-AUC and balanced accuracy with CIs):

| Configuration | Predictors | Contents | ROC-AUC, random 5-fold | ROC-AUC, leave-one-city-out [95 % CI] | F1 | MCC |
|---|---:|---|---:|---:|---:|---:|
| `A-full` | 63 | All 61 non-target SUHI columns + geometry (reference; contains the target's terms) | 0.999 | 0.998 [0.997, 0.998] | 0.974 | 0.948 |
| `A-narrow` | 61 | A-full minus LST_urb_day_CT and LST_rur_day | 0.999 | 0.997 [0.996, 0.997] | 0.969 | 0.938 |
| `A-noLST-keepUHI` | 39 | A-full minus all 24 LST_* columns (UHI_* still present) | 0.997 | 0.995 [0.995, 0.996] | 0.964 | 0.928 |
| `A-noLST` | 28 | NDVI, dNDVI, DEM, morphology, geometry; no LST_* or UHI_* (primary Task A) | 0.946 | 0.910 [0.888, 0.929] | 0.819 | 0.647 |
| `A-noLST+pov` | 29 | A-noLST + ACS poverty rate | 0.947 | 0.912 [0.892, 0.930] | 0.822 | 0.652 |
| `A-lst-only` | 1 | LST_urb_day_CT alone (LST-only baseline) | 0.722 | 0.619 [0.536, 0.696] | 0.630 | 0.221 |

- With the target's own components (`A-full`) the model is close to perfect, because it can
  subtract `LST_rur_day` from `LST_urb_day_CT`.
- Dropping just those two columns (`A-narrow`) barely changes anything: the other `LST_*` and
  `UHI_*` columns still encode the target.
- Dropping all 24 `LST_*` columns but keeping the `UHI_*` columns (`A-noLST-keepUHI`) still
  leaks, because the seasonal and night-time SUHI columns are differences built from the same
  temperatures.
- **`A-noLST` is the leak-free model**: ROC-AUC 0.910 [0.888, 0.929] for
  LightGBM and 0.899 [0.884, 0.914] for logistic regression in held-out
  cities. It measures how well high surface urban heat can be predicted from vegetation,
  elevation and built form alone.
- Adding the ACS poverty rate (`A-noLST+pov`) changes leave-one-city-out ROC-AUC from
  0.910 to
  0.912.

**Predictors of `A-noLST`** (28):

- **NDVI** (12): `NDVI_rur`, `NDVI_rur_summer`, `NDVI_rur_winter`, `NDVI_urb_CT`, `NDVI_urb_CT_act`, `NDVI_urb_CT_act_summer`, `NDVI_urb_CT_act_winter`, `NDVI_urb_CT_summer`, `NDVI_urb_CT_winter`, `NDVI_urb_all`, `NDVI_urb_all_summer`, `NDVI_urb_all_winter`
- **DeltaNDVI** (6): `DelNDVI_annual`, `DelNDVI_summer`, `DelNDVI_winter`, `DelNDVI_annual_city`, `DelNDVI_summer_city`, `DelNDVI_winter_city`
- **DEM** (5): `DEM_rur`, `DEM_urb_CT`, `DEM_urb_CT_act`, `DEM_urb_all`, `DelDEM`
- **Morphology** (3): `Area`, `Total_Count`, `Urban_Count`
- **Geometry** (2): `water_frac`, `log_aland`

Figure: [`results/ablation.png`](results/ablation.png).
