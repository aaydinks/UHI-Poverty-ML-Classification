# Pipeline diagram

`pipeline.py` draws the pipeline diagram in three columns (Datasets → Models → Validation). Its numbers
are read from the built sample (`median_values.json`) and the shared modules, so they cannot
drift from the analysis.

## What the diagram shows

- **Datasets.** The SUHI database (Chakraborty et al. 2020), the 2019 Gazetteer and
  cartographic boundaries on 2010 tracts, and the ACS 2015–2019 poverty rate (table B17001)
  feed one national analysis sample. The HUD Low Poverty Index is a robustness outcome only
  (dashed).
- **Models.** Poverty: ACS poverty rate ≥ the national median → `High_Pov`, on `B-env`.
  UHI: `UHI_annual_day` ≥ the national median → `High_UHI`, on `A-noLST`, which drops every
  `LST_*` and `UHI_*` column, because the target is computed from LST (section 3.1). The
  majority class, logistic regression and LightGBM run with settings fixed in advance and
  τ = 0.5. They are compared, not blended, and no threshold is tuned on test data.
- **Validation.** Leave-one-city-out is the headline: the fold grid holds urbanized areas, one
  of which is held out per fold. County- and UA-grouped 5-fold, UA-buffered state and region
  hold-outs and the 20 largest UAs are also reported.
- **Preprocessing.** Logistic regression gets median imputation, log1p on the skewed size
  columns and standardisation, fitted inside each fold; LightGBM gets the raw inputs.
- **Metrics.** The full panel, each with a 95 % UA cluster-bootstrap CI.
- **Interpretability.** Out-of-fold TreeSHAP for LightGBM under UA-grouped 5-fold,
  attributed to groups that keep each exact identity whole (section 5.2).
- **Tuned models (section 4.6).** Dashed and secondary: a nested Optuna search over XGBoost,
  LightGBM, CatBoost and a soft-voting ensemble, with τ chosen on training cities only. The
  diagram gives no numbers for them; they are in section 4.6.

## Numbers shown

| Element | Shown as | Source |
|---|---|---|
| Analysis sample: tracts | 54,243 | `data/processed/median_values.json` |
| Analysis sample: urbanized areas | 486 | `data/processed/median_values.json` |
| Poverty target cut (ACS poverty rate, national median) | 10.5% | `data/processed/median_values.json` |
| UHI target cut (UHI_annual_day, national median, °C) | 1.45 °C | `data/processed/median_values.json` |
| Predictors in B-env (poverty task) | 64 | `uhi_poverty.features.CONFIGS` |
| Predictors in A-noLST (UHI task) | 28 | `uhi_poverty.features.CONFIGS` |
| Decision threshold τ | 0.5 | `uhi_poverty.models.THRESHOLD` |
| Leave-one-city-out folds (one per UA) | 486 | `data/processed/median_values.json` |
| Training UAs per leave-one-city-out fold | 485 | `data/processed/median_values.json` |
| County- and UA-grouped folds | 5 | `uhi_poverty.validation.SCHEMES` |
| Largest UAs held out in turn | 20 | `uhi_poverty.validation.N_MAJOR_UA` |
| Cluster-bootstrap resamples of UAs | 1,000 | `uhi_poverty.metrics.B_DEFAULT` |
| Metrics in the panel | 10 | `uhi_poverty.metrics.PANEL` |
| TreeSHAP folds (UA-grouped) | 5 | `uhi_poverty.interpret` |
| ACS vintage | 2015–2019 | `data/README.md (S2)` |
| Gazetteer / boundary vintage | 2019 | `data/README.md (S4, S6)` |
| SUHI database reference year | 2020 | `data/README.md (S1)` |

All values: [`results/pipeline.csv`](results/pipeline.csv).
Figure: [`results/pipeline.png`](results/pipeline.png).
