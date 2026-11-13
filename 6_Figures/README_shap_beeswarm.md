# SHAP beeswarm

The beeswarm shows the SHAP values of the top 10 features per task for two of the evaluated
models ([`results/shap_beeswarm.png`](results/shap_beeswarm.png)). The top row is LightGBM, a
gradient-boosted tree ensemble; the bottom row is logistic regression. Both use the fixed
settings of `models.py`.

- **Panels A and C, `High_UHI` without temperature input** (`A-noLST`): the models use no
  `LST_*` or `UHI_*` column, because the urban and rural LST terms define the label
  (section 3.1).
- **Panels B and D, `High_Pov` from the ACS poverty rate** (`B-env`): positive SHAP raises the
  predicted probability of *high* poverty.
- Both models are explained out of fold under the UA-grouped 5-fold split (seed
  37), on the same folds, on the national sample. Each panel shows
  3,000 randomly chosen tracts. Colour is the feature's percentile among all tracts.
  The x-scales differ between panels.
- **LightGBM** values are exact TreeSHAP in log-odds (section 5.1). `shap_beeswarm.py` reads
  the cached attributions and stops if its ranking differs from
  [`5.2/results/feature_importance.csv`](../5_SHAP_Interpretation/5.2_Attribution_and_Stability/results/feature_importance.csv);
  it matches.
- **Logistic regression** values are exact linear SHAP. Every preprocessing step (median
  imputation, `log1p` on the size counts, standardisation) acts on one column, so with the
  training fold as background a feature's SHAP value is its coefficient times its
  standardised value, and the intercept is the base value
  (`interpret.logistic_rows`). The contributions reproduce the cached out-of-fold logistic
  scores exactly.
- **†** marks a top-10 feature whose SHAP values are offset by another top-10 feature of the
  same attribution group (correlation of their SHAP values ≤ -0.9). Exact identities
  (a dNDVI column and its two NDVI terms; `Area`, `Total_Count` and `Urban_Count`) stay
  collinear in logistic regression too, and its L2 penalty can give two members large
  coefficients of opposite sign. Their individual rows then overstate what the model uses,
  and only their sum is meaningful. The comparison below is therefore made on attribution
  groups (mean |sum of SHAP| within a group), which is exact for both models.

## LightGBM vs logistic regression

**UHI (panels A and C).** The two models agree on what matters. Both rank the summer-vegetation group first and tract size second, and the third place goes to water share for LightGBM and to annual vegetation for logistic regression. They differ in how the vegetation signal is carried. LightGBM sends it almost entirely through `DelNDVI_summer` (mean |SHAP| 1.390, more than twice the next feature). Logistic regression spreads it over the urban NDVI terms and the dNDVI columns. Logistic regression's two largest single features, `Total_Count` (3.929) and `Area` (3.169), largely cancel each other: the whole tract-size group adds 1.569, less than either one alone. The same happens to the elevation pair: `DEM_urb_CT_act` is logistic regression's third feature, but the elevation group ranks 6 of 6 (LightGBM: 5). Features in both top 10s: `Area`, `DEM_urb_CT_act`, `DelNDVI_summer`, `Total_Count`, `Urban_Count`.

| Attribution group | LightGBM rank | LightGBM mean \|Σ SHAP\| | Logistic rank | Logistic mean \|Σ SHAP\| |
|---|---:|---:|---:|---:|
| Vegetation: summer | 1 | 1.542 | 1 | 2.062 |
| Tract size | 2 | 1.041 | 2 | 1.569 |
| Water share | 3 | 0.448 | 5 | 0.517 |
| Vegetation: annual | 4 | 0.421 | 3 | 1.391 |
| Elevation | 5 | 0.292 | 6 | 0.173 |
| Vegetation: winter | 6 | 0.257 | 4 | 0.716 |

**Poverty (panels B and D).** Here the models disagree. LightGBM's largest group is tract size, followed by the three vegetation groups (winter, summer, annual), and its attribution is flat: its top three features lie between 0.188 and 0.226. Logistic regression leans on night-time surface heat. Its two largest groups are annual-night heat (2.620) and winter-night heat (1.792), which LightGBM ranks 11 and 10 of 12. Vegetation follows in logistic regression (ranks 3, 4, 9), and tract size ranks 10. Read this as a difference in what each model can use: a linear model fits one slope per column, so it picks up the broad poverty gradient carried by the many night-LST columns, while the tree ensemble gets its discrimination from thresholds and interactions on tract size and vegetation. Features in both top 10s: `Area`, `DelNDVI_annual_city`.

| Attribution group | LightGBM rank | LightGBM mean \|Σ SHAP\| | Logistic rank | Logistic mean \|Σ SHAP\| |
|---|---:|---:|---:|---:|
| Tract size | 1 | 0.432 | 10 | 0.243 |
| Vegetation: winter | 2 | 0.294 | 9 | 0.450 |
| Vegetation: summer | 3 | 0.246 | 4 | 0.886 |
| Vegetation: annual | 4 | 0.233 | 3 | 0.939 |
| Elevation | 5 | 0.223 | 11 | 0.207 |
| Heat: summer night | 6 | 0.178 | 5 | 0.800 |
| Heat: winter day | 7 | 0.158 | 7 | 0.567 |
| Heat: annual day | 8 | 0.110 | 6 | 0.693 |
| Heat: summer day | 9 | 0.107 | 8 | 0.488 |
| Heat: winter night | 10 | 0.093 | 2 | 1.792 |
| Heat: annual night | 11 | 0.092 | 1 | 2.620 |
| Water share | 12 | 0.064 | 12 | 0.056 |

**Reading the two rows.** Logistic-regression rows always show one clean colour gradient,
because each SHAP value is a linear function of one (transformed) feature. LightGBM rows can
mix colours, which signals non-linear effects and interactions. Compare ranks across the two
rows, not magnitudes: logistic regression's per-feature values are inflated by offsetting
coefficients on collinear columns. SHAP values describe each fitted model's predictions, not
causal effects. The grouped attribution of LightGBM and its stability across 15 fold models
are in [section 5.2](../5_SHAP_Interpretation/5.2_Attribution_and_Stability/README.md).

## Top 10 features

**Panel A — LightGBM, UHI**

| Rank | Feature | Attribution group | Mean \|SHAP\| (log-odds) |
|---:|---|---|---:|
| 1 | `DelNDVI_summer` | Vegetation: summer | 1.390 |
| 2 | `Area` | Tract size | 0.538 |
| 3 | `Total_Count` | Tract size | 0.452 |
| 4 | `water_frac` | Water share | 0.448 |
| 5 | `Urban_Count` | Tract size | 0.260 |
| 6 | `DelDEM` | Elevation | 0.180 |
| 7 | `DelNDVI_annual` | Vegetation: annual | 0.159 |
| 8 | `NDVI_urb_CT` | Vegetation: annual | 0.154 |
| 9 | `DEM_urb_CT_act` | Elevation | 0.131 |
| 10 | `NDVI_urb_CT_summer` | Vegetation: summer | 0.131 |

**Panel B — LightGBM, poverty**

| Rank | Feature | Attribution group | Mean \|SHAP\| (log-odds) |
|---:|---|---|---:|
| 1 | `DelNDVI_winter` | Vegetation: winter | 0.226 |
| 2 | `DelNDVI_annual_city` | Vegetation: annual | 0.191 |
| 3 | `Total_Count` | Tract size | 0.188 |
| 4 | `Area` | Tract size | 0.186 |
| 5 | `DelDEM` | Elevation | 0.172 |
| 6 | `UHI_winter_day_city` | Heat: winter day | 0.135 |
| 7 | `DelNDVI_annual` | Vegetation: annual | 0.131 |
| 8 | `NDVI_urb_CT_summer` | Vegetation: summer | 0.124 |
| 9 | `Urban_Count` | Tract size | 0.092 |
| 10 | `NDVI_rur_summer` | Vegetation: summer | 0.092 |

**Panel C — logistic regression, UHI**

| Rank | Feature | Attribution group | Mean \|SHAP\| (log-odds) |
|---:|---|---|---:|
| 1 | `Total_Count` † | Tract size | 3.929 |
| 2 | `Area` † | Tract size | 3.169 |
| 3 | `DEM_urb_CT_act` † | Elevation | 1.918 |
| 4 | `NDVI_urb_CT_act_summer` | Vegetation: summer | 1.353 |
| 5 | `NDVI_urb_CT_act` | Vegetation: annual | 1.182 |
| 6 | `NDVI_urb_all_summer` | Vegetation: summer | 1.079 |
| 7 | `Urban_Count` | Tract size | 1.077 |
| 8 | `DelNDVI_summer` | Vegetation: summer | 0.974 |
| 9 | `log_aland` † | Tract size | 0.943 |
| 10 | `DEM_urb_CT` † | Elevation | 0.934 |

**Panel D — logistic regression, poverty**

| Rank | Feature | Attribution group | Mean \|SHAP\| (log-odds) |
|---:|---|---|---:|
| 1 | `DEM_urb_all` | Elevation | 1.309 |
| 2 | `LST_urb_night_all_winter` | Heat: winter night | 0.837 |
| 3 | `Area` | Tract size | 0.823 |
| 4 | `LST_rur_night_winter` | Heat: winter night | 0.815 |
| 5 | `LST_urb_night_CT_winter` | Heat: winter night | 0.811 |
| 6 | `UHI_annual_night_city` | Heat: annual night | 0.791 |
| 7 | `DelNDVI_annual_city` | Vegetation: annual | 0.750 |
| 8 | `NDVI_urb_CT_act_summer` | Vegetation: summer | 0.747 |
| 9 | `NDVI_urb_CT_act` | Vegetation: annual | 0.720 |
| 10 | `LST_urb_day_CT_act_summer` | Heat: summer day | 0.670 |

All values: [`results/shap_top_features.csv`](results/shap_top_features.csv) and
[`results/shap_group_importance.csv`](results/shap_group_importance.csv).
