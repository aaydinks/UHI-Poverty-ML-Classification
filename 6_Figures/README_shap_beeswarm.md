# SHAP beeswarm

The beeswarm shows the SHAP values of the top 10 features per task for the two evaluated models
([`results/shap_beeswarm.png`](results/shap_beeswarm.png)):

- **Panel A, `High_UHI` without temperature input** (`A-noLST`): the model uses no `LST_*` or
  `UHI_*` column, because the urban and rural LST terms define the label (section 3.1).
- **Panel B, `High_Pov` from the ACS poverty rate** (`B-env`): positive SHAP raises the
  predicted probability of *high* poverty.
- SHAP values are LightGBM's exact TreeSHAP in log-odds, computed out of fold under the
  UA-grouped 5-fold split (section 5.1, seed 37) on the national sample;
  each panel shows 3,000 randomly chosen tracts. Colour is the feature's percentile among
  all tracts.
- `shap_beeswarm.py` reads the cached attributions and stops if its ranking differs from
  [`5.2/results/feature_importance.csv`](../5_SHAP_Interpretation/5.2_Attribution_and_Stability/results/feature_importance.csv);
  it matches.

**Panel A**

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

**Panel B**

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

`DelNDVI_summer` leads panel A with a mean |SHAP| of 1.390, more than
twice the next feature. Panel B is flatter: its top three features lie between
0.188 and 0.226.

**Reading the figure.** Several features belong to exact identities (a dNDVI column and its two
NDVI terms, or `Area`, `Total_Count` and `Urban_Count`), and TreeSHAP splits the credit between
the members of an identity arbitrarily. Rank per feature is therefore not interpretable on its
own; the grouped attribution and its stability across 15 fold models are in
[section 5.2](../5_SHAP_Interpretation/5.2_Attribution_and_Stability/README.md). SHAP values
describe the fitted model's predictions, not causal effects.
