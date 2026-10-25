# SHAP beeswarm: top 10 features per task by mean |SHAP|

| task | rank | feature | group | mean_abs_shap | matches_5_2 |
|---|---:|---|---|---:|---|
| A | 1 | DelNDVI_summer | Vegetation: summer | 1.390 | yes |
| A | 2 | Area | Tract size | 0.538 | yes |
| A | 3 | Total_Count | Tract size | 0.452 | yes |
| A | 4 | water_frac | Water share | 0.448 | yes |
| A | 5 | Urban_Count | Tract size | 0.260 | yes |
| A | 6 | DelDEM | Elevation | 0.180 | yes |
| A | 7 | DelNDVI_annual | Vegetation: annual | 0.159 | yes |
| A | 8 | NDVI_urb_CT | Vegetation: annual | 0.154 | yes |
| A | 9 | DEM_urb_CT_act | Elevation | 0.131 | yes |
| A | 10 | NDVI_urb_CT_summer | Vegetation: summer | 0.131 | yes |
| B | 1 | DelNDVI_winter | Vegetation: winter | 0.226 | yes |
| B | 2 | DelNDVI_annual_city | Vegetation: annual | 0.191 | yes |
| B | 3 | Total_Count | Tract size | 0.188 | yes |
| B | 4 | Area | Tract size | 0.186 | yes |
| B | 5 | DelDEM | Elevation | 0.172 | yes |
| B | 6 | UHI_winter_day_city | Heat: winter day | 0.135 | yes |
| B | 7 | DelNDVI_annual | Vegetation: annual | 0.131 | yes |
| B | 8 | NDVI_urb_CT_summer | Vegetation: summer | 0.124 | yes |
| B | 9 | Urban_Count | Tract size | 0.092 | yes |
| B | 10 | NDVI_rur_summer | Vegetation: summer | 0.092 | yes |

Out-of-fold TreeSHAP (log-odds), all tracts of the primary frame, seed 37. `matches_5_2`: same feature and value at the same rank in 5.2/results/feature_importance.csv.
