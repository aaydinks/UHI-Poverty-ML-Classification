# SHAP beeswarm: top 10 features per task and model by mean |SHAP|

| task | model | rank | feature | group | mean_abs_shap | offset_in_group | matches_5_2 |
|---|---|---:|---|---|---:|---|---|
| A | lgbm | 1 | DelNDVI_summer | Vegetation: summer | 1.390 | no | yes |
| A | lgbm | 2 | Area | Tract size | 0.538 | no | yes |
| A | lgbm | 3 | Total_Count | Tract size | 0.452 | no | yes |
| A | lgbm | 4 | water_frac | Water share | 0.448 | no | yes |
| A | lgbm | 5 | Urban_Count | Tract size | 0.260 | no | yes |
| A | lgbm | 6 | DelDEM | Elevation | 0.180 | no | yes |
| A | lgbm | 7 | DelNDVI_annual | Vegetation: annual | 0.159 | no | yes |
| A | lgbm | 8 | NDVI_urb_CT | Vegetation: annual | 0.154 | no | yes |
| A | lgbm | 9 | DEM_urb_CT_act | Elevation | 0.131 | no | yes |
| A | lgbm | 10 | NDVI_urb_CT_summer | Vegetation: summer | 0.131 | no | yes |
| B | lgbm | 1 | DelNDVI_winter | Vegetation: winter | 0.226 | no | yes |
| B | lgbm | 2 | DelNDVI_annual_city | Vegetation: annual | 0.191 | no | yes |
| B | lgbm | 3 | Total_Count | Tract size | 0.188 | no | yes |
| B | lgbm | 4 | Area | Tract size | 0.186 | no | yes |
| B | lgbm | 5 | DelDEM | Elevation | 0.172 | no | yes |
| B | lgbm | 6 | UHI_winter_day_city | Heat: winter day | 0.135 | no | yes |
| B | lgbm | 7 | DelNDVI_annual | Vegetation: annual | 0.131 | no | yes |
| B | lgbm | 8 | NDVI_urb_CT_summer | Vegetation: summer | 0.124 | no | yes |
| B | lgbm | 9 | Urban_Count | Tract size | 0.092 | no | yes |
| B | lgbm | 10 | NDVI_rur_summer | Vegetation: summer | 0.092 | no | yes |
| A | logistic | 1 | Total_Count | Tract size | 3.929 | yes | – |
| A | logistic | 2 | Area | Tract size | 3.169 | yes | – |
| A | logistic | 3 | DEM_urb_CT_act | Elevation | 1.918 | yes | – |
| A | logistic | 4 | NDVI_urb_CT_act_summer | Vegetation: summer | 1.353 | no | – |
| A | logistic | 5 | NDVI_urb_CT_act | Vegetation: annual | 1.182 | no | – |
| A | logistic | 6 | NDVI_urb_all_summer | Vegetation: summer | 1.079 | no | – |
| A | logistic | 7 | Urban_Count | Tract size | 1.077 | no | – |
| A | logistic | 8 | DelNDVI_summer | Vegetation: summer | 0.974 | no | – |
| A | logistic | 9 | log_aland | Tract size | 0.943 | yes | – |
| A | logistic | 10 | DEM_urb_CT | Elevation | 0.934 | yes | – |
| B | logistic | 1 | DEM_urb_all | Elevation | 1.309 | no | – |
| B | logistic | 2 | LST_urb_night_all_winter | Heat: winter night | 0.837 | no | – |
| B | logistic | 3 | Area | Tract size | 0.823 | no | – |
| B | logistic | 4 | LST_rur_night_winter | Heat: winter night | 0.815 | no | – |
| B | logistic | 5 | LST_urb_night_CT_winter | Heat: winter night | 0.811 | no | – |
| B | logistic | 6 | UHI_annual_night_city | Heat: annual night | 0.791 | no | – |
| B | logistic | 7 | DelNDVI_annual_city | Vegetation: annual | 0.750 | no | – |
| B | logistic | 8 | NDVI_urb_CT_act_summer | Vegetation: summer | 0.747 | no | – |
| B | logistic | 9 | NDVI_urb_CT_act | Vegetation: annual | 0.720 | no | – |
| B | logistic | 10 | LST_urb_day_CT_act_summer | Heat: summer day | 0.670 | no | – |

Out-of-fold SHAP (log-odds), all tracts of the primary frame, UA-grouped 5-fold, seed 37. `offset_in_group`: offset by another top-10 feature of the same attribution group (SHAP correlation ≤ -0.9). `matches_5_2`: same feature and value at the same rank in 5.2/results/feature_importance.csv (LightGBM).
