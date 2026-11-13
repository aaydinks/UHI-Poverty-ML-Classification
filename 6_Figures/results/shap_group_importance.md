# SHAP by attribution group, per task and model

| task | model | group | n_features | mean_abs_group_shap | rank |
|---|---|---|---:|---:|---:|
| A | lgbm | Vegetation: summer | 6 | 1.542 | 1 |
| A | lgbm | Tract size | 4 | 1.041 | 2 |
| A | lgbm | Water share | 1 | 0.448 | 3 |
| A | lgbm | Vegetation: annual | 6 | 0.421 | 4 |
| A | lgbm | Elevation | 5 | 0.292 | 5 |
| A | lgbm | Vegetation: winter | 6 | 0.257 | 6 |
| B | lgbm | Tract size | 4 | 0.432 | 1 |
| B | lgbm | Vegetation: winter | 6 | 0.294 | 2 |
| B | lgbm | Vegetation: summer | 6 | 0.246 | 3 |
| B | lgbm | Vegetation: annual | 6 | 0.233 | 4 |
| B | lgbm | Elevation | 5 | 0.223 | 5 |
| B | lgbm | Heat: summer night | 6 | 0.178 | 6 |
| B | lgbm | Heat: winter day | 6 | 0.158 | 7 |
| B | lgbm | Heat: annual day | 6 | 0.110 | 8 |
| B | lgbm | Heat: summer day | 6 | 0.107 | 9 |
| B | lgbm | Heat: winter night | 6 | 0.093 | 10 |
| B | lgbm | Heat: annual night | 6 | 0.092 | 11 |
| B | lgbm | Water share | 1 | 0.064 | 12 |
| A | logistic | Vegetation: summer | 6 | 2.062 | 1 |
| A | logistic | Tract size | 4 | 1.569 | 2 |
| A | logistic | Vegetation: annual | 6 | 1.391 | 3 |
| A | logistic | Vegetation: winter | 6 | 0.716 | 4 |
| A | logistic | Water share | 1 | 0.517 | 5 |
| A | logistic | Elevation | 5 | 0.173 | 6 |
| B | logistic | Heat: annual night | 6 | 2.620 | 1 |
| B | logistic | Heat: winter night | 6 | 1.792 | 2 |
| B | logistic | Vegetation: annual | 6 | 0.939 | 3 |
| B | logistic | Vegetation: summer | 6 | 0.886 | 4 |
| B | logistic | Heat: summer night | 6 | 0.800 | 5 |
| B | logistic | Heat: annual day | 6 | 0.693 | 6 |
| B | logistic | Heat: winter day | 6 | 0.567 | 7 |
| B | logistic | Heat: summer day | 6 | 0.488 | 8 |
| B | logistic | Vegetation: winter | 6 | 0.450 | 9 |
| B | logistic | Tract size | 4 | 0.243 | 10 |
| B | logistic | Elevation | 5 | 0.207 | 11 |
| B | logistic | Water share | 1 | 0.056 | 12 |

Mean over tracts of |sum of SHAP within the group| (log-odds); rank 1 = largest.
