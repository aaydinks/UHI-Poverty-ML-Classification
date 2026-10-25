# Grouped attribution and permutation importance, out of fold

| task | group | n_features | mean_abs_shap | share | rank | rank_min | rank_max | rank_stable | perm_auc_drop | perm_auc_drop_min | perm_auc_drop_max |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| A | Vegetation: summer | 6 | 1.542 | 0.385 | 1 | 1 | 1 | yes | 0.157 | 0.122 | 0.210 |
| A | Tract size | 4 | 1.041 | 0.260 | 2 | 2 | 2 | yes | 0.065 | 0.046 | 0.085 |
| A | Water share | 1 | 0.448 | 0.112 | 3 | 3 | 4 | yes | 0.035 | 0.024 | 0.045 |
| A | Vegetation: annual | 6 | 0.421 | 0.105 | 4 | 3 | 5 | yes | 0.012 | 0.004 | 0.018 |
| A | Elevation | 5 | 0.292 | 0.073 | 5 | 3 | 6 | no | 0.007 | -0.008 | 0.014 |
| A | Vegetation: winter | 6 | 0.257 | 0.064 | 6 | 5 | 6 | yes | 0.006 | 0.001 | 0.011 |
| B | Tract size | 4 | 0.432 | 0.194 | 1 | 1 | 1 | yes | 0.049 | 0.018 | 0.087 |
| B | Vegetation: winter | 6 | 0.294 | 0.132 | 2 | 2 | 4 | yes | 0.032 | 0.017 | 0.040 |
| B | Vegetation: summer | 6 | 0.246 | 0.110 | 3 | 2 | 4 | yes | 0.016 | -0.002 | 0.028 |
| B | Vegetation: annual | 6 | 0.233 | 0.105 | 4 | 2 | 6 | no | 0.014 | 0.008 | 0.024 |
| B | Elevation | 5 | 0.223 | 0.100 | 5 | 2 | 6 | no | 0.023 | 0.010 | 0.047 |
| B | Heat: summer night | 6 | 0.178 | 0.080 | 6 | 5 | 8 | no | 0.015 | 0.003 | 0.027 |
| B | Heat: winter day | 6 | 0.158 | 0.071 | 7 | 5 | 8 | no | 0.007 | 0.001 | 0.018 |
| B | Heat: annual day | 6 | 0.110 | 0.049 | 8 | 7 | 11 | no | 0.006 | 0.002 | 0.012 |
| B | Heat: summer day | 6 | 0.107 | 0.048 | 9 | 7 | 10 | no | 0.007 | 0.002 | 0.014 |
| B | Heat: winter night | 6 | 0.093 | 0.042 | 10 | 7 | 11 | no | 0.001 | -0.015 | 0.009 |
| B | Heat: annual night | 6 | 0.092 | 0.041 | 11 | 8 | 12 | no | 0.005 | -0.000 | 0.009 |
| B | Water share | 1 | 0.064 | 0.029 | 12 | 11 | 12 | yes | 0.003 | 0.001 | 0.005 |

`mean_abs_shap`: mean over tracts of |sum of SHAP within the group| (log-odds), seed-37 held-out folds. `rank_min`/`rank_max`: across the 15 fold models. `perm_auc_drop`: mean drop in held-out ROC-AUC when the group's columns are permuted together (15 fold models × 5 draws).
