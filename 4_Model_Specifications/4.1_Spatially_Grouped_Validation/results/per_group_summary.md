# Distribution of within-group ROC-AUC

| task | model | unit | n_units | n_eligible | auc_median | auc_p25 | auc_p75 | auc_min | auc_max | share_auc_above_0.5 | share_auc_at_least_0.7 | mean_across_units_auc | mean_across_units_mcc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | lgbm | UA (leave-one-city-out) | 486 | 272 | 0.943 | 0.892 | 0.973 | 0.513 | 1.000 | 1.000 | 0.982 | 0.922 | 0.639 |
| A | logistic | UA (leave-one-city-out) | 486 | 272 | 0.936 | 0.861 | 0.967 | 0.106 | 1.000 | 0.989 | 0.941 | 0.894 | 0.623 |
| B | lgbm | UA (leave-one-city-out) | 486 | 277 | 0.794 | 0.707 | 0.846 | 0.182 | 0.981 | 0.975 | 0.783 | 0.771 | 0.360 |
| B | logistic | UA (leave-one-city-out) | 486 | 277 | 0.774 | 0.677 | 0.833 | 0.346 | 0.962 | 0.964 | 0.708 | 0.748 | 0.329 |
| A | lgbm | state (leave-one-state-out) | 51 | 49 | 0.934 | 0.892 | 0.957 | 0.713 | 0.976 | 1.000 | 1.000 | 0.918 | 0.632 |
| A | logistic | state (leave-one-state-out) | 51 | 49 | 0.916 | 0.874 | 0.949 | 0.609 | 0.976 | 1.000 | 0.959 | 0.899 | 0.625 |
| B | lgbm | state (leave-one-state-out) | 51 | 50 | 0.779 | 0.727 | 0.826 | 0.482 | 0.887 | 0.980 | 0.800 | 0.765 | 0.373 |
| B | logistic | state (leave-one-state-out) | 51 | 50 | 0.764 | 0.712 | 0.801 | 0.472 | 0.862 | 0.980 | 0.780 | 0.744 | 0.323 |
