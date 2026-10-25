# Large-metro sensitivity

## Concentration

| top_k_uas | tracts | share_of_tracts | uas |
|---:|---:|---:|---|
| 1 | 4,217 | 0.078 | New York--Newark, NY--NJ--CT |
| 3 | 9,034 | 0.167 | New York--Newark, NY--NJ--CT; Los Angeles--Long Beach--Anaheim, CA; Chicago, IL--IN |
| 5 | 11,617 | 0.214 | New York--Newark, NY--NJ--CT; Los Angeles--Long Beach--Anaheim, CA; Chicago, IL--IN; Philadelphia, PA--NJ--DE--MD; Miami, FL |
| 10 | 16,898 | 0.312 | New York--Newark, NY--NJ--CT; Los Angeles--Long Beach--Anaheim, CA; Chicago, IL--IN; Philadelphia, PA--NJ--DE--MD; Miami, FL; Detroit, MI; Washington, DC--VA--MD; Dallas--Fort Worth--Arlington, TX; Boston, MA--NH--RI; Houston, TX |
| 20 | 23,777 | 0.438 |  |


Gini coefficient of tracts across 486 UAs: 0.685.

## Models without the largest UAs (UA-grouped 5-fold)

| analysis | task | model | top_uas_dropped | n | roc_auc | roc_auc_ci_low | roc_auc_ci_high | mcc | f1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| model, UA-grouped 5-fold | A | logistic | 0 | 54,243 | 0.898 | 0.883 | 0.913 | 0.659 | 0.832 |
| model, UA-grouped 5-fold | A | logistic | 1 | 50,026 | 0.899 | 0.881 | 0.915 | 0.654 | 0.822 |
| model, UA-grouped 5-fold | A | logistic | 3 | 45,209 | 0.902 | 0.885 | 0.917 | 0.657 | 0.814 |
| model, UA-grouped 5-fold | A | logistic | 10 | 37,345 | 0.901 | 0.883 | 0.917 | 0.653 | 0.797 |
| model, UA-grouped 5-fold | A | lgbm | 0 | 54,243 | 0.908 | 0.886 | 0.927 | 0.646 | 0.819 |
| model, UA-grouped 5-fold | A | lgbm | 1 | 50,026 | 0.909 | 0.883 | 0.931 | 0.642 | 0.809 |
| model, UA-grouped 5-fold | A | lgbm | 3 | 45,209 | 0.920 | 0.906 | 0.932 | 0.684 | 0.827 |
| model, UA-grouped 5-fold | A | lgbm | 10 | 37,345 | 0.915 | 0.897 | 0.929 | 0.676 | 0.810 |
| model, UA-grouped 5-fold | B | logistic | 0 | 54,243 | 0.712 | 0.669 | 0.750 | 0.324 | 0.664 |
| model, UA-grouped 5-fold | B | logistic | 1 | 50,026 | 0.719 | 0.691 | 0.749 | 0.325 | 0.665 |
| model, UA-grouped 5-fold | B | logistic | 3 | 45,209 | 0.749 | 0.732 | 0.768 | 0.369 | 0.692 |
| model, UA-grouped 5-fold | B | logistic | 10 | 37,345 | 0.749 | 0.732 | 0.764 | 0.363 | 0.694 |
| model, UA-grouped 5-fold | B | lgbm | 0 | 54,243 | 0.772 | 0.752 | 0.791 | 0.401 | 0.719 |
| model, UA-grouped 5-fold | B | lgbm | 1 | 50,026 | 0.777 | 0.759 | 0.793 | 0.412 | 0.724 |
| model, UA-grouped 5-fold | B | lgbm | 3 | 45,209 | 0.772 | 0.755 | 0.790 | 0.394 | 0.717 |
| model, UA-grouped 5-fold | B | lgbm | 10 | 37,345 | 0.778 | 0.764 | 0.791 | 0.403 | 0.728 |


## Pooled vs per-city (leave-one-city-out)

| task | model | pooled_roc_auc | pooled_mcc | pooled_accuracy | n_eligible_uas | mean_city_roc_auc | median_city_roc_auc | city_roc_auc_p25 | city_roc_auc_p75 | mean_city_mcc | mean_city_accuracy | median_city_roc_auc_top10 | median_city_roc_auc_others |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | logistic | 0.899 | 0.661 | 0.830 | 272 | 0.894 | 0.936 | 0.861 | 0.967 | 0.623 | 0.843 | 0.902 | 0.938 |
| A | lgbm | 0.910 | 0.647 | 0.823 | 272 | 0.922 | 0.943 | 0.892 | 0.973 | 0.639 | 0.854 | 0.911 | 0.943 |
| B | logistic | 0.714 | 0.330 | 0.665 | 277 | 0.748 | 0.774 | 0.677 | 0.833 | 0.329 | 0.669 | 0.782 | 0.774 |
| B | lgbm | 0.777 | 0.414 | 0.705 | 277 | 0.771 | 0.794 | 0.707 | 0.846 | 0.360 | 0.714 | 0.765 | 0.795 |


## Association without the largest UAs

| analysis | top_uas_dropped | n | odds_ratio | or_ci_low | or_ci_high | cmh_or | cmh_or_ci_low | cmh_or_ci_high | cramers_v |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| association (2.2) | 0 | 54,243 | 2.531 | 2.445 | 2.620 | 3.792 | 3.638 | 3.952 | 0.228 |
| association (2.2) | 1 | 50,026 | 2.605 | 2.513 | 2.701 | 3.885 | 3.721 | 4.056 | 0.235 |
| association (2.2) | 3 | 45,209 | 2.617 | 2.520 | 2.719 | 4.017 | 3.838 | 4.204 | 0.235 |
| association (2.2) | 10 | 37,345 | 2.612 | 2.503 | 2.726 | 4.032 | 3.833 | 4.241 | 0.231 |
