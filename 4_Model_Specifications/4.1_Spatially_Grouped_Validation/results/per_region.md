# Leave-one-region-out, metrics within each held-out region

| task | model | region | n_tracts | prevalence | accuracy | f1 | mcc | roc_auc |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | logistic | Midwest | 11,349 | 0.549 | 0.797 | 0.798 | 0.607 | 0.898 |
| A | logistic | Northeast | 11,094 | 0.587 | 0.869 | 0.895 | 0.732 | 0.914 |
| A | logistic | South | 18,652 | 0.441 | 0.817 | 0.779 | 0.628 | 0.906 |
| A | logistic | West | 13,148 | 0.468 | 0.716 | 0.716 | 0.437 | 0.785 |
| A | lgbm | Midwest | 11,349 | 0.549 | 0.850 | 0.868 | 0.697 | 0.917 |
| A | lgbm | Northeast | 11,094 | 0.587 | 0.829 | 0.847 | 0.658 | 0.908 |
| A | lgbm | South | 18,652 | 0.441 | 0.830 | 0.807 | 0.654 | 0.910 |
| A | lgbm | West | 13,148 | 0.468 | 0.751 | 0.719 | 0.499 | 0.811 |
| B | logistic | Midwest | 11,349 | 0.506 | 0.678 | 0.718 | 0.366 | 0.744 |
| B | logistic | Northeast | 11,094 | 0.431 | 0.733 | 0.718 | 0.476 | 0.805 |
| B | logistic | South | 18,652 | 0.552 | 0.523 | 0.304 | 0.183 | 0.644 |
| B | logistic | West | 13,148 | 0.480 | 0.593 | 0.405 | 0.202 | 0.635 |
| B | lgbm | Midwest | 11,349 | 0.506 | 0.728 | 0.728 | 0.455 | 0.785 |
| B | lgbm | Northeast | 11,094 | 0.431 | 0.708 | 0.713 | 0.451 | 0.782 |
| B | lgbm | South | 18,652 | 0.552 | 0.568 | 0.517 | 0.179 | 0.631 |
| B | lgbm | West | 13,148 | 0.480 | 0.583 | 0.679 | 0.251 | 0.679 |
