# Models and baselines (tau = 0.5; 95 % UA cluster-bootstrap CIs)

| task | scheme | baseline | config | model | prevalence | accuracy | balanced_accuracy | f1 | mcc | roc_auc | pr_auc | roc_auc_ci_low | roc_auc_ci_high | mcc_ci_low | mcc_ci_high |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | loco | Majority class | A-noLST | majority | 0.500 | 0.336 | 0.336 | 0.286 | -0.331 | 0.500 | 0.500 | 0.500 | 0.500 | -0.364 | -0.288 |
| A | loco | LST-only, logistic | A-lst-only | logistic | 0.500 | 0.526 | 0.526 | 0.477 | 0.054 | 0.582 | 0.535 | 0.535 | 0.647 | -0.045 | 0.165 |
| A | loco | LST-only, LightGBM | A-lst-only | lgbm | 0.500 | 0.610 | 0.610 | 0.630 | 0.221 | 0.619 | 0.552 | 0.536 | 0.696 | 0.064 | 0.357 |
| A | loco | Logistic regression (A-noLST) | A-noLST | logistic | 0.500 | 0.830 | 0.830 | 0.834 | 0.661 | 0.899 | 0.867 | 0.884 | 0.914 | 0.630 | 0.690 |
| A | loco | LightGBM (A-noLST) | A-noLST | lgbm | 0.500 | 0.823 | 0.823 | 0.819 | 0.647 | 0.910 | 0.890 | 0.888 | 0.929 | 0.566 | 0.709 |
| A | ua5 | Majority class | A-noLST | majority | 0.500 | 0.500 | 0.500 | 0.444 | -0.001 | 0.500 | 0.500 | 0.500 | 0.500 | -0.114 | 0.100 |
| A | ua5 | LST-only, logistic | A-lst-only | logistic | 0.500 | 0.530 | 0.530 | 0.482 | 0.061 | 0.590 | 0.544 | 0.548 | 0.650 | -0.036 | 0.173 |
| A | ua5 | LST-only, LightGBM | A-lst-only | lgbm | 0.500 | 0.613 | 0.613 | 0.630 | 0.227 | 0.635 | 0.574 | 0.555 | 0.710 | 0.068 | 0.367 |
| A | ua5 | Logistic regression (A-noLST) | A-noLST | logistic | 0.500 | 0.829 | 0.829 | 0.832 | 0.659 | 0.898 | 0.868 | 0.883 | 0.913 | 0.629 | 0.688 |
| A | ua5 | LightGBM (A-noLST) | A-noLST | lgbm | 0.500 | 0.823 | 0.823 | 0.819 | 0.646 | 0.908 | 0.889 | 0.886 | 0.927 | 0.566 | 0.707 |
| B | loco | Majority class | B-env | majority | 0.500 | 0.391 | 0.391 | 0.414 | -0.219 | 0.500 | 0.500 | 0.500 | 0.500 | -0.253 | -0.189 |
| B | loco | Location only, logistic | geo-only | logistic | 0.500 | 0.542 | 0.542 | 0.543 | 0.083 | 0.559 | 0.544 | 0.540 | 0.578 | 0.040 | 0.127 |
| B | loco | Location only, LightGBM | geo-only | lgbm | 0.500 | 0.563 | 0.563 | 0.537 | 0.126 | 0.579 | 0.575 | 0.554 | 0.605 | 0.083 | 0.169 |
| B | loco | No heat variables, logistic | B-nonthermal | logistic | 0.500 | 0.660 | 0.660 | 0.654 | 0.320 | 0.697 | 0.679 | 0.651 | 0.736 | 0.272 | 0.364 |
| B | loco | No heat variables, LightGBM | B-nonthermal | lgbm | 0.500 | 0.702 | 0.702 | 0.720 | 0.408 | 0.768 | 0.744 | 0.754 | 0.783 | 0.383 | 0.430 |
| B | loco | Logistic regression (B-env) | B-env | logistic | 0.500 | 0.665 | 0.665 | 0.666 | 0.330 | 0.714 | 0.694 | 0.670 | 0.752 | 0.274 | 0.382 |
| B | loco | LightGBM (B-env) | B-env | lgbm | 0.500 | 0.705 | 0.705 | 0.725 | 0.414 | 0.777 | 0.761 | 0.758 | 0.796 | 0.385 | 0.445 |
| B | ua5 | Majority class | B-env | majority | 0.500 | 0.500 | 0.500 | 0.545 | -0.000 | 0.500 | 0.500 | 0.500 | 0.500 | -0.040 | 0.043 |
| B | ua5 | Location only, logistic | geo-only | logistic | 0.500 | 0.536 | 0.536 | 0.549 | 0.072 | 0.555 | 0.542 | 0.532 | 0.575 | 0.029 | 0.115 |
| B | ua5 | Location only, LightGBM | geo-only | lgbm | 0.500 | 0.558 | 0.558 | 0.515 | 0.118 | 0.578 | 0.572 | 0.555 | 0.604 | 0.076 | 0.161 |
| B | ua5 | No heat variables, logistic | B-nonthermal | logistic | 0.500 | 0.655 | 0.655 | 0.649 | 0.310 | 0.695 | 0.679 | 0.650 | 0.733 | 0.262 | 0.352 |
| B | ua5 | No heat variables, LightGBM | B-nonthermal | lgbm | 0.500 | 0.695 | 0.695 | 0.718 | 0.395 | 0.763 | 0.743 | 0.747 | 0.779 | 0.368 | 0.420 |
| B | ua5 | Logistic regression (B-env) | B-env | logistic | 0.500 | 0.662 | 0.662 | 0.664 | 0.324 | 0.712 | 0.692 | 0.669 | 0.750 | 0.268 | 0.375 |
| B | ua5 | LightGBM (B-env) | B-env | lgbm | 0.500 | 0.698 | 0.698 | 0.719 | 0.401 | 0.772 | 0.754 | 0.752 | 0.791 | 0.370 | 0.435 |
