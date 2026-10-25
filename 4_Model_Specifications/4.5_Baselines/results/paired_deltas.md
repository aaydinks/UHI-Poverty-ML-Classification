# Paired differences, Δ = a − b (95 % UA cluster-bootstrap CI)

| task | scheme | comparison | metric | a | b | delta | ci_low | ci_high | share_reps_above_0 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| A | loco | LightGBM − logistic | roc_auc | 0.910 | 0.899 | 0.011 | 0.000 | 0.020 | 0.978 |
| A | loco | LightGBM − logistic | mcc | 0.647 | 0.661 | -0.013 | -0.069 | 0.026 | 0.348 |
| A | loco | LightGBM − majority class | roc_auc | 0.910 | 0.500 | 0.410 | 0.388 | 0.429 | 1.000 |
| A | loco | LightGBM − majority class | mcc | 0.647 | -0.331 | 0.978 | 0.899 | 1.043 | 1.000 |
| A | loco | LightGBM (A-noLST) − LightGBM (LST only) | roc_auc | 0.910 | 0.619 | 0.290 | 0.219 | 0.365 | 1.000 |
| A | loco | LightGBM (A-noLST) − LightGBM (LST only) | mcc | 0.647 | 0.221 | 0.426 | 0.326 | 0.539 | 1.000 |
| A | ua5 | LightGBM − logistic | roc_auc | 0.908 | 0.898 | 0.009 | -0.001 | 0.018 | 0.959 |
| A | ua5 | LightGBM − logistic | mcc | 0.646 | 0.659 | -0.013 | -0.068 | 0.028 | 0.356 |
| A | ua5 | LightGBM − majority class | roc_auc | 0.908 | 0.500 | 0.408 | 0.386 | 0.427 | 1.000 |
| A | ua5 | LightGBM − majority class | mcc | 0.646 | -0.001 | 0.646 | 0.558 | 0.756 | 1.000 |
| A | ua5 | LightGBM (A-noLST) − LightGBM (LST only) | roc_auc | 0.908 | 0.635 | 0.272 | 0.202 | 0.348 | 1.000 |
| A | ua5 | LightGBM (A-noLST) − LightGBM (LST only) | mcc | 0.646 | 0.227 | 0.418 | 0.303 | 0.539 | 1.000 |
| B | loco | LightGBM − logistic | roc_auc | 0.777 | 0.714 | 0.064 | 0.016 | 0.116 | 0.999 |
| B | loco | LightGBM − logistic | mcc | 0.414 | 0.330 | 0.085 | 0.024 | 0.151 | 1.000 |
| B | loco | LightGBM − majority class | roc_auc | 0.777 | 0.500 | 0.277 | 0.258 | 0.296 | 1.000 |
| B | loco | LightGBM − majority class | mcc | 0.414 | -0.219 | 0.633 | 0.591 | 0.684 | 1.000 |
| B | loco | LightGBM (B-env) − LightGBM (location only) | roc_auc | 0.777 | 0.579 | 0.198 | 0.171 | 0.230 | 1.000 |
| B | loco | LightGBM (B-env) − LightGBM (location only) | mcc | 0.414 | 0.126 | 0.288 | 0.240 | 0.341 | 1.000 |
| B | loco | LightGBM (B-env) − LightGBM (no heat variables) | roc_auc | 0.777 | 0.768 | 0.009 | -0.002 | 0.019 | 0.955 |
| B | loco | LightGBM (B-env) − LightGBM (no heat variables) | mcc | 0.414 | 0.408 | 0.007 | -0.017 | 0.027 | 0.759 |
| B | ua5 | LightGBM − logistic | roc_auc | 0.772 | 0.712 | 0.060 | 0.011 | 0.113 | 0.998 |
| B | ua5 | LightGBM − logistic | mcc | 0.401 | 0.324 | 0.078 | 0.012 | 0.148 | 0.992 |
| B | ua5 | LightGBM − majority class | roc_auc | 0.772 | 0.500 | 0.272 | 0.252 | 0.291 | 1.000 |
| B | ua5 | LightGBM − majority class | mcc | 0.401 | -0.000 | 0.402 | 0.350 | 0.451 | 1.000 |
| B | ua5 | LightGBM (B-env) − LightGBM (location only) | roc_auc | 0.772 | 0.578 | 0.193 | 0.166 | 0.223 | 1.000 |
| B | ua5 | LightGBM (B-env) − LightGBM (location only) | mcc | 0.401 | 0.118 | 0.284 | 0.231 | 0.336 | 1.000 |
| B | ua5 | LightGBM (B-env) − LightGBM (no heat variables) | roc_auc | 0.772 | 0.763 | 0.008 | -0.002 | 0.018 | 0.933 |
| B | ua5 | LightGBM (B-env) − LightGBM (no heat variables) | mcc | 0.401 | 0.395 | 0.006 | -0.014 | 0.025 | 0.763 |
