# Contributions against the cached out-of-fold scores

| task | config | n_tracts | n_features | fold_models_of_record | max_abs_error_probability | max_abs_error_logit |
|---|---|---:|---:|---:|---|---|
| A | A-noLST | 54,243 | 28 | 5 | 7.1e-15 | 3.4e-14 |
| B | B-env | 54,243 | 64 | 5 | 3.3e-15 | 2.0e-14 |

Error = |sigmoid(bias + sum of SHAP) − OOF score|, and the same on the logit scale, over every tract.
