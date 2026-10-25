# 5 SHAP interpretation

Which model the SHAP values explain, how they map back to named variables, and which input
groups the two evaluated models rely on, keeping attribution apart from causal or mechanistic
claims.

| Section | Folder | Script | Main outputs |
|---|---|---|---|
| 5.1 The explained model | [`5.1_Explained_Model/`](5.1_Explained_Model/) | `explained_model.py` | the model explained (the evaluated LightGBM, fitted on the input columns themselves); the check that its SHAP values reproduce the cached out-of-fold predictions; attribution groups that keep each exact identity whole |
| 5.2 Attribution and stability | [`5.2_Attribution_and_Stability/`](5.2_Attribution_and_Stability/) | `attribution.py` | grouped and per-feature SHAP importance out of fold, grouped permutation importance, and rank stability across 15 fold models |

Run with `make interpret shap`. `make interpret` fills the attribution cache
(`data/interim/interpret/`, under a minute on 15 cores) from the shared module
`src/uhi_poverty/interpret.py`; it needs the out-of-fold cache from `make engine`, which it
checks the attributions against.

The attribution groups and the identities they respect are defined in
`src/uhi_poverty/features.py`. `tests/test_interpret.py` fails if an identity verified in
section 3.1 is split across two groups.
