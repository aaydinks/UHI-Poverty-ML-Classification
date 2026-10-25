# 3 UHI model predictors

The UHI target is a difference of two land-surface temperature columns, so a classifier of high
UHI that sees those temperatures can reconstruct its own target. This section states the exact
formula and builds the UHI model from inputs that contain no temperature.

| Section | Folder | Script | Main outputs |
|---|---|---|---|
| 3.1 The exact UHI formula | [`3.1_UHI_Formula/`](3.1_UHI_Formula/) | `verify_formula.py` | the formula and definitions; every difference identity in the file checked to floating-point precision |
| 3.2 Gradient-boosted model without LST | [`3.2_GBM_Without_LST/`](3.2_GBM_Without_LST/) | `gbm_no_lst.py` | the leak-free `A-noLST` model, the ablation ladder, and the full predictor list of every configuration |

Run with `make predictors` (3.2 reads the cached out-of-fold predictions; run `make engine` first).

The feature configurations are defined once in `src/uhi_poverty/features.py`, and
`tests/test_feature_policy.py` fails if any `LST_*` or `UHI_*` column, the target, or an
identifier enters `A-noLST`.
