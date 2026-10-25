# 4.5 Baselines

## Purpose

This section tests whether the gradient-boosted models beat simpler alternatives: a
majority-class baseline, a plain statistical model, and, for the heat task, a model that only
sees land-surface temperature.

## Method

`baselines.py` compares, under leave-one-city-out and UA-grouped 5-fold:

- the majority class (`DummyClassifier("most_frequent")`);
- logistic regression (median imputation → `log1p` on area and pixel counts → standardisation);
- LightGBM (fixed settings);
- reference baselines: for Task A, `LST_urb_day_CT` alone (an LST-only model); for
  Task B, location only (centroid latitude/longitude + Census region) and the same
  non-thermal features as `A-noLST` with no heat variable (`B-nonthermal`).

Paired differences use the same UA bootstrap draws for both models (B = 1,000).

A constant score carries no ranking, so the majority class's ROC-AUC is 0.5 and its PR-AUC
equals the prevalence by definition. Its threshold metrics are computed from its predictions as
they are. With a median split the two classes differ by one tract, so holding out a city tips
the training majority toward the class that city has less of. Pooled over leave-one-city-out
folds, the majority class is therefore worse than chance (section 4.3), which makes
"LightGBM − majority" differences in MCC larger than LightGBM's gain over a no-skill rule
(MCC 0). That gain is simply LightGBM's own MCC.

## Results

Leave-one-city-out (UA-grouped 5-fold is in [`results/baselines.md`](results/baselines.md)):

| Task | Model | ROC-AUC [95 % CI] | MCC | Balanced accuracy | PR-AUC |
|---|---|---:|---:|---:|---:|
| A | Majority class | 0.500 [0.500, 0.500] | -0.331 | 0.336 | 0.500 |
| A | LST-only, logistic | 0.582 [0.535, 0.647] | 0.054 | 0.526 | 0.535 |
| A | LST-only, LightGBM | 0.619 [0.536, 0.696] | 0.221 | 0.610 | 0.552 |
| A | Logistic regression (A-noLST) | 0.899 [0.884, 0.914] | 0.661 | 0.830 | 0.867 |
| A | LightGBM (A-noLST) | 0.910 [0.888, 0.929] | 0.647 | 0.823 | 0.890 |
| B | Majority class | 0.500 [0.500, 0.500] | -0.219 | 0.391 | 0.500 |
| B | Location only, logistic | 0.559 [0.540, 0.578] | 0.083 | 0.542 | 0.544 |
| B | Location only, LightGBM | 0.579 [0.554, 0.605] | 0.126 | 0.563 | 0.575 |
| B | No heat variables, logistic | 0.697 [0.651, 0.736] | 0.320 | 0.660 | 0.679 |
| B | No heat variables, LightGBM | 0.768 [0.754, 0.783] | 0.408 | 0.702 | 0.744 |
| B | Logistic regression (B-env) | 0.714 [0.670, 0.752] | 0.330 | 0.665 | 0.694 |
| B | LightGBM (B-env) | 0.777 [0.758, 0.796] | 0.414 | 0.705 | 0.761 |

Paired differences, leave-one-city-out ([`results/paired_deltas.md`](results/paired_deltas.md)):

| Task | Comparison | Δ ROC-AUC [95 % CI] | Δ MCC [95 % CI] |
|---|---|---:|---:|
| A | LightGBM − logistic | 0.011 [0.000, 0.020] | -0.013 [-0.069, 0.026] |
| A | LightGBM − majority class | 0.410 [0.388, 0.429] | 0.978 [0.899, 1.043] |
| A | LightGBM (A-noLST) − LightGBM (LST only) | 0.290 [0.219, 0.365] | 0.426 [0.326, 0.539] |
| B | LightGBM − logistic | 0.064 [0.016, 0.116] | 0.085 [0.024, 0.151] |
| B | LightGBM − majority class | 0.277 [0.258, 0.296] | 0.633 [0.591, 0.684] |
| B | LightGBM (B-env) − LightGBM (location only) | 0.198 [0.171, 0.230] | 0.288 [0.240, 0.341] |
| B | LightGBM (B-env) − LightGBM (no heat variables) | 0.009 [-0.002, 0.019] | 0.007 [-0.017, 0.027] |

`LST_urb_day_CT` is one of the two terms of the target (`UHI_annual_day` = `LST_urb_day_CT` −
`LST_rur_day`, section 3.1), yet on its own it predicts high SUHI poorly in held-out cities
(ROC-AUC 0.619). SUHI is measured against each city's own rural reference, so the same
tract temperature can be a strong heat island in one city and none in another. The leak-free
`A-noLST` model, which sees no temperature at all, does far better (Δ ROC-AUC
0.290 [0.219, 0.365]).

**Heat and poverty.** Adding every SUHI and LST variable to the non-thermal features
changes LightGBM's poverty ROC-AUC by 0.009 [-0.002, 0.019], an improvement the data cannot distinguish from zero
under leave-one-city-out: the model without any heat variable reaches ROC-AUC
0.768, against 0.777
with them. Nearly all of the poverty model's skill comes from vegetation, elevation and built
form, the same inputs that predict high SUHI in Task A (ROC-AUC
0.910).

Figure: [`results/baselines.png`](results/baselines.png).
