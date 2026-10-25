# 4.4 Full metric panel with uncertainty

## Purpose

This section gathers everything needed to judge the classifiers in one table: class
prevalence, the full set of classification metrics rather than F1 alone, and confidence
intervals.

## Method

`metric_panel.py` scores every model (majority class, logistic regression, LightGBM) × task
(A: `High_UHI` on `A-noLST`; B: `High_Pov` on `B-env`) × validation scheme (7 schemes) from
the shared out-of-fold predictions. For each it reports n and prevalence; accuracy, balanced
accuracy, precision, recall, specificity and F1 at τ = 0.5; ROC-AUC and PR-AUC; MCC and the
Brier score. Each metric has a percentile 95 % CI from a cluster bootstrap that resamples
urbanized areas (B = 1,000, seed 37), so the intervals respect the nesting of tracts in cities.
The 5-fold schemes also report the across-fold SD.

## Results

Headline scheme, leave-one-city-out (estimate [95 % CI]):

| Task | Model | ROC-AUC | PR-AUC | MCC | F1 | Balanced accuracy |
|---|---|---:|---:|---:|---:|---:|
| A | Majority class | 0.500 [0.500, 0.500] | 0.500 [0.446, 0.545] | -0.331 [-0.364, -0.288] | 0.286 [0.232, 0.346] | 0.336 [0.320, 0.357] |
| A | Logistic regression | 0.899 [0.884, 0.914] | 0.867 [0.834, 0.890] | 0.661 [0.630, 0.690] | 0.834 [0.809, 0.853] | 0.830 [0.815, 0.844] |
| A | LightGBM | 0.910 [0.888, 0.929] | 0.890 [0.868, 0.910] | 0.647 [0.566, 0.709] | 0.819 [0.768, 0.856] | 0.823 [0.782, 0.854] |
| B | Majority class | 0.500 [0.500, 0.500] | 0.500 [0.479, 0.522] | -0.219 [-0.253, -0.189] | 0.414 [0.343, 0.475] | 0.391 [0.374, 0.407] |
| B | Logistic regression | 0.714 [0.670, 0.752] | 0.694 [0.667, 0.726] | 0.330 [0.274, 0.382] | 0.666 [0.620, 0.696] | 0.665 [0.637, 0.691] |
| B | LightGBM | 0.777 [0.758, 0.796] | 0.761 [0.734, 0.791] | 0.414 [0.385, 0.445] | 0.725 [0.707, 0.742] | 0.705 [0.691, 0.720] |

The majority class scores below chance on MCC and balanced accuracy under leave-one-city-out
because its constant prediction flips toward the class each held-out city has less of
([`../4.3_Confusion_Matrices_and_Denominators/README.md`](../4.3_Confusion_Matrices_and_Denominators/README.md)).
Prevalence is 0.500 for Task A and
0.500 for Task B, so PR-AUC has a no-skill floor near
0.5. Every model × scheme row is in [`results/metric_panel.md`](results/metric_panel.md) (long
format: [`results/metric_panel.csv`](results/metric_panel.csv)); the forest plot is
[`results/forest_plot.png`](results/forest_plot.png).
