# 4.2 Holding out complete states and major urban areas

## Purpose

This section holds out whole states and major urban areas, and measures how much random
cross-validation overstates performance when tracts are spatially dependent.

## Method

- **Optimism table.** For each task and learned model, the random 5-fold out-of-fold predictions
  are compared with UA-grouped 5-fold, leave-one-state-out (UA-buffered), each of the 20 largest
  UAs held out, and leave-one-city-out. Δ = random − held-out is computed on the same tracts,
  with a paired UA cluster-bootstrap 95 % CI (B = 1,000, the same UA draws for both).
- **Spatial dependence.** Moran's I with row-standardised weights on each tract's 8 nearest
  neighbours (great-circle distance between Gazetteer centroids), with a 999-permutation
  p-value; and, for every scheme, the share of validation tracts that have at least one of their
  8 nearest neighbours in the training fold.

## Results

**Optimism in ROC-AUC** ([`results/optimism.md`](results/optimism.md) also has MCC and F1):

| Task | Model | Random 5-fold ROC-AUC | Δ vs UA-grouped 5-fold | Δ vs Leave-one-state-out (UA-buffered) | Δ vs Each of the 20 largest UAs held out | Δ vs Leave-one-city-out |
|---|---|---:|---:|---:|---:|---:|
| A | Logistic regression | 0.905 | 0.007 [0.003, 0.013] | 0.010 [0.006, 0.016] | 0.012 [0.004, 0.025] | 0.006 [0.003, 0.011] |
| A | LightGBM | 0.946 | 0.038 [0.022, 0.055] | 0.038 [0.028, 0.050] | 0.067 [0.031, 0.108] | 0.036 [0.020, 0.053] |
| B | Logistic regression | 0.760 | 0.048 [0.016, 0.092] | 0.050 [0.014, 0.098] | 0.082 [0.008, 0.166] | 0.046 [0.014, 0.090] |
| B | LightGBM | 0.839 | 0.068 [0.052, 0.086] | 0.104 [0.080, 0.126] | 0.077 [0.044, 0.114] | 0.062 [0.046, 0.079] |

For Task A, LightGBM's ROC-AUC falls from 0.946 under random folds to
0.910 when the city is held out; for Task B it falls from 0.839 to
0.777.

**Spatial dependence** ([`results/morans_i.md`](results/morans_i.md)):

| Variable | Moran's I (k = 8) | Permutation p |
|---|---:|---:|
| Daytime SUHI (°C) | 0.740 | 0.001 |
| ACS poverty rate | 0.560 | 0.001 |
| Tract NDVI | 0.847 | 0.001 |
| Summer ΔNDVI (urban − rural) | 0.816 | 0.001 |

Under random 5-fold, 100.0 % of validation tracts have a nearest
neighbour in training; under leave-one-city-out the share is 4.5 %, and
under the UA-buffered state hold-out 0.1 %
([`results/train_neighbour_share.md`](results/train_neighbour_share.md)).

Figures: [`results/optimism_auc.png`](results/optimism_auc.png),
[`results/train_neighbour_share.png`](results/train_neighbour_share.png).

The outcomes and the predictors are strongly autocorrelated, and a random split leaves nearly
every validation tract beside a training tract. That is why held-out validation is reported
throughout, with leave-one-city-out as the headline.
