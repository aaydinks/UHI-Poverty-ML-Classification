# 4.1 Spatially grouped validation

## Purpose

This section tests whether the models generalise to places they were not trained on. Random
splits put neighbouring tracts of the same city in training and test, so every scheme here holds
out whole spatial units, with leave-one-city-out as the headline.

## Method

Both tasks (Task A: `High_UHI` on `A-noLST`; Task B: `High_Pov` on `B-env`) were scored with the
majority class, logistic regression and LightGBM (fixed settings) under six grouped
schemes. Every learned step is refitted inside each training fold, and τ = 0.5 throughout.

| Scheme | Test fold | Training data |
|---|---|---|
| County-grouped 5-fold | whole counties (`GEOID[:5]`) | other counties. Same-city neighbours can still sit across a county line, so this scheme is a lower bar. |
| UA-grouped 5-fold | whole urbanized areas (UAs) | other UAs, stratified by class |
| **Leave-one-city-out** | one UA | all other UAs (one fold per UA, 486 folds) |
| Leave-one-state-out | every tract in one state | only UAs with **no** tracts in that state (UA buffer), so no city is split between training and test |
| Leave-one-region-out | every tract in one Census region | only UAs with no tracts in that region (UA buffer) |
| 20 largest UAs | one of the 20 largest UAs | all other UAs |

Leave-one-city-out is the headline scheme because it measures how the models transfer to a
new city, and a UA is the unit the SUHI database defines (the "cities" of the paper).

## Results

Pooled out-of-fold ROC-AUC ([`results/pooled_metrics.md`](results/pooled_metrics.md) has the full panel):

| Task | Model | County-grouped 5-fold | UA-grouped 5-fold | Leave-one-city-out | Leave-one-state-out (UA-buffered) | Leave-one-region-out (UA-buffered) | Each of the 20 largest UAs held out | MCC, leave-one-city-out |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | Majority class | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | -0.331 |
| A | Logistic regression | 0.900 | 0.898 | 0.899 | 0.895 | 0.873 | 0.855 | 0.661 |
| A | LightGBM | 0.917 | 0.908 | 0.910 | 0.908 | 0.889 | 0.861 | 0.647 |
| B | Majority class | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | -0.219 |
| B | Logistic regression | 0.740 | 0.712 | 0.714 | 0.710 | 0.643 | 0.702 | 0.330 |
| B | LightGBM | 0.791 | 0.772 | 0.777 | 0.735 | 0.679 | 0.769 | 0.414 |

The majority class's negative MCC when one city, state, region or large UA is held out
(-0.331 on Task A under leave-one-city-out) is not an error:
removing a group tips the training majority toward the class that group has less of, so the
constant prediction runs against the held-out labels (explained in
[`../4.3_Confusion_Matrices_and_Denominators/README.md`](../4.3_Confusion_Matrices_and_Denominators/README.md)).

**Within each held-out city.** Under leave-one-city-out, the median ROC-AUC inside a UA
(UAs with ≥ 30 tracts and both classes) is 0.943 for LightGBM
and 0.936 for logistic regression on Task A, and
0.794 and 0.774 on Task B
(272 and 277 eligible UAs;
[`results/per_group_summary.md`](results/per_group_summary.md),
[`results/per_ua_loco.csv`](results/per_ua_loco.csv)). The pooled figure rewards a model for
getting the *level* of a city right; the within-city figure asks whether it ranks tracts inside
a city it has never seen.

**Regions.** Holding out a whole Census region lowers LightGBM's pooled ROC-AUC below
leave-one-city-out for both tasks (Task A 0.889 vs
0.910; Task B 0.679 vs
0.777), because the model must transfer to a climate and
settlement pattern it has not seen. The weakest held-out region is the
West for Task A (ROC-AUC 0.811) and the
South for Task B (0.631;
[`results/per_region.md`](results/per_region.md)).
Per-state results are in [`results/per_state.csv`](results/per_state.csv).

Figures: [`results/pooled_auc_by_scheme.png`](results/pooled_auc_by_scheme.png),
[`results/per_ua_auc_loco.png`](results/per_ua_auc_loco.png).
