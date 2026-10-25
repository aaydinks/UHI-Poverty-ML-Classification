# 1.5 Large-metro influence

## Purpose

Tests whether a handful of very large metropolitan areas dominate the sample and drive the
results.

## Method

`metro_influence.py` reuses the cached out-of-fold predictions to:

1. measure how concentrated tracts are in the largest urbanized areas (UAs);
2. retrain and re-evaluate the primary Task A (`A-noLST`) and Task B (`B-env`) models, logistic
   regression and LightGBM, under UA-grouped 5-fold with all UAs and without the largest 1, 3
   and 10 UAs;
3. compare pooled leave-one-city-out metrics (each tract counts once, so big UAs weigh more)
   with the mean across cities (each UA counts once), using UAs with ≥ 30 tracts and both
   classes;
4. recompute the association of section 2.2 (OR and CMH OR) without the largest 1, 3 and 10 UAs.

## Results

**Concentration** ([`results/concentration.png`](results/concentration.png)). The largest UA
(New York--Newark, NY--NJ--CT) holds 7.8 % of the
54,243 tracts; the largest 5 hold
21.4 %, the largest 10
31.2 % and the largest 20
43.8 % (Gini 0.69 across 486 UAs).

**Models without the largest UAs** (ROC-AUC, UA-grouped 5-fold;
[`results/metro_sensitivity.md`](results/metro_sensitivity.md),
[`results/metro_sensitivity.png`](results/metro_sensitivity.png)):

| Task | Model | All UAs | Top 1 out | Top 3 out | Top 10 out |
|---|---|---:|---:|---:|---:|
| A | Logistic regression | 0.898 | 0.899 | 0.902 | 0.901 |
| A | LightGBM | 0.908 | 0.909 | 0.920 | 0.915 |
| B | Logistic regression | 0.712 | 0.719 | 0.749 | 0.749 |
| B | LightGBM | 0.772 | 0.777 | 0.772 | 0.778 |

**Pooled vs per-city** (leave-one-city-out; per-city = ROC-AUC within each held-out UA):

| Task | Model | Pooled ROC-AUC | Mean across cities | Median across cities [IQR] | Median, 10 largest UAs | Median, other UAs | Eligible UAs |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Logistic regression | 0.899 | 0.894 | 0.936 [0.861, 0.967] | 0.902 | 0.938 | 272 |
| A | LightGBM | 0.910 | 0.922 | 0.943 [0.892, 0.973] | 0.911 | 0.943 | 272 |
| B | Logistic regression | 0.714 | 0.748 | 0.774 [0.677, 0.833] | 0.782 | 0.774 | 277 |
| B | LightGBM | 0.777 | 0.771 | 0.794 [0.707, 0.846] | 0.765 | 0.795 | 277 |

The median ROC-AUC *within* a held-out city is higher than the pooled figure for every model (LightGBM: Task A 0.943 vs pooled 0.910; Task B 0.794 vs 0.777). The models rank tracts inside an unseen city better than they place cities relative to each other, so pooled numbers understate within-city discrimination rather than overstate it.

**Association without the largest UAs.** The odds ratio stays between 2.53 and
2.62 and the CMH OR between 3.79 and 4.03 whether
the largest 0, 1, 3 or 10 UAs are excluded.

**Conclusion.** No conclusion depends on a few metros. Removing the 1, 3 or 10 largest UAs never lowers ROC-AUC; the largest change is a rise of 0.037 (logistic regression, Task B, top 3 out), so the large metros make prediction slightly harder, not easier. The association keeps its direction and size, and the largest UAs are not the easiest cities to predict (3 of 4 model × task pairs have a lower median within-city ROC-AUC in the ten largest UAs than in the rest).
