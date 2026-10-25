# 5.2 Attribution and stability

## Purpose

Ranks the input groups by their contribution to the two evaluated models' out-of-fold
predictions, and checks whether those ranks hold across refits. SHAP values describe
contributions to a fitted model's predictions, not causes.

## Method

- `attribution.py` reads the out-of-fold attributions of 5.1 for the two evaluated models:
  Task A, `High_UHI` from `A-noLST` (no temperature input), and Task B, `High_Pov` (ACS
  poverty rate) from `B-env`.
- Importance per attribution group: the mean over tracts of |sum of the group's SHAP values|,
  pooled over the held-out folds. Groups keep every exact identity whole (5.1).
- Grouped permutation importance: the drop in held-out ROC-AUC when all of a group's columns are
  permuted together, a check that does not depend on SHAP.
- Stability: the folds were redrawn with seeds 38 and 39, giving 15 fold models. A group's rank
  is interpreted only if it moves by at most 2 places across all 15
  (this rule was fixed before the results were seen).

## Results

**Task A (`High_UHI`, no temperature input)**

| Rank | Group | Columns | Mean \|Σ SHAP\| | Share | Rank across 15 models | Stable | Δ ROC-AUC when permuted |
|---:|---|---:|---:|---:|---|---|---:|
| 1 | Vegetation: summer | 6 | 1.542 | 38.5 % | 1–1 | yes | 0.157 |
| 2 | Tract size | 4 | 1.041 | 26.0 % | 2–2 | yes | 0.065 |
| 3 | Water share | 1 | 0.448 | 11.2 % | 3–4 | yes | 0.035 |
| 4 | Vegetation: annual | 6 | 0.421 | 10.5 % | 3–5 | yes | 0.012 |
| 5 | Elevation | 5 | 0.292 | 7.3 % | 3–6 | no | 0.007 |
| 6 | Vegetation: winter | 6 | 0.257 | 6.4 % | 5–6 | yes | 0.006 |

**Task B (`High_Pov`, ACS poverty)**

| Rank | Group | Columns | Mean \|Σ SHAP\| | Share | Rank across 15 models | Stable | Δ ROC-AUC when permuted |
|---:|---|---:|---:|---:|---|---|---:|
| 1 | Tract size | 4 | 0.432 | 19.4 % | 1–1 | yes | 0.049 |
| 2 | Vegetation: winter | 6 | 0.294 | 13.2 % | 2–4 | yes | 0.032 |
| 3 | Vegetation: summer | 6 | 0.246 | 11.0 % | 2–4 | yes | 0.016 |
| 4 | Vegetation: annual | 6 | 0.233 | 10.5 % | 2–6 | no | 0.014 |
| 5 | Elevation | 5 | 0.223 | 10.0 % | 2–6 | no | 0.023 |
| 6 | Heat: summer night | 6 | 0.178 | 8.0 % | 5–8 | no | 0.015 |
| 7 | Heat: winter day | 6 | 0.158 | 7.1 % | 5–8 | no | 0.007 |
| 8 | Heat: annual day | 6 | 0.110 | 4.9 % | 7–11 | no | 0.006 |
| 9 | Heat: summer day | 6 | 0.107 | 4.8 % | 7–10 | no | 0.007 |
| 10 | Heat: winter night | 6 | 0.093 | 4.2 % | 7–11 | no | 0.001 |
| 11 | Heat: annual night | 6 | 0.092 | 4.1 % | 8–12 | no | 0.005 |
| 12 | Water share | 1 | 0.064 | 2.9 % | 11–12 | yes | 0.003 |

Group importances agree closely across the 15 fold models (mean pairwise Spearman
0.935 on Task A and
0.925 on Task B; minimum
0.657 and 0.832;
[`results/stability.md`](results/stability.md)).

- **Task A.** Vegetation: summer ranks first in every fold model
  (38.5 % of the grouped attribution), followed by
  tract size. With every temperature input removed, the model relies on
  vegetation and tract geometry.
- **Task B.** Tract size ranks first in every fold model
  (19.4 %). The six heat groups together carry
  33.1 % of the grouped attribution, but none of them holds a stable
  rank (ranks 5–12). The stable groups are
  tract size, winter vegetation, summer vegetation and water share.

**Attribution is not added value.** A heat group can receive attribution because it substitutes
for correlated non-thermal inputs. Removing every heat variable from the poverty model changes
its ROC-AUC by 0.008 [-0.002,
0.018] under UA-grouped 5-fold and by 0.009
[-0.002, 0.019] under leave-one-city-out
(section 4.5). The heat attributions in Task B therefore do not show that heat carries
information about poverty beyond vegetation, elevation and tract geometry.

Figures: [`results/group_importance.png`](results/group_importance.png),
[`results/shap_summary.png`](results/shap_summary.png) (per-feature values; the credit within an
identity is arbitrary). Per-feature means are in
[`results/feature_importance.md`](results/feature_importance.md).
