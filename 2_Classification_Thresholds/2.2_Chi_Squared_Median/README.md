# 2.2 Chi-squared test at the models' median

## Purpose

Tests the association between high SUHI and high poverty at the same cut-offs the classifiers
use, with the full 2×2 table, effect sizes with confidence intervals, and an account of the
spatial dependence between tracts in the same city.

## Method

`chi_squared.py` takes `High_UHI` and `High_Pov` from the built sample, where
`src/uhi_poverty/labels.py` defines them, so the cut-offs are exactly the classifiers'
national medians. It reports:

- class counts, the 2×2 table with row and column percentages and expected counts
  ([`results/contingency_table.md`](results/contingency_table.md));
- χ² (df = 1) with and without Yates' correction, φ / Cramér's V, the odds ratio with a Woolf
  CI, and the risk ratio and risk difference;
- dependence-aware versions: a Cochran–Mantel–Haenszel common OR stratified by UA with the
  Breslow–Day test, a UA cluster-bootstrap CI for the OR (B = 1,000), and a logistic regression
  with UA-clustered standard errors;
- sensitivity analyses: the LPI-based poverty label, a 20 % "poverty area" cut-off, SUHI at the
  66th and 75th percentiles, both outcomes at the 75th percentile, Puerto Rico included, and the
  ten largest UAs excluded.

## Results

**Primary sample** (n = 54,243 tracts; [`results/chi_squared_results.md`](results/chi_squared_results.md)).
High-SUHI tracts have **153 % higher odds** of being high-poverty tracts than low-SUHI
tracts: OR = 2.53 (95 % CI 2.44–2.62;
UA cluster-bootstrap CI 2.25–2.89, i.e.
125–189 % higher odds). In probabilities, 61.4 %
of high-SUHI tracts are high-poverty against 38.6 %
of low-SUHI tracts (risk difference 0.228, risk ratio
1.59). Cramér's V = 0.228; χ²(1) =
2,821.4, p < 1e-16.

**Dependence.** Within urbanized areas the association is larger, not smaller: the CMH OR
stratified by UA is 3.79 (3.64–3.95).
The Breslow–Day test rejects a common OR across UAs (p < 1e-16), so
the strength of the association varies from city to city. The logistic regression with
UA-clustered standard errors gives OR 2.53
(2.22–2.89).

**Sensitivity.** Across the sensitivity analyses the OR ranges from 2.48 to
2.74 ([`results/sensitivity.csv`](results/sensitivity.csv)); none reverses the direction.

Figure: [`results/conditional_probabilities.png`](results/conditional_probabilities.png).
