# Chi-squared heatmap

`chi_squared_heatmap.py` draws the 2×2 heatmap of SUHI class against poverty class. It
reads every number from section 2.2
([`2_Classification_Thresholds/2.2_Chi_Squared_Median/results/chi_squared_results.json`](../2_Classification_Thresholds/2.2_Chi_Squared_Median/results/chi_squared_results.json)) and does not re-estimate
anything.

- **Sample and thresholds.** The table is the primary national sample (n = 54,243 tracts in
  486 urbanized areas), split at the national medians that define the classifiers'
  labels.
- **Framing.** The title leads with the effect size. The statistics sit directly under it
  in bold: the odds ratio with its UA cluster-bootstrap CI and Cramér's V, then χ² with its
  degrees of freedom, its p-value and n.
- **Dependence.** The CI resamples urbanized areas, and a footnote gives the
  Cochran–Mantel–Haenszel OR stratified by UA.

## Numbers shown

| Cell | Count | Share of n |
|---|---:|---:|
| Low SUHI, low poverty | 16,653 | 30.7 % |
| Low SUHI, high poverty | 10,468 | 19.3 % |
| High SUHI, low poverty | 10,468 | 19.3 % |
| High SUHI, high poverty | 16,654 | 30.7 % |

| Statistic | Value |
|---|---|
| Odds ratio (95 % UA cluster-bootstrap CI, B = 1,000) | 2.53 (2.25–2.89) |
| Cramér's V | 0.228 |
| χ²(1) | 2,821.4, p < 1e-16 |
| CMH OR by UA (95 % CI) | 3.79 (3.64–3.95) |
| n | 54,243 |

All values: [`results/chi_squared_heatmap.csv`](results/chi_squared_heatmap.csv).
Figure: [`results/chi_squared_heatmap.png`](results/chi_squared_heatmap.png).
