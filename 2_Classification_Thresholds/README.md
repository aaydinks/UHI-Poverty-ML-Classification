# 2 Classification thresholds

How both outcomes are dichotomised, the class counts, and the chi-squared analysis of their
association at those cut-offs.

| Section | Folder | Script | Main outputs |
|---|---|---|---|
| 2.1 Median split and precedent | [`2.1_Median_Precedent/`](2.1_Median_Precedent/) | `medians.py` | national cut values and class counts; 66th/75th-percentile sensitivity cuts; the Smargiassi et al. (2009) precedent |
| 2.2 Chi-squared at the models' median | [`2.2_Chi_Squared_Median/`](2.2_Chi_Squared_Median/) | `chi_squared.py` | 2×2 table, χ², Cramér's V, OR with CIs, CMH by UA, cluster-bootstrap and clustered-SE versions, sensitivity analyses |

Run with `make thresholds`.

One rule holds everywhere: `High_UHI` and `High_Pov` are national median splits of the
primary sample, defined once in `src/uhi_poverty/labels.py`. The classifiers (sections 3 and 4)
and the chi-squared test import the same labels, so they always use the same cut-offs.
