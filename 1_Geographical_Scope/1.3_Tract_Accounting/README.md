# 1.3 Tracts used and excluded

## Purpose

Accounts for every tract: which were analysed, which were excluded and why, and how regional
differences and the clustering of tracts in cities are handled.

## Method

`tract_accounting.py` reports the sample-construction steps built once by `make build`
(`src/uhi_poverty/build.py`), breaks the exclusions down by state and reason, and describes how
tracts cluster in urbanized areas. Every merge is one-to-one on 2010 tract codes, and the counts
reconcile (the build fails otherwise).

## Results

[`results/exclusion_flow.md`](results/exclusion_flow.md), drawn in
[`results/flow_diagram.png`](results/flow_diagram.png):

| Step | Rule | Tracts | Removed |
|---|---|---:|---:|
| 0 | All 2010 tracts, 50 states + DC (2019 Gazetteer; coverage universe) | 73,056 |  |
| 1 | In the SUHI database, 50 states + DC (urbanized-area tracts only) | 54,998 | 18,067 |
| 2 | 50 states + DC (Puerto Rico held out as the sensitivity frame) | 54,998 | 873 |
| 3 | Complete SUHI target and features | 54,761 | 237 |
| 3g | …with Gazetteer geometry (water_frac, log_aland) | 54,752 | 9 |
| 4 | Matched to ACS 2015–2019 with poverty universe >= 50 | 54,243 | 509 |
| 5 | Primary analysis sample | 54,243 |  |
| 5a | …with HUD LPI present (robustness frame) | 54,237 | 6 |
| 5b | …plus Puerto Rico (sensitivity frame) | 55,095 |  |

By reason, over the 50 states + DC and Puerto Rico
([`results/excluded_by_state_and_reason.csv`](results/excluded_by_state_and_reason.csv)):

| Reason | Tracts |
|---|---:|
| Not urban: no SUHI value (source limitation) | 18,139 |
| Territory (Puerto Rico; kept as the sensitivity frame) | 873 |
| Missing SUHI fields | 237 |
| Not in the 2019 Gazetteer (tract renumbered after 2010) | 9 |
| ACS unmatched or poverty universe < 50 persons | 509 |
| **Included (primary sample)** | **54,243** |

Of the ACS exclusions, 509 tracts have a poverty universe below 50
persons (median total population 0;
71.3 % have fewer than 100 residents);
0 more have no ACS row (the SUHI tracts without one
were already removed as renumbered after 2010). The states with the most ACS exclusions are
NY 76, CA 53, TX 44.

**Clustering** ([`results/cluster_sizes.csv`](results/cluster_sizes.csv)). The
54,243 tracts are nested in 486 urbanized areas. The median UA has
35 tracts (interquartile range 22–76; largest
4,217). The largest UA holds 7.8 % of all tracts and the ten
largest hold 31.2 %. 209 UAs have fewer than 30
tracts (4,295 tracts in all).

## How limitations, regions and clustering are handled

- **Limitations.** Rural tracts have no SUHI value by construction: the database only covers
  tracts inside Census urbanized areas, so results describe urban tracts.
- **Regional differences.** No tract is dropped because of its region. Instead, regional
  transfer is tested by holding out whole Census regions and states (section 4.1).
- **Clustering.** Tracts are nested in 486 UAs. This is handled by UA-grouped
  validation (leave-one-city-out is the headline) and by confidence intervals from a bootstrap
  that resamples UAs (sections 2.2 and 4.4). UAs with fewer than 30 tracts, or with only one
  class, stay in every pooled metric but get no per-city metric.
