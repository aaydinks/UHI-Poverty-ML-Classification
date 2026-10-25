# 1 Geographical scope

The study area, a national map, a full account of included and excluded tracts by state and
city, and whether a few large metros drive the results. Every analysis uses one national
sample.

| Section | Folder | Script | Main outputs |
|---|---|---|---|
| 1.1 National focus | [`1.1_National_Focus/`](1.1_National_Focus/) | `scope_summary.py` | scope in numbers; study-area description |
| 1.2 National heat map | [`Heatmap/`](Heatmap/) | `make_heatmap.py` | F1 tract map, F2 urbanized-area map, F3 regional metro panels, F4 summer LST map |
| 1.3 Tracts used and excluded | [`1.3_Tract_Accounting/`](1.3_Tract_Accounting/) | `tract_accounting.py` | exclusion flow and diagram; exclusions by state and reason; cluster sizes |
| 1.4 Coverage by state and urban area | [`1.4_Coverage_States_and_Urban_Areas/`](1.4_Coverage_States_and_Urban_Areas/) | `coverage.py` | tracts by state (all 51) and by urbanized area; top-50 UAs; included/excluded lists; coverage maps |
| 1.5 Large-metro influence | [`1.5_Large_Metro_Influence/`](1.5_Large_Metro_Influence/) | `metro_influence.py` | concentration; models and association without the largest UAs; pooled vs per-city metrics |

Run with `make scope` (1.5 reads the cached out-of-fold predictions; run `make engine` first).
Each README ends with a summary of its part of the analysis. All numbers in those READMEs are
written by the scripts from their `results/` files.
