# Confusion matrices by city

Two figures, one for the UHI model and one for the poverty model, each show ten confusion
matrices in a 2 × 5 grid, with the count and percentage in each cell, F1 · Accuracy / MCC above
each panel, and the F1 spread in the header.

## Design

- **Panels.** Each panel is one of the ten largest urbanized areas by tract count
  (New York, Los Angeles, Chicago, Philadelphia, Miami, Detroit, Washington, Dallas, Boston, Houston), held out from training under leave-one-city-out validation. The
  model never saw a tract of the city it is scored on.
- **Model and labels.** The cached leave-one-city-out predictions of the headline LightGBM
  (`validation.oof(task, config, "lgbm", "loco")`): Task A `A-noLST` (no LST or SUHI
  predictors) for `High_UHI`, Task B `B-env` for `High_Pov`, both at τ = 0.5. Poverty is the ACS
  poverty rate against the national median.
- **Denominators** (per
  [`4.3 DEFINITIONS.md`](../4_Model_Specifications/4.3_Confusion_Matrices_and_Denominators/DEFINITIONS.md)).
  The cell percentage is the count over the city's tracts, and the cell says so; the four cells
  sum to 100 %. TPR (TP / actual high) and TNR (TN / actual low) are in each panel's subtitle
  with their numerators and denominators.
- **Header.** The pooled leave-one-city-out F1 over all 54,243 tracts and
  486 UAs, with its 95 % UA cluster-bootstrap CI (B = 1,000), then the mean,
  SD and range of F1 across the ten cities. The script checks the pooled estimate and CI
  against
  [`4.4 metric_panel.csv`](../4_Model_Specifications/4.4_Metric_Panel_with_Uncertainty/results/metric_panel.csv)
  to 1e-9.
- **Shading** is the cell's share of the city's tracts on one scale for all ten panels, so
  panels can be compared. The UHI panels are blue and the poverty panels orange.

## Results

| Panels | Task | Tracts | Pooled F1 [95 % CI] | Pooled MCC [95 % CI] | F1 across the 10 cities |
|---|---|---:|---|---|---|
| UHI | A `A-noLST` | 54,243 | 0.819 [0.768, 0.856] | 0.647 [0.566, 0.709] | mean 0.833 · SD 0.171 · range [0.363, 0.924] |
| Poverty | B `B-env` | 54,243 | 0.725 [0.707, 0.742] | 0.414 [0.385, 0.445] | mean 0.687 · SD 0.119 · range [0.421, 0.796] |

| Panels | Rank | City | n | TN | FP | FN | TP | TPR | TNR | F1 | Acc | MCC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UHI | 1 | New York | 4,217 | 771 | 383 | 227 | 2,836 | 0.926 (2,836/3,063) | 0.668 (771/1,154) | 0.903 | 0.855 | 0.623 |
| UHI | 2 | Los Angeles | 2,758 | 626 | 5 | 1,655 | 472 | 0.222 (472/2,127) | 0.992 (626/631) | 0.363 | 0.398 | 0.238 |
| UHI | 3 | Chicago | 2,059 | 359 | 297 | 28 | 1,375 | 0.980 (1,375/1,403) | 0.547 (359/656) | 0.894 | 0.842 | 0.629 |
| UHI | 4 | Philadelphia | 1,395 | 455 | 102 | 57 | 781 | 0.932 (781/838) | 0.817 (455/557) | 0.908 | 0.886 | 0.761 |
| UHI | 5 | Miami | 1,188 | 321 | 89 | 214 | 564 | 0.725 (564/778) | 0.783 (321/410) | 0.788 | 0.745 | 0.485 |
| UHI | 6 | Detroit | 1,148 | 262 | 30 | 104 | 752 | 0.879 (752/856) | 0.897 (262/292) | 0.918 | 0.883 | 0.725 |
| UHI | 7 | Washington | 1,142 | 377 | 124 | 35 | 606 | 0.945 (606/641) | 0.752 (377/501) | 0.884 | 0.861 | 0.721 |
| UHI | 8 | Dallas | 1,118 | 370 | 152 | 63 | 533 | 0.894 (533/596) | 0.709 (370/522) | 0.832 | 0.808 | 0.618 |
| UHI | 9 | Boston | 970 | 346 | 47 | 47 | 530 | 0.919 (530/577) | 0.880 (346/393) | 0.919 | 0.903 | 0.799 |
| UHI | 10 | Houston | 903 | 145 | 67 | 40 | 651 | 0.942 (651/691) | 0.684 (145/212) | 0.924 | 0.882 | 0.657 |
| Poverty | 1 | New York | 4,217 | 1,140 | 1,157 | 242 | 1,678 | 0.874 (1,678/1,920) | 0.496 (1,140/2,297) | 0.706 | 0.668 | 0.393 |
| Poverty | 2 | Los Angeles | 2,758 | 719 | 460 | 231 | 1,348 | 0.854 (1,348/1,579) | 0.610 (719/1,179) | 0.796 | 0.749 | 0.483 |
| Poverty | 3 | Chicago | 2,059 | 750 | 348 | 193 | 768 | 0.799 (768/961) | 0.683 (750/1,098) | 0.740 | 0.737 | 0.483 |
| Poverty | 4 | Philadelphia | 1,395 | 684 | 141 | 134 | 436 | 0.765 (436/570) | 0.829 (684/825) | 0.760 | 0.803 | 0.593 |
| Poverty | 5 | Miami | 1,188 | 467 | 30 | 499 | 192 | 0.278 (192/691) | 0.940 (467/497) | 0.421 | 0.555 | 0.275 |
| Poverty | 6 | Detroit | 1,148 | 285 | 251 | 45 | 567 | 0.926 (567/612) | 0.532 (285/536) | 0.793 | 0.742 | 0.505 |
| Poverty | 7 | Washington | 1,142 | 504 | 334 | 61 | 243 | 0.799 (243/304) | 0.601 (504/838) | 0.552 | 0.654 | 0.354 |
| Poverty | 8 | Dallas | 1,118 | 339 | 234 | 178 | 367 | 0.673 (367/545) | 0.592 (339/573) | 0.640 | 0.631 | 0.266 |
| Poverty | 9 | Boston | 970 | 470 | 183 | 33 | 284 | 0.896 (284/317) | 0.720 (470/653) | 0.724 | 0.777 | 0.578 |
| Poverty | 10 | Houston | 903 | 168 | 195 | 112 | 428 | 0.793 (428/540) | 0.463 (168/363) | 0.736 | 0.660 | 0.271 |

Figures: [`results/confusion_by_city_uhi.png`](results/confusion_by_city_uhi.png),
[`results/confusion_by_city_poverty.png`](results/confusion_by_city_poverty.png). Tables:
[`results/confusion_by_city_panels.csv`](results/confusion_by_city_panels.csv),
[`results/confusion_by_city_pooled.csv`](results/confusion_by_city_pooled.csv).

Rebuild: `uv run python 6_Figures/confusion_by_city.py` (reads the
out-of-fold cache that `make engine` fills).
