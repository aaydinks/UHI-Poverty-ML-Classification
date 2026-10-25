# UHI-Poverty-ML-Classification

Code and results for research paper *Urban Heat Intensity: Association, Prediction, and Impact on
Impoverished Areas*. The analysis classifies U.S. census tracts by surface
urban heat island (SUHI) intensity and by poverty, and tests their association, on a national
scale of 54,243 tracts in 486 urbanized areas across the 50 states and DC.

Every numbered analysis has a folder containing one script, a `results/` folder, and a README
whose numbers are written by the script from its results.

| Folder | Contents |
|---|---|
| [`1_Geographical_Scope/`](1_Geographical_Scope/) | National scope, heat maps, tract accounting, coverage, large-metro influence |
| [`2_Classification_Thresholds/`](2_Classification_Thresholds/) | Median splits and their precedent; the chi-squared test at the models' cut-offs |
| [`3_UHI_Model_Predictors/`](3_UHI_Model_Predictors/) | The UHI formula; the UHI model without any temperature input |
| [`4_Model_Specifications/`](4_Model_Specifications/) | Spatially grouped validation, confusion matrices, metric panel with CIs, baselines, nested-tuned models and a soft-voting ensemble |
| [`5_SHAP_Interpretation/`](5_SHAP_Interpretation/) | The model SHAP explains; grouped attribution and its stability |
| [`6_Figures/`](6_Figures/) | The paper's figures, each with its script, the numbers it shows, and a README |
| [`data/`](data/) | Sources, manifest and the API-key instructions |
| [`src/uhi_poverty/`](src/uhi_poverty/) | Shared code imported by every script |
| [`tests/`](tests/) | Unit tests on synthetic fixtures, plus data checks that skip without `data/` |

## Shared code

| Module | Responsibility |
|---|---|
| `paths.py` | Repository and data locations (`UHI_DATA_DIR`, `UHI_SOURCE_DIR` overrides) |
| `acquire.py` | Downloads and copies with hash checks, Census API calls with the key redacted, `data/manifest.csv` |
| `geo.py` | GEOID normalisation, the state/region table, urbanized-area helpers |
| `build.py` | The analysis sample, exclusion flow, and median values |
| `labels.py` | `High_UHI`, `High_Pov` and `High_Pov_LPI`, with orientation checks |
| `features.py` | Feature families and the named configurations (`A-noLST`, `B-env`, …) |
| `models.py` | Majority class, logistic regression and LightGBM with fixed settings; the tuned pipelines (XGBoost, CatBoost and the optional preprocessing steps) |
| `validation.py` | Validation splitters and the cached out-of-fold engine |
| `metrics.py` | Confusion counts with denominators, the metric panel, UA cluster bootstrap, paired differences |
| `association.py` | 2×2 statistics, CMH by UA, cluster-bootstrap OR, clustered logistic regression |
| `spatial.py` | Nearest neighbours, Moran's I, train-neighbour exposure |
| `tuning.py` | Nested Optuna tuning, in-fold thresholds, the soft-voting ensemble and its cache (section 4.6) |
| `interpret.py` | Out-of-fold TreeSHAP, grouped permutation importance and ALE for the evaluated LightGBM models, cached |
| `maps.py`, `plotting.py`, `report.py` | Map layout, figure style, Markdown and number formatting |
