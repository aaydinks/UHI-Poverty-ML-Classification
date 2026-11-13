# UHI-Poverty-ML-Classification

Code and results for *Urban Heat Intensity: Association, Prediction, and Impact on Impoverished
Areas*. The analysis classifies U.S. census tracts (2010 definitions) in Census urbanized areas of
the 50 states and the District of Columbia by surface urban heat island (SUHI) intensity and by
poverty rate, and tests the association between the two.

## Layout

Each analysis folder holds one script and a `results/` folder of CSV, JSON and PNG outputs.

| Folder | Contents |
|---|---|
| `1_Geographical_Scope/` | Study-area summary (1.1), national maps (`Heatmap`), tract exclusion flow (1.3), coverage by state and urbanized area (1.4), analyses without the largest urbanized areas (1.5) |
| `2_Classification_Thresholds/` | Median cut-offs and class counts (2.1); chi-squared test, effect sizes and sensitivity analyses (2.2) |
| `3_UHI_Model_Predictors/` | UHI difference identities (3.1); predictor sets and the ablation (3.2) |
| `4_Model_Specifications/` | Grouped validation (4.1), random against held-out validation and Moran's I (4.2), confusion matrices (4.3), metric panel with confidence intervals (4.4), baselines (4.5), tuned models and soft-voting ensemble (4.6), every model in one panel (4.7) |
| `5_SHAP_Interpretation/` | The explained model and attribution groups (5.1); grouped attribution and its stability (5.2) |
| `6_Figures/` | Scripts and outputs for the paper's figures |
| `data/` | Data sources, download manifest and small processed files |
| `src/uhi_poverty/` | Shared code imported by every script |
| `tests/` | Unit tests; data checks skip when `data/` is absent |

## Reproducing the results

Requirements: Python 3.12 or 3.13, [uv](https://docs.astral.sh/uv/), `make`, and a free Census API
key from <https://api.census.gov/data/key_signup.html> in a `.env` file at the repository root:

```
CENSUS_API_KEY=your-key
```

Install the locked environment, then run the steps in order:

```bash
uv sync
make data         # download the six sources into data/raw and write data/manifest.csv
make build        # build the analysis sample in data/processed
make engine       # out-of-fold predictions of the fixed-setting models (cached in data/interim)
make models       # 4.1-4.5
make predictors   # 3.1-3.2
make thresholds   # 2.1-2.2
make scope        # 1.1-1.5 and the national maps
make interpret    # out-of-fold SHAP values (cached)
make shap         # 5.1-5.2
make tune         # nested tuning (cached; several hours)
make tuned        # 4.6
make export       # the final tuned models in 4.6/models
make figures      # 6_Figures
make panel        # 4.7
make test
```

`make all` runs the same sequence. Downloads are reused when present, the SUHI database and the
Low Poverty Index are verified against pinned SHA-256 values, and every downloaded file's hash is
recorded in `data/manifest.csv`. Every cache under `data/interim/` is keyed to its inputs and
settings and is recomputed when they change. All random steps use seed 37. `UHI_DATA_DIR` moves the data tree.

## Licence

The code is released under the MIT licence ([`LICENSE`](LICENSE)). The source datasets keep their
own terms: the SUHI database is CC BY 4.0; the Census and HUD files are U.S. Government works.
