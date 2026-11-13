# Data

`make data` fills `data/raw/<source>/` and writes [`manifest.csv`](manifest.csv): source, URL or
original local path, retrieval date, bytes, SHA-256 and licence for every file. `make build`
turns the raw files into `data/processed/analysis_tracts.parquet` plus the small tracked files
listed below. `data/raw/`, `data/interim/` and every `*.parquet` file are gitignored; rerun the
two targets to rebuild them.

## Sources

| ID | Dataset | Where it comes from | Vintage | Role |
|---|---|---|---|---|
| S1 | SUHI database (Chakraborty et al., 2020) | Mendeley Data `x9mv4krnm2` v3, file `Census_UHI_US_Urbanized_recalculated.csv` (CC BY 4.0). Copied from the local download, which must hash to Mendeley's published SHA-256 `7c0c39af…a25585`; acquisition stops otherwise. | 2010 tracts, 2010 UAs | Heat, vegetation, elevation and morphology features; UHI target; UA group key |
| S2 | ACS 2015–2019 5-year: `B17001_001E/M`, `B17001_002E/M`, `B01003_001E` | Census API `https://api.census.gov/data/2019/acs/acs5`, one call per state (50 + DC); raw JSON cached per state | 2010 tracts (the last 5-year release on 2010 geography) | Primary poverty outcome |
| S3 | HUD AFFH Low Poverty Index, `AFFHT0006` | HUD Open Data, copied from the local download | 2010 tracts; ACS 2011–2015 | Robustness outcome (low index = poorer) |
| S4 | Cartographic boundary tracts 2019, 1:500k | `https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_{SS}_tract_500k.zip`, 51 files | 2010 tracts | Maps |
| S5 | Cartographic boundary states 2019, 1:500k | `…/GENZ2019/shp/cb_2019_us_state_500k.zip` | – | Map outlines |
| S6 | 2019 Gazetteer, tracts | `https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2019_Gazetteer/2019_Gaz_tracts_national.zip` | 2010 tracts | Centroids, land and water area, all-tract universe |

`make data` reads the two local CSVs (S1 and S3) from `~/Downloads` by default; set
`UHI_SOURCE_DIR` to point elsewhere. `UHI_DATA_DIR` moves the whole data tree.

ACS sentinel values are handled on read: negative estimates (Census uses them for "not
available") become missing, and a margin of error of `-555555555` ("no sampling error") becomes 0.

## Census API key

The ACS calls need a free key from <https://api.census.gov/data/key_signup.html>. Put it in a
`.env` file at the repo root (gitignored):

```
CENSUS_API_KEY=your-key
```

or export it in your shell. The key is read from the environment only; request URLs are
redacted before they reach a log line or the manifest. Without a key,
`uv run python -m uhi_poverty.acquire --poverty-source svi2018` falls back to CDC SVI 2018
`EP_POV` (ACS 2014–2018), which is logged as a deviation.

## Tracked outputs of `make build`

| File | Contents |
|---|---|
| `processed/exclusion_flow.csv` | The sample-construction steps with counts |
| `processed/median_values.json` | National cut values, 66th/75th percentiles, class counts, orientation checks |
| `processed/acs_unmatched.csv` | SUHI tracts with no ACS 2015–2019 row (post-2010 tract code changes) |
