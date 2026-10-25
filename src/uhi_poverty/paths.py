"""Repository and data locations.

`UHI_DATA_DIR` moves the whole data tree (raw, interim, processed, manifest).
`UHI_SOURCE_DIR` points at the folder holding the two local source CSVs.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("UHI_DATA_DIR", ROOT / "data"))
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MANIFEST = DATA / "manifest.csv"

ANALYSIS_PARQUET = PROCESSED / "analysis_tracts.parquet"
EXCLUSION_FLOW = PROCESSED / "exclusion_flow.csv"
MEDIANS_JSON = PROCESSED / "median_values.json"
OOF_DIR = INTERIM / "oof"
INTERPRET_DIR = INTERIM / "interpret"
TUNED_DIR = INTERIM / "tuned"

SOURCE_DIR = Path(os.environ.get("UHI_SOURCE_DIR", Path.home() / "Downloads"))

SUHI_CSV = RAW / "suhi" / "Census_UHI_US_Urbanized_recalculated.csv"
LPI_CSV = RAW / "hud_lpi" / "Low Poverty Index.csv"
ACS_DIR = RAW / "acs2019"
TRACTS_DIR = RAW / "cb2019_tracts"
STATES_ZIP = RAW / "cb2019_states" / "cb_2019_us_state_500k.zip"
GAZ_2019 = RAW / "gazetteer2019" / "2019_Gaz_tracts_national.zip"
SVI_CSV = RAW / "svi2018" / "SVI_2018_US.csv"


def rel(path: Path) -> str:
    """Path relative to the repo root (POSIX), or with the home folder shortened to `~`."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return "~/" + path.relative_to(Path.home()).as_posix()
        except ValueError:
            return path.name


def results_dir(script_file: str) -> Path:
    """The `results/` folder next to an analysis script, created on demand."""
    out = Path(script_file).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out
