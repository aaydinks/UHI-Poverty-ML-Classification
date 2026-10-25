from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uhi_poverty import paths

HAS_DATA = paths.RAW.exists() and paths.SUHI_CSV.exists()
HAS_BUILD = paths.ANALYSIS_PARQUET.exists()

needs_data = pytest.mark.skipif(not HAS_DATA, reason="data/raw is absent (run `make data`)")
needs_build = pytest.mark.skipif(not HAS_BUILD, reason="analysis sample absent (run `make build`)")


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    """A small frame of tracts nested in UAs that span counties, states and regions."""
    rng = np.random.default_rng(0)
    rows = []
    # (ua, [(state_fips, region, n_tracts), ...]); UA 00003 spans two states
    layout = [
        ("00001", [("06", "West", 30)]),
        ("00002", [("06", "West", 12)]),
        ("00003", [("32", "West", 10), ("06", "West", 8)]),
        ("00004", [("36", "Northeast", 40)]),
        ("00005", [("34", "Northeast", 9), ("36", "Northeast", 6)]),
        ("00006", [("17", "Midwest", 25)]),
        ("00007", [("48", "South", 35)]),
        ("00008", [("12", "South", 14)]),
        ("00009", [("48", "South", 7)]),
    ]
    serial = 0
    for ua, parts in layout:
        for state, region, n in parts:
            for i in range(n):
                serial += 1
                county = f"{state}{(i % 3) * 2 + 1:03d}"
                rows.append({"GEOID": f"{county}{serial:06d}", "ua": ua, "state_fips": state,
                             "county_fips": county, "region": region})
    df = pd.DataFrame(rows)
    df["x1"] = rng.normal(size=len(df))
    df["x2"] = rng.normal(size=len(df))
    df["y"] = (df["x1"] + 0.5 * rng.normal(size=len(df)) > 0).astype(int)
    return df
