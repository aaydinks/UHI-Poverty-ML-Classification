"""Feature families and the named feature configurations.

Every `UHI_*` column is an urban-minus-rural LST difference, so a model that
predicts `High_UHI` must drop both the `LST_*` and the `UHI_*` families to stay
free of the target. The geometry pair (`water_frac`, `log_aland`) comes from the
Gazetteer and is added to every `A-*` rung so that each rung removes only what
its name says.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_UHI = "UHI_annual_day"

LST = [
    "LST_rur_day", "LST_rur_day_summer", "LST_rur_day_winter",
    "LST_rur_night", "LST_rur_night_summer", "LST_rur_night_winter",
    "LST_urb_day_CT", "LST_urb_day_CT_act", "LST_urb_day_CT_act_summer",
    "LST_urb_day_CT_act_winter", "LST_urb_day_CT_summer", "LST_urb_day_CT_winter",
    "LST_urb_day_all", "LST_urb_day_all_summer", "LST_urb_day_all_winter",
    "LST_urb_night_CT", "LST_urb_night_CT_act", "LST_urb_night_CT_act_summer",
    "LST_urb_night_CT_act_winter", "LST_urb_night_CT_summer", "LST_urb_night_CT_winter",
    "LST_urb_night_all", "LST_urb_night_all_summer", "LST_urb_night_all_winter",
]
SUHI = [
    "UHI_annual_day", "UHI_annual_night", "UHI_summer_day", "UHI_summer_night",
    "UHI_winter_day", "UHI_winter_night",
    "UHI_annual_day_city", "UHI_annual_night_city", "UHI_summer_day_city",
    "UHI_summer_night_city", "UHI_winter_day_city", "UHI_winter_night_city",
]
NDVI = [
    "NDVI_rur", "NDVI_rur_summer", "NDVI_rur_winter",
    "NDVI_urb_CT", "NDVI_urb_CT_act", "NDVI_urb_CT_act_summer", "NDVI_urb_CT_act_winter",
    "NDVI_urb_CT_summer", "NDVI_urb_CT_winter",
    "NDVI_urb_all", "NDVI_urb_all_summer", "NDVI_urb_all_winter",
]
DELTA_NDVI = [
    "DelNDVI_annual", "DelNDVI_summer", "DelNDVI_winter",
    "DelNDVI_annual_city", "DelNDVI_summer_city", "DelNDVI_winter_city",
]
DEM = ["DEM_rur", "DEM_urb_CT", "DEM_urb_CT_act", "DEM_urb_all", "DelDEM"]
MORPHOLOGY = ["Area", "Total_Count", "Urban_Count"]
GEOMETRY = ["water_frac", "log_aland"]
LOCATION = ["lat", "lon", "region_Northeast", "region_Midwest", "region_South", "region_West"]
POVERTY = ["pov_rate"]

FAMILIES = {
    "LST": LST,
    "SUHI": SUHI,
    "NDVI": NDVI,
    "DeltaNDVI": DELTA_NDVI,
    "DEM": DEM,
    "Morphology": MORPHOLOGY,
    "Geometry": GEOMETRY,
    "Location": LOCATION,
    "Poverty": POVERTY,
}

SUHI_FEATURES = LST + SUHI + NDVI + DELTA_NDVI + DEM + MORPHOLOGY  # 62 SUHI-file columns
SKEWED = ["Area", "Total_Count", "Urban_Count"]

# Never model inputs.
IDENTIFIERS = [
    "system:index", "Census_geoid", "GEOID", "Urban_geoid", "Urban_name", "Urban_name_extended",
    "ua", "ua_name", "state_fips", "state_abbr", "state_name", "county_fips", "tract",
    "region", "division", "ua_primary_state",
]

NON_THERMAL = NDVI + DELTA_NDVI + DEM + MORPHOLOGY + GEOMETRY  # 28

SEASONS = {"annual": "", "summer": "_summer", "winter": "_winter"}

# Exact difference identities in the SUHI file, (difference, urban term, rural
# term), each verified to floating-point precision by section 3.1.
IDENTITIES = (
    [(f"UHI_{s}_{t}", f"LST_urb_{t}_CT{x}", f"LST_rur_{t}{x}")
     for s, x in SEASONS.items() for t in ("day", "night")]
    + [(f"UHI_{s}_{t}_city", f"LST_urb_{t}_all{x}", f"LST_rur_{t}{x}")
       for s, x in SEASONS.items() for t in ("day", "night")]
    + [(f"DelNDVI_{s}", f"NDVI_urb_CT{x}", f"NDVI_rur{x}") for s, x in SEASONS.items()]
    + [(f"DelNDVI_{s}_city", f"NDVI_urb_all{x}", f"NDVI_rur{x}") for s, x in SEASONS.items()]
    + [("DelDEM", "DEM_urb_CT", "DEM_rur")]
)

# Groups for attribution (sections 5.1-5.2). Credit among the terms of an exact
# identity is arbitrary, so every identity sits inside one group. The tract and
# city SUHI variants share their rural reference, which makes heat separable by
# season and time of day but not into tract versus city, or SUHI versus LST.
# Near-duplicates stay together too: `_CT_act` with `_CT`, and `Area` with
# `log_aland` (r = 0.99 on the log scale).
ATTRIBUTION_GROUPS: dict[str, list[str]] = {
    **{f"Heat: {s} {t}": [f"UHI_{s}_{t}", f"UHI_{s}_{t}_city", f"LST_urb_{t}_CT{x}",
                          f"LST_urb_{t}_CT_act{x}", f"LST_urb_{t}_all{x}", f"LST_rur_{t}{x}"]
       for s, x in SEASONS.items() for t in ("day", "night")},
    **{f"Vegetation: {s}": [f"DelNDVI_{s}", f"DelNDVI_{s}_city", f"NDVI_urb_CT{x}",
                            f"NDVI_urb_CT_act{x}", f"NDVI_urb_all{x}", f"NDVI_rur{x}"]
       for s, x in SEASONS.items()},
    "Elevation": DEM,
    "Tract size": ["Area", "Total_Count", "Urban_Count", "log_aland"],
    "Water share": ["water_frac"],
    "Location": LOCATION,
    "Poverty": POVERTY,
}

CONFIGS: dict[str, list[str]] = {
    "A-full": [c for c in SUHI_FEATURES if c != TARGET_UHI] + GEOMETRY,
    "A-narrow": [c for c in SUHI_FEATURES
                 if c not in (TARGET_UHI, "LST_urb_day_CT", "LST_rur_day")] + GEOMETRY,
    "A-noLST-keepUHI": [c for c in SUHI_FEATURES if c != TARGET_UHI and c not in LST]
    + GEOMETRY,
    "A-noLST": NON_THERMAL,
    "A-noLST+pov": NON_THERMAL + POVERTY,
    "A-lst-only": ["LST_urb_day_CT"],
    "B-env": SUHI_FEATURES + GEOMETRY,
    "B-nonthermal": NON_THERMAL,
    "geo-only": LOCATION,
}

CONFIG_NOTES = {
    "A-full": "All 61 non-target SUHI columns + geometry (reference; contains the target's terms)",
    "A-narrow": "A-full minus LST_urb_day_CT and LST_rur_day",
    "A-noLST-keepUHI": "A-full minus all 24 LST_* columns (UHI_* still present)",
    "A-noLST": "NDVI, dNDVI, DEM, morphology, geometry; no LST_* or UHI_* (primary Task A)",
    "A-noLST+pov": "A-noLST + ACS poverty rate",
    "A-lst-only": "LST_urb_day_CT alone (LST-only baseline)",
    "B-env": "All SUHI columns incl. UHI_annual_day, + geometry (primary Task B)",
    "B-nonthermal": "NDVI, dNDVI, DEM, morphology, geometry (no heat)",
    "geo-only": "Centroid latitude/longitude + Census region (location baseline)",
}

TASKS = {
    "A": {"target": "High_UHI", "primary": "A-noLST", "label": "Task A: High_UHI"},
    "B": {"target": "High_Pov", "primary": "B-env", "label": "Task B: High_Pov"},
}


# Engineered terms offered to the tuned models of section 4.6. Each is (name, kind,
# column a, column b): a difference a − b, a ratio a / (b + 0.001), a square of a,
# or a product a × b. UHI_annual_day is multiplied by the first three numeric
# non-identifier columns of the SUHI file.
INTERACTION_COLUMNS = ["Area", "DEM_rur", "DEM_urb_CT"]
ENGINEERED_B = [
    ("temp_seasonal_diff", "diff", "UHI_summer_day", "UHI_winter_day"),
    ("temp_seasonal_ratio", "ratio", "UHI_summer_day", "UHI_winter_day"),
    ("temp_day_night_diff", "diff", "UHI_annual_day", "UHI_annual_night"),
    ("temp_day_night_ratio", "ratio", "UHI_annual_day", "UHI_annual_night"),
    ("temp_vs_city", "diff", "UHI_annual_day", "UHI_annual_day_city"),
    ("UHI_annual_day_squared", "square", "UHI_annual_day", None),
] + [(f"UHI_x_{c}", "product", "UHI_annual_day", c) for c in INTERACTION_COLUMNS]
# Task A must not see any temperature, so it gets the same terms built on the
# vegetation difference instead of the heat difference. There is no day/night
# vegetation, so the two day-night terms have no analogue.
ENGINEERED_A = [
    ("veg_seasonal_diff", "diff", "DelNDVI_summer", "DelNDVI_winter"),
    ("veg_seasonal_ratio", "ratio", "DelNDVI_summer", "DelNDVI_winter"),
    ("veg_vs_city", "diff", "DelNDVI_annual", "DelNDVI_annual_city"),
    ("DelNDVI_annual_squared", "square", "DelNDVI_annual", None),
] + [(f"DelNDVI_x_{c}", "product", "DelNDVI_annual", c) for c in INTERACTION_COLUMNS]
ENGINEERED = {"A": ENGINEERED_A, "B": ENGINEERED_B}


def engineered_sources(task: str) -> set[str]:
    return {c for _, _, a, b in ENGINEERED[task] for c in (a, b) if c is not None}


def engineer(X: pd.DataFrame, task: str) -> pd.DataFrame:
    """X with the task's engineered terms appended; computed row by row, nothing fitted."""
    out = X.copy()
    for name, kind, a, b in ENGINEERED[task]:
        if kind == "diff":
            v = X[a] - X[b]
        elif kind == "ratio":
            v = X[a] / (X[b] + 0.001)
        elif kind == "square":
            v = X[a] ** 2
        else:
            v = X[a] * X[b]
        # a denominator of exactly -0.001 would give ±inf; leave it to the imputer
        out[name] = v.replace([np.inf, -np.inf], np.nan)
    return out


def config(name: str) -> list[str]:
    return list(CONFIGS[name])


def family_of(column: str) -> str:
    for fam, cols in FAMILIES.items():
        if column in cols:
            return fam
    raise KeyError(column)


def attribution_groups(config_name: str) -> dict[str, list[str]]:
    """The attribution groups present in a configuration, restricted to its columns."""
    cols = set(CONFIGS[config_name])
    out = {g: [c for c in members if c in cols] for g, members in ATTRIBUTION_GROUPS.items()}
    return {g: members for g, members in out.items() if members}


def feature_list_table() -> pd.DataFrame:
    """Long table: one row per (config, feature), with the feature's family."""
    rows = []
    for name, cols in CONFIGS.items():
        for i, c in enumerate(cols, 1):
            rows.append({"config": name, "n_features": len(cols), "order": i, "feature": c,
                         "family": family_of(c), "config_description": CONFIG_NOTES[name]})
    return pd.DataFrame(rows)
