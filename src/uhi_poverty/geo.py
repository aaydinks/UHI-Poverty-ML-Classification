"""Tract identifiers, the state/region table, and urbanized-area (UA) helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

# FIPS -> (abbreviation, name, Census region, Census division)
STATES: dict[str, tuple[str, str, str, str]] = {
    "01": ("AL", "Alabama", "South", "East South Central"),
    "02": ("AK", "Alaska", "West", "Pacific"),
    "04": ("AZ", "Arizona", "West", "Mountain"),
    "05": ("AR", "Arkansas", "South", "West South Central"),
    "06": ("CA", "California", "West", "Pacific"),
    "08": ("CO", "Colorado", "West", "Mountain"),
    "09": ("CT", "Connecticut", "Northeast", "New England"),
    "10": ("DE", "Delaware", "South", "South Atlantic"),
    "11": ("DC", "District of Columbia", "South", "South Atlantic"),
    "12": ("FL", "Florida", "South", "South Atlantic"),
    "13": ("GA", "Georgia", "South", "South Atlantic"),
    "15": ("HI", "Hawaii", "West", "Pacific"),
    "16": ("ID", "Idaho", "West", "Mountain"),
    "17": ("IL", "Illinois", "Midwest", "East North Central"),
    "18": ("IN", "Indiana", "Midwest", "East North Central"),
    "19": ("IA", "Iowa", "Midwest", "West North Central"),
    "20": ("KS", "Kansas", "Midwest", "West North Central"),
    "21": ("KY", "Kentucky", "South", "East South Central"),
    "22": ("LA", "Louisiana", "South", "West South Central"),
    "23": ("ME", "Maine", "Northeast", "New England"),
    "24": ("MD", "Maryland", "South", "South Atlantic"),
    "25": ("MA", "Massachusetts", "Northeast", "New England"),
    "26": ("MI", "Michigan", "Midwest", "East North Central"),
    "27": ("MN", "Minnesota", "Midwest", "West North Central"),
    "28": ("MS", "Mississippi", "South", "East South Central"),
    "29": ("MO", "Missouri", "Midwest", "West North Central"),
    "30": ("MT", "Montana", "West", "Mountain"),
    "31": ("NE", "Nebraska", "Midwest", "West North Central"),
    "32": ("NV", "Nevada", "West", "Mountain"),
    "33": ("NH", "New Hampshire", "Northeast", "New England"),
    "34": ("NJ", "New Jersey", "Northeast", "Middle Atlantic"),
    "35": ("NM", "New Mexico", "West", "Mountain"),
    "36": ("NY", "New York", "Northeast", "Middle Atlantic"),
    "37": ("NC", "North Carolina", "South", "South Atlantic"),
    "38": ("ND", "North Dakota", "Midwest", "West North Central"),
    "39": ("OH", "Ohio", "Midwest", "East North Central"),
    "40": ("OK", "Oklahoma", "South", "West South Central"),
    "41": ("OR", "Oregon", "West", "Pacific"),
    "42": ("PA", "Pennsylvania", "Northeast", "Middle Atlantic"),
    "44": ("RI", "Rhode Island", "Northeast", "New England"),
    "45": ("SC", "South Carolina", "South", "South Atlantic"),
    "46": ("SD", "South Dakota", "Midwest", "West North Central"),
    "47": ("TN", "Tennessee", "South", "East South Central"),
    "48": ("TX", "Texas", "South", "West South Central"),
    "49": ("UT", "Utah", "West", "Mountain"),
    "50": ("VT", "Vermont", "Northeast", "New England"),
    "51": ("VA", "Virginia", "South", "South Atlantic"),
    "53": ("WA", "Washington", "West", "Pacific"),
    "54": ("WV", "West Virginia", "South", "South Atlantic"),
    "55": ("WI", "Wisconsin", "Midwest", "East North Central"),
    "56": ("WY", "Wyoming", "West", "Mountain"),
}

# Territories: (abbreviation, name). Puerto Rico is the only one in the SUHI database.
TERRITORIES: dict[str, tuple[str, str]] = {
    "60": ("AS", "American Samoa"),
    "66": ("GU", "Guam"),
    "69": ("MP", "Northern Mariana Islands"),
    "72": ("PR", "Puerto Rico"),
    "78": ("VI", "U.S. Virgin Islands"),
}
PUERTO_RICO = "72"
REGIONS = ("Northeast", "Midwest", "South", "West")


def normalize_geoid(values, width: int = 11) -> pd.Series:
    """Zero-padded string GEOIDs from ints, floats or strings.

    The SUHI file stores `Census_geoid` as an integer, so every state with a
    FIPS code below 10 loses its leading zero unless it is padded back.
    """
    s = pd.Series(values)
    if pd.api.types.is_float_dtype(s):
        if s.isna().any():
            raise ValueError("GEOID column has missing values")
        s = s.astype("int64")
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    if not s.str.fullmatch(r"\d+").all():
        raise ValueError("GEOIDs must be digits only")
    if (s.str.len() > width).any():
        raise ValueError(f"GEOID longer than {width} characters")
    return s.str.zfill(width)


def state_of(geoids: pd.Series) -> pd.Series:
    return geoids.str[:2]


def county_of(geoids: pd.Series) -> pd.Series:
    return geoids.str[:5]


def is_territory(fips) -> np.ndarray:
    return pd.Series(fips).isin(list(TERRITORIES)).to_numpy()


def state_table(include_territories: bool = False) -> pd.DataFrame:
    """One row per jurisdiction: fips, abbr, name, region, division, territory flag."""
    rows = [
        {"state_fips": f, "state_abbr": a, "state_name": n, "region": r, "division": d,
         "territory": False}
        for f, (a, n, r, d) in STATES.items()
    ]
    if include_territories:
        rows += [
            {"state_fips": f, "state_abbr": a, "state_name": n, "region": "Territory",
             "division": "Territory", "territory": True}
            for f, (a, n) in TERRITORIES.items()
        ]
    return pd.DataFrame(rows)


def lookup(fips: pd.Series, field: str) -> pd.Series:
    """Map state FIPS to `abbr`, `name`, `region` or `division`; territories included."""
    table = state_table(include_territories=True).set_index("state_fips")
    column = {"abbr": "state_abbr", "name": "state_name"}.get(field, field)
    return pd.Series(fips).map(table[column])


def ua_primary_state(ua: pd.Series, state_fips: pd.Series) -> pd.Series:
    """The state holding most of each UA's tracts (ties go to the lower FIPS code)."""
    counts = (
        pd.DataFrame({"ua": np.asarray(ua), "state": np.asarray(state_fips)})
        .value_counts()
        .rename("n")
        .reset_index()
        .sort_values(["ua", "n", "state"], ascending=[True, False, True])
    )
    return counts.drop_duplicates("ua").set_index("ua")["state"]


def ua_states_spanned(ua: pd.Series, state_fips: pd.Series) -> pd.Series:
    """Sorted, comma-joined abbreviations of every state a UA has tracts in."""
    abbr = lookup(pd.Series(np.asarray(state_fips)), "abbr")
    frame = pd.DataFrame({"ua": np.asarray(ua), "abbr": abbr.to_numpy()})
    return frame.groupby("ua")["abbr"].agg(lambda s: ",".join(sorted(set(s))))
