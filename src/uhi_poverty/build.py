"""Build the analysis sample: `make build`.

Writes
- `data/processed/analysis_tracts.parquet`: every SUHI tract, with geometry,
  ACS and LPI columns, frame flags (`in_primary`, `in_lpi_frame`, `in_pr_frame`),
  an exclusion reason, and the shared labels;
- `data/processed/exclusion_flow.csv`: the step table with counts;
- `data/processed/median_values.json`: the national cut values and class counts;
- `data/processed/acs_unmatched.csv`: SUHI tracts with no ACS 2015–2019 row.

Every merge is validated one-to-one and every step logs its row count.
"""

from __future__ import annotations

import json
import logging
import zipfile

import numpy as np
import pandas as pd

from . import features, geo, labels, paths, report

log = logging.getLogger("uhi_poverty.build")

ACS_UNIVERSE_MIN = 50
ACS_NO_SAMPLING_ERROR = -555555555


# --------------------------------------------------------------------------- readers


def read_suhi() -> pd.DataFrame:
    df = pd.read_csv(paths.SUHI_CSV, low_memory=False)
    df["GEOID"] = geo.normalize_geoid(df["Census_geoid"])
    df["ua"] = geo.normalize_geoid(df["Urban_geoid"], width=5)
    df["ua_name"] = df["Urban_name"]
    if df["GEOID"].duplicated().any():
        raise ValueError("duplicate GEOIDs in the SUHI file")
    return df


def read_gazetteer(path=None) -> pd.DataFrame:
    """2019 Gazetteer (2010 tract codes): centroids, land and water area."""
    path = path or paths.GAZ_2019
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".txt"))
        with z.open(name) as f:
            gaz = pd.read_csv(f, sep="\t", dtype={"GEOID": str}, encoding="latin-1")
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz["GEOID"] = geo.normalize_geoid(gaz["GEOID"])
    gaz = gaz.rename(columns={"INTPTLAT": "lat", "INTPTLONG": "lon"})
    return gaz[["GEOID", "USPS", "ALAND", "AWATER", "lat", "lon"]]


def read_acs() -> pd.DataFrame:
    """ACS 2015–2019 B17001/B01003 for every cached state, sentinels set to NaN."""
    frames = []
    for path in sorted(paths.ACS_DIR.glob("acs5_2019_B17001_*.json")):
        rows = json.loads(path.read_text())
        frames.append(pd.DataFrame(rows[1:], columns=rows[0]))
    acs = pd.concat(frames, ignore_index=True)
    acs["GEOID"] = acs["state"] + acs["county"] + acs["tract"]
    for col in ("B17001_001E", "B17001_002E", "B01003_001E", "B17001_001M", "B17001_002M"):
        v = pd.to_numeric(acs[col], errors="coerce")
        if col.endswith("M"):
            v = v.mask(v == ACS_NO_SAMPLING_ERROR, 0)
        acs[col] = v.mask(v < 0)
    acs = acs.rename(columns={"B17001_001E": "pov_universe", "B17001_001M": "pov_universe_moe",
                              "B17001_002E": "pov_below", "B17001_002M": "pov_below_moe",
                              "B01003_001E": "population"})
    acs["pov_rate"] = acs["pov_below"] / acs["pov_universe"].where(acs["pov_universe"] > 0)
    if acs["GEOID"].duplicated().any():
        raise ValueError("duplicate GEOIDs in the ACS responses")
    return acs[["GEOID", "pov_universe", "pov_universe_moe", "pov_below", "pov_below_moe",
                "population", "pov_rate"]]


def read_lpi() -> pd.DataFrame:
    lpi = pd.read_csv(paths.LPI_CSV, dtype={"GEOID": str}, encoding="utf-8-sig")
    lpi["GEOID"] = geo.normalize_geoid(lpi["GEOID"])
    if lpi["GEOID"].duplicated().any():
        raise ValueError("duplicate GEOIDs in the LPI file")
    lpi["LPI"] = pd.to_numeric(lpi["POV_IDX"], errors="coerce")
    lpi.loc[(lpi["LPI"] < 0) | (lpi["LPI"] > 99), "LPI"] = np.nan
    return lpi[["GEOID", "LPI"]]


# --------------------------------------------------------------------------- build


def _merge(left, right, how="left", **kw):
    out = left.merge(right, on="GEOID", how=how, validate="one_to_one", **kw)
    log.info("merge %s: %d -> %d rows", how, len(left), len(out))
    return out


def build() -> pd.DataFrame:
    gaz = read_gazetteer()
    suhi = read_suhi()
    acs = read_acs()
    lpi = read_lpi()
    log.info("Gazetteer %d, SUHI %d, ACS %d, LPI %d rows", len(gaz), len(suhi), len(acs),
             len(lpi))

    df = _merge(suhi, gaz.drop(columns="USPS"), indicator="_gaz")
    df["has_geometry"] = df.pop("_gaz").eq("both")
    df = _merge(df, acs, indicator="_acs")
    df["acs_matched"] = df.pop("_acs").eq("both")
    df = _merge(df, lpi)

    df["state_fips"] = geo.state_of(df["GEOID"])
    df["county_fips"] = geo.county_of(df["GEOID"])
    df["state_abbr"] = geo.lookup(df["state_fips"], "abbr")
    df["region"] = geo.lookup(df["state_fips"], "region")
    df["division"] = geo.lookup(df["state_fips"], "division")
    if df["state_abbr"].isna().any():
        raise ValueError("unknown state FIPS in the SUHI file")
    df["territory"] = geo.is_territory(df["state_fips"])
    df["ua_primary_state"] = df["ua"].map(geo.ua_primary_state(df["ua"], df["state_fips"]))
    for r in geo.REGIONS:
        df[f"region_{r}"] = (df["region"] == r).astype(float)

    area = df["ALAND"] + df["AWATER"]
    df["water_frac"] = df["AWATER"] / area.where(area > 0)
    df["log_aland"] = np.log1p(df["ALAND"])

    complete_suhi = df[features.SUHI_FEATURES].notna().all(axis=1)
    complete_geom = df["has_geometry"] & df[features.GEOMETRY].notna().all(axis=1)
    acs_ok = df["acs_matched"] & (df["pov_universe"] >= ACS_UNIVERSE_MIN) & df["pov_rate"].notna()

    reason = pd.Series("included", index=df.index)
    reason[~acs_ok & df["acs_matched"]] = "ACS poverty universe < 50"
    reason[~df["acs_matched"]] = "ACS unmatched"
    reason[~complete_geom] = "missing geometry (not in 2019 Gazetteer)"
    reason[~complete_suhi] = "missing SUHI fields"
    reason[df["territory"]] = "territory (Puerto Rico)"
    df["exclusion_reason"] = reason

    usable = complete_suhi & complete_geom & acs_ok
    df["in_primary"] = usable & ~df["territory"]
    df["in_lpi_frame"] = df["in_primary"] & df["LPI"].notna()
    df["in_pr_frame"] = usable & (df["in_primary"] | (df["state_fips"] == geo.PUERTO_RICO))

    cuts = labels.medians(df, df["in_primary"], df["in_lpi_frame"])
    df = labels.add_labels(df, cuts)
    checks = labels.assert_orientation(df[df["in_primary"]])

    keep = (["GEOID", "ua", "ua_name", "Urban_name_extended", "state_fips", "state_abbr",
             "county_fips", "region", "division", "territory", "ua_primary_state", "lat", "lon",
             "ALAND", "AWATER", "has_geometry", "acs_matched"]
            + features.SUHI_FEATURES + features.GEOMETRY
            + [f"region_{r}" for r in geo.REGIONS]
            + ["pov_universe", "pov_universe_moe", "pov_below", "pov_below_moe", "population",
               "pov_rate", "LPI", "exclusion_reason", "in_primary", "in_lpi_frame",
               "in_pr_frame", "High_UHI", "High_Pov", "High_Pov_LPI"])
    df = df[keep].sort_values("GEOID").reset_index(drop=True)

    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.ANALYSIS_PARQUET, index=False)
    flow = exclusion_flow(df, gaz)
    flow.to_csv(paths.EXCLUSION_FLOW, index=False)
    unmatched = df.loc[~df["acs_matched"] & ~df["territory"],
                       ["GEOID", "ua_name", "state_abbr", "county_fips"]]
    unmatched.to_csv(paths.PROCESSED / "acs_unmatched.csv", index=False)
    report.write_json(median_summary(df, cuts, checks), paths.MEDIANS_JSON)
    log.info("primary sample: %d tracts in %d UAs", int(df["in_primary"].sum()),
             df.loc[df["in_primary"], "ua"].nunique())
    return df


def exclusion_flow(df: pd.DataFrame, gaz: pd.DataFrame) -> pd.DataFrame:
    states = gaz["GEOID"].str[:2]
    gaz_states = gaz[~states.isin(list(geo.TERRITORIES))]
    n_gaz_states = len(gaz_states)
    n_gaz_pr = int((states == geo.PUERTO_RICO).sum())
    n_not_urban = int((~gaz_states["GEOID"].isin(df["GEOID"])).sum())
    terr = df["territory"]
    n_suhi_not_in_gaz = int((~terr & ~df["has_geometry"]).sum())
    s3 = ~terr & ~df["exclusion_reason"].eq("missing SUHI fields")
    s3g = s3 & ~df["exclusion_reason"].str.startswith("missing geometry")
    s4 = df["in_primary"]
    rows = [
        ("0", "All 2010 tracts, 50 states + DC (2019 Gazetteer; coverage universe)",
         n_gaz_states, None, f"Puerto Rico adds {n_gaz_pr:,} tracts"),
        ("1", "In the SUHI database, 50 states + DC (urbanized-area tracts only)",
         int((~terr).sum()), n_not_urban,
         f"Gazetteer tracts with no SUHI value: not urban (source limitation); "
         f"{n_suhi_not_in_gaz} SUHI codes are absent from the 2019 Gazetteer"),
        ("2", "50 states + DC (Puerto Rico held out as the sensitivity frame)",
         int((~terr).sum()), int(terr.sum()),
         f"{int(terr.sum()):,} Puerto Rico SUHI tracts moved to the sensitivity frame"),
        ("3", "Complete SUHI target and features",
         int(s3.sum()), int((~terr).sum() - s3.sum()), "missing SUHI fields"),
        ("3g", "…with Gazetteer geometry (water_frac, log_aland)",
         int(s3g.sum()), int(s3.sum() - s3g.sum()), "not in the 2019 Gazetteer"),
        ("4", f"Matched to ACS 2015–2019 with poverty universe >= {ACS_UNIVERSE_MIN}",
         int(s4.sum()), int(s3g.sum() - s4.sum()), "ACS unmatched or small universe"),
        ("5", "Primary analysis sample", int(s4.sum()), None, ""),
        ("5a", "…with HUD LPI present (robustness frame)", int(df["in_lpi_frame"].sum()),
         int(s4.sum() - df["in_lpi_frame"].sum()), "no LPI value"),
        ("5b", "…plus Puerto Rico (sensitivity frame)", int(df["in_pr_frame"].sum()), None,
         f"+{int(df['in_pr_frame'].sum() - s4.sum()):,} Puerto Rico tracts"),
    ]
    flow = pd.DataFrame(rows, columns=["step", "rule", "n_tracts", "n_removed", "note"])
    # reconcile: removals between steps 1 and 5 must account for every excluded tract
    removed = flow.loc[flow["step"].isin(["3", "3g", "4"]), "n_removed"].sum()
    if int((~terr).sum()) - removed != int(s4.sum()):
        raise AssertionError("exclusion flow does not reconcile")
    return flow


def median_summary(df: pd.DataFrame, cuts: dict, checks: dict) -> dict:
    p = df[df["in_primary"]]
    lpi = df[df["in_lpi_frame"]]
    pr = df[df["in_pr_frame"]]
    return {
        "cuts": {
            "High_UHI": {"column": "UHI_annual_day", "rule": ">= median", "value": cuts[
                "UHI_annual_day"], "unit": "degC", "frame": "primary"},
            "High_Pov": {"column": "pov_rate", "rule": ">= median", "value": cuts["pov_rate"],
                         "unit": "proportion", "frame": "primary"},
            "High_Pov_LPI": {"column": "LPI", "rule": "< median (low index = poorer)",
                             "value": cuts["LPI"], "unit": "index 0-99", "frame": "lpi (5a)"},
        },
        "percentiles_primary": {
            "UHI_annual_day": {q: float(p["UHI_annual_day"].quantile(v))
                               for q, v in (("p50", .5), ("p66", .66), ("p75", .75))},
            "pov_rate": {q: float(p["pov_rate"].quantile(v))
                         for q, v in (("p50", .5), ("p66", .66), ("p75", .75))},
        },
        "class_counts": {
            "primary": {k: labels.class_counts(p[k]) for k in ("High_UHI", "High_Pov")},
            "lpi_frame": {k: labels.class_counts(lpi[k])
                          for k in ("High_UHI", "High_Pov", "High_Pov_LPI")},
            "pr_frame": {k: labels.class_counts(pr[k]) for k in ("High_UHI", "High_Pov")},
        },
        "orientation_checks": checks,
        "n": {"primary": int(len(p)), "lpi_frame": int(len(lpi)), "pr_frame": int(len(pr)),
              "primary_uas": int(p["ua"].nunique())},
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build()


if __name__ == "__main__":
    main()
