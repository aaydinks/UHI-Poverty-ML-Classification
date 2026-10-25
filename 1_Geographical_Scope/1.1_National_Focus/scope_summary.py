"""Section 1.1: the study's geographic scope, in numbers.

Writes `results/scope_summary.json` (tracts, UAs, states, Puerto Rico counts and
the primary sample n) and the README with the study-area description.
"""

from __future__ import annotations

from pathlib import Path

from uhi_poverty import build, geo, paths, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)


def main() -> None:
    df = validation.load_analysis()
    gaz = build.read_gazetteer()
    p = df[df["in_primary"]]
    pr = df[df["state_fips"] == geo.PUERTO_RICO]
    gaz_states = gaz[~geo.is_territory(gaz["GEOID"].str[:2])]
    summary = {
        "all_2010_tracts_50_states_dc": int(len(gaz_states)),
        "all_2010_tracts_puerto_rico": int((gaz["GEOID"].str[:2] == geo.PUERTO_RICO).sum()),
        "suhi_tracts_total": int(len(df)),
        "suhi_tracts_50_states_dc": int((~df["territory"]).sum()),
        "suhi_tracts_puerto_rico": int(len(pr)),
        "suhi_uas_total": int(df["ua"].nunique()),
        "suhi_uas_puerto_rico_only": int(df.loc[df["territory"], "ua"].nunique()),
        "jurisdictions_with_suhi_tracts": int(df.loc[~df["territory"], "state_fips"].nunique()),
        "primary_n": int(len(p)),
        "primary_uas": int(p["ua"].nunique()),
        "primary_states": int(p["state_fips"].nunique()),
        "primary_share_of_all_2010_tracts": float(len(p) / len(gaz_states)),
        "lpi_frame_n": int(df["in_lpi_frame"].sum()),
        "pr_frame_n": int(df["in_pr_frame"].sum()),
        "pr_frame_puerto_rico_tracts": int((df["in_pr_frame"] & df["territory"]).sum()),
        "primary_by_region": p["region"].value_counts().to_dict(),
    }
    report.write_json(summary, OUT / "scope_summary.json")
    write_readme()


def write_readme() -> None:
    s = report.read_json(OUT / "scope_summary.json")
    reg = s["primary_by_region"]
    regions = ", ".join(f"{r} {reg[r]:,}" for r in geo.REGIONS)
    text = f"""# 1.1 National focus

## Purpose

States the study area: which tracts, urbanized areas and states the national analysis covers.

## Method

Every analysis uses one national sample (`make build`), and `scope_summary.py` records its
extent in [`results/scope_summary.json`](results/scope_summary.json). Every join uses 2010
tract codes.

## Results

| | Count |
|---|---:|
| All 2010 census tracts, 50 states + DC | {s['all_2010_tracts_50_states_dc']:,} |
| Tracts in the SUHI database, 50 states + DC | {s['suhi_tracts_50_states_dc']:,} |
| Tracts in the SUHI database, Puerto Rico | {s['suhi_tracts_puerto_rico']:,} |
| Urbanized areas (UAs) in the SUHI database | {s['suhi_uas_total']:,} (of which {s['suhi_uas_puerto_rico_only']} in Puerto Rico) |
| **Primary analysis sample** | **{s['primary_n']:,} tracts in {s['primary_uas']} UAs across {s['primary_states']} jurisdictions (50 states + DC)** |
| Share of all 2010 tracts in the 50 states + DC | {report.fmt_pct(s['primary_share_of_all_2010_tracts'])} |
| By Census region | {regions} |
| Robustness frame with HUD LPI | {s['lpi_frame_n']:,} |
| Sensitivity frame with Puerto Rico | {s['pr_frame_n']:,} ({s['pr_frame_puerto_rico_tracts']:,} Puerto Rico tracts) |
"""
    report.write_text(text, HERE / "README.md")


if __name__ == "__main__":
    main()
