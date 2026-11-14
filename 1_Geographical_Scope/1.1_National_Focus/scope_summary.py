"""Section 1.1: the study's geographic scope, in numbers.

Writes `results/scope_summary.json` (tracts, UAs, states and the primary sample n).
"""

from __future__ import annotations

from pathlib import Path

from uhi_poverty import build, paths, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)


def main() -> None:
    df = validation.load_analysis()
    gaz = build.read_gazetteer()
    p = df[df["in_primary"]]
    summary = {
        "all_2010_tracts_50_states_dc": int(len(gaz)),
        "suhi_tracts_50_states_dc": int(len(df)),
        "suhi_uas_total": int(df["ua"].nunique()),
        "jurisdictions_with_suhi_tracts": int(df["state_fips"].nunique()),
        "primary_n": int(len(p)),
        "primary_uas": int(p["ua"].nunique()),
        "primary_states": int(p["state_fips"].nunique()),
        "primary_share_of_all_2010_tracts": float(len(p) / len(gaz)),
        "lpi_frame_n": int(df["in_lpi_frame"].sum()),
        "primary_by_region": p["region"].value_counts().to_dict(),
    }
    report.write_json(summary, OUT / "scope_summary.json")


if __name__ == "__main__":
    main()
