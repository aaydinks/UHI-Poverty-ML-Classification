"""Section 1.4: coverage by state and by urbanized area.

Writes `results/tracts_by_state.{csv,md}` (all 51 jurisdictions),
`results/tracts_by_urban_area.csv`, `results/major_urban_areas.md` (top 50),
`results/included_excluded.md`, `results/coverage_map_tracts.png`,
`results/coverage_by_state_map.png` and the README.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch

from uhi_poverty import build, geo, maps, paths, plotting, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
STATUS_COLORS = {"Analysis sample": plotting.BLUE,
                 "Excluded: missing data": plotting.AQUA,
                 "Not urban (no SUHI value)": plotting.NEUTRAL}


def by_state(df: pd.DataFrame, gaz: pd.DataFrame) -> pd.DataFrame:
    gaz = gaz.assign(state_fips=gaz["GEOID"].str[:2])
    p = df[df["in_primary"]]
    t = geo.state_table()
    t["all_2010_tracts"] = t["state_fips"].map(gaz.groupby("state_fips").size())
    t["suhi_tracts"] = t["state_fips"].map(df.groupby("state_fips").size()).fillna(0).astype(int)
    t["analysis_tracts"] = t["state_fips"].map(p.groupby("state_fips").size()).fillna(0).astype(int)
    t["coverage_pct"] = 100 * t["analysis_tracts"] / t["all_2010_tracts"]
    t["pct_high_uhi"] = 100 * t["state_fips"].map(p.groupby("state_fips")["High_UHI"].mean())
    t["pct_high_pov"] = 100 * t["state_fips"].map(p.groupby("state_fips")["High_Pov"].mean())
    t["n_uas"] = t["state_fips"].map(p.groupby("state_fips")["ua"].nunique()).fillna(0).astype(int)
    return t[["state_fips", "state_abbr", "state_name", "region", "analysis_tracts",
              "suhi_tracts", "all_2010_tracts", "coverage_pct", "pct_high_uhi", "pct_high_pov",
              "n_uas"]].sort_values("state_name").reset_index(drop=True)


def by_ua(df: pd.DataFrame) -> pd.DataFrame:
    p = df[df["in_primary"]]
    g = p.groupby("ua")
    out = pd.DataFrame({
        "ua_name": g["ua_name"].first(),
        "primary_state": df.groupby("ua")["ua_primary_state"].first(),
        "n_tracts": g.size(),
        "pct_high_uhi": 100 * g["High_UHI"].mean(),
        "pct_high_pov": 100 * g["High_Pov"].mean(),
        "median_uhi_annual_day": g["UHI_annual_day"].median(),
        "median_pov_rate": g["pov_rate"].median(),
    }).dropna(subset=["n_tracts"])
    out["n_tracts"] = out["n_tracts"].astype(int)
    out["states_spanned"] = geo.ua_states_spanned(p["ua"], p["state_fips"])
    out["primary_state"] = geo.lookup(out["primary_state"], "abbr")
    out["suhi_tracts"] = df.groupby("ua").size()
    out = out.reset_index().sort_values("n_tracts", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def excluded_md(df: pd.DataFrame, st: pd.DataFrame) -> str:
    in_uas = set(df.loc[df["in_primary"], "ua"])
    all_uas = df.groupby("ua").agg(ua_name=("ua_name", "first"), n=("GEOID", "size"))
    dropped = all_uas[~all_uas.index.isin(in_uas)].copy()
    dropped["reason"] = "every tract excluded for missing data"
    partial = df[~df["in_primary"] & df["ua"].isin(in_uas)]
    lines = [
        "# States and urbanized areas included and excluded\n",
        f"**Included:** the study area is the 50 states + DC, and all 51 jurisdictions have "
        f"tracts in the analysis sample; {len(in_uas)} urbanized areas.\n",
        *([f"**Urbanized areas with no tract in the analysis sample ({len(dropped)}):**\n",
           report.md_table(dropped.reset_index()[["ua", "ua_name", "n", "reason"]].rename(
               columns={"n": "SUHI tracts"}))]
          if len(dropped) else
          ["**Urbanized areas:** every urbanized area in the SUHI database has at least one "
           "tract in the analysis sample.\n"]),
        f"\n**Partly excluded:** {len(partial):,} tracts in included UAs were dropped for missing "
        "SUHI fields, renumbering after 2010, or a small ACS universe (section 1.3).\n",
    ]
    return "\n".join(lines)


def status_map(df: pd.DataFrame, shapes, states) -> dict:
    status = pd.Series("Not urban (no SUHI value)", index=shapes.index)
    look = df.set_index("GEOID")
    g = shapes["GEOID"]
    in_suhi = g.isin(look.index)
    prim = g.map(look["in_primary"]).fillna(False).astype(bool)
    status[in_suhi] = "Excluded: missing data"
    status[prim] = "Analysis sample"
    shapes = shapes.assign(status=pd.Categorical(status, categories=list(STATUS_COLORS)))
    fig = plt.figure(figsize=(10, 6.6))
    axes = maps.national_axes(fig)
    maps.draw(axes, shapes, states, {"cmap": ListedColormap(list(STATUS_COLORS.values())),
                                     "categorical": True}, column="status")
    handles = [Patch(facecolor=c, label=f"{k} ({(status == k).sum():,})")
               for k, c in STATUS_COLORS.items()]
    axes["conus"].legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, -0.12),
                         fontsize=8)
    axes["conus"].set_title("Coverage: every 2010 census tract by analysis status", loc="left",
                            fontsize=11)
    plotting.savefig(fig, OUT / "coverage_map_tracts.png")
    return {k: int((status == k).sum()) for k in STATUS_COLORS}


def state_map(st: pd.DataFrame, states) -> None:
    s = states.merge(st[["state_fips", "analysis_tracts"]], left_on="STATEFP",
                     right_on="state_fips", how="left")
    s["state_fips"] = s["STATEFP"]
    fig = plt.figure(figsize=(10, 6.6))
    axes = maps.national_axes(fig)
    norm = Normalize(vmin=0, vmax=float(st["analysis_tracts"].max()))
    for key, ax in axes.items():
        fips = None if key == "conus" else key
        crs = maps.CONUS_CRS if key == "conus" else maps.NON_CONUS[key]
        part = maps._part(s, fips, crs)
        part.plot(ax=ax, column="analysis_tracts", cmap=plotting.sequential_cmap(), norm=norm,
                  edgecolor="white", linewidth=0.6)
        if key == "conus":
            ax.set_xlim(-2.4e6, 2.3e6)
            ax.set_ylim(2.6e5, 3.2e6)
            for _, r in part.iterrows():
                if r.geometry.area < 5e10:  # too small to hold a label; see the table
                    continue
                c = r.geometry.representative_point()
                v = r["analysis_tracts"]
                ink = "white" if v > 0.5 * norm.vmax else plotting.INK
                ax.text(c.x, c.y, f"{r['STUSPS']}\n{int(v):,}", ha="center", va="center",
                        fontsize=5.5, color=ink)
        ax.set_aspect("equal")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plotting.sequential_cmap())
    cax = fig.add_axes([0.56, 0.06, 0.38, 0.025])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Tracts in the analysis sample", color=plotting.INK_2, fontsize=8)
    cb.outline.set_visible(False)
    axes["conus"].set_title("Analysis-sample tracts by state", loc="left", fontsize=11)
    plotting.note(fig, "Small states are unlabelled; every count is in tracts_by_state.md.",
                  y=0.0)
    plotting.savefig(fig, OUT / "coverage_by_state_map.png")


def main() -> None:
    plotting.apply_style()
    df = validation.load_analysis()
    gaz = build.read_gazetteer()
    st = by_state(df, gaz)
    report.write_table(st, OUT / "tracts_by_state", digits=1,
                       title="Tracts by state (all 51 jurisdictions)",
                       note="coverage_pct = analysis tracts / all 2010 tracts. pct_high_* = share "
                            "of the state's analysis tracts in the positive class.")
    ua = by_ua(df)
    ua.to_csv(OUT / "tracts_by_urban_area.csv", index=False)
    top = ua.head(50)[["rank", "ua_name", "primary_state", "states_spanned", "n_tracts",
                       "pct_high_uhi", "pct_high_pov"]]
    report.write_text("# The 50 largest urbanized areas in the analysis sample\n\n"
                      + report.md_table(top, digits=1)
                      + "\nUAs are named after their principal cities. pct_high_* = share of the "
                        "UA's tracts in the positive class.\n", OUT / "major_urban_areas.md")
    report.write_text(excluded_md(df, st), OUT / "included_excluded.md")
    shapes = maps.load_tracts()
    states = maps.load_states()
    counts = status_map(df, shapes, states)
    state_map(st, states)
    report.write_json({"tract_status_counts_cb2019": counts,
                       "n_included_uas": int(ua.shape[0]),
                       "n_states": int((st["analysis_tracts"] > 0).sum())},
                      OUT / "coverage_summary.json")
    write_readme()


def write_readme() -> None:
    st = pd.read_csv(OUT / "tracts_by_state.csv", dtype={"state_fips": str})
    ua = pd.read_csv(OUT / "tracts_by_urban_area.csv", dtype={"ua": str})
    cs = report.read_json(OUT / "coverage_summary.json")
    hi = st.sort_values("coverage_pct", ascending=False).head(3)
    lo = st.sort_values("coverage_pct").head(3)
    fmt = ", ".join
    multi = int(ua["states_spanned"].str.contains(",").sum())
    text = f"""# 1.4 Coverage by state and urbanized area

## Purpose

Shows which states and cities are in the sample, how much of each is covered, and which are left
out.

## Method

`coverage.py` tabulates the analysis sample by state and by urbanized area (UA) and maps the
status of every 2010 tract.

- [`results/tracts_by_state.md`](results/tracts_by_state.md): all 51 jurisdictions, with the
  analysis n, SUHI n, all 2010 tracts, coverage %, and the share of tracts that are
  `High_UHI` and `High_Pov`.
- [`results/tracts_by_urban_area.csv`](results/tracts_by_urban_area.csv): every included UA,
  with its name (principal cities), primary state, every state it spans, n, % `High_UHI` and
  % `High_Pov`. [`results/major_urban_areas.md`](results/major_urban_areas.md) lists the 50
  largest.
- [`results/included_excluded.md`](results/included_excluded.md): jurisdictions and UAs left
  out, and why.
- [`results/coverage_map_tracts.png`](results/coverage_map_tracts.png): every 2010 tract
  coloured by status; [`results/coverage_by_state_map.png`](results/coverage_by_state_map.png):
  analysis tracts per state.

## Results

All {cs['n_states']} jurisdictions (50 states + DC) and {cs['n_included_uas']} urbanized areas
are in the analysis sample. Coverage (analysis tracts / all 2010 tracts) is highest in
{fmt(f"{r.state_abbr} ({r.coverage_pct:.0f} %)" for r in hi.itertuples())} and lowest in
{fmt(f"{r.state_abbr} ({r.coverage_pct:.0f} %)" for r in lo.itertuples())}, reflecting how
urban each state is. The largest UAs are
{fmt(f"{r.ua_name} ({r.n_tracts:,})" for r in ua.head(5).itertuples())}; {multi} UAs span more
than one state.

Tracts on the cartographic map, by status: {'; '.join(f"{k}: {v:,}" for k, v in cs['tract_status_counts_cb2019'].items())}.

**"Cities" means urbanized areas.** The SUHI database's unit is the Census Urbanized Area
(2010), a continuously built-up area named after its principal city or cities (for example
"Minneapolis--St. Paul, MN--WI"). Throughout, "cities" means UAs; leave-one-city-out
(section 4.1) holds out one UA at a time.
"""
    report.write_text(text, HERE / "README.md")


if __name__ == "__main__":
    main()
