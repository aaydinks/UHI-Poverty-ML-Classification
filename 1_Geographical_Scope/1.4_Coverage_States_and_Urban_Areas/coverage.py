"""Section 1.4: coverage by state and by urbanized area.

Writes `results/tracts_by_state.csv` (all 51 jurisdictions),
`results/tracts_by_urban_area.csv`, `results/coverage_summary.json`,
`results/coverage_map_tracts.png` and `results/coverage_by_state_map.png`.
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
    plotting.note(fig, "Small states are unlabelled; every count is in tracts_by_state.csv.",
                  y=0.0)
    plotting.savefig(fig, OUT / "coverage_by_state_map.png")


def main() -> None:
    plotting.apply_style()
    df = validation.load_analysis()
    gaz = build.read_gazetteer()
    st = by_state(df, gaz)
    report.write_table(st, OUT / "tracts_by_state")
    ua = by_ua(df)
    ua.to_csv(OUT / "tracts_by_urban_area.csv", index=False)
    shapes = maps.load_tracts()
    states = maps.load_states()
    counts = status_map(df, shapes, states)
    state_map(st, states)
    report.write_json({"tract_status_counts_cb2019": counts,
                       "n_included_uas": int(ua.shape[0]),
                       "n_states": int((st["analysis_tracts"] > 0).sum())},
                      OUT / "coverage_summary.json")


if __name__ == "__main__":
    main()
