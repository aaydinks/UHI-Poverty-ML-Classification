"""Section 1.3: which tracts are used, which are excluded, and why.

Writes `results/exclusion_flow.{csv,md}`, `results/excluded_by_state_and_reason.csv`,
`results/cluster_sizes.{csv,md}`, `results/flow_diagram.png` and the README.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from uhi_poverty import build, geo, paths, plotting, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
REASONS = ["not urban (no SUHI value; source limitation)", "territory (Puerto Rico)",
           "missing SUHI fields", "missing geometry (not in 2019 Gazetteer)",
           "ACS unmatched or poverty universe < 50"]


def by_state(df: pd.DataFrame, gaz: pd.DataFrame) -> pd.DataFrame:
    gaz = gaz.assign(state_fips=gaz["GEOID"].str[:2])
    gaz = gaz[gaz["state_fips"].isin(list(geo.STATES) + [geo.PUERTO_RICO])]
    reason = df["exclusion_reason"].replace({"ACS unmatched": REASONS[4],
                                             "ACS poverty universe < 50": REASONS[4]})
    t = pd.crosstab(df["state_fips"], reason)
    out = pd.DataFrame(index=sorted(gaz["state_fips"].unique()))
    out["all_2010_tracts"] = gaz.groupby("state_fips").size()
    out["suhi_tracts"] = df.groupby("state_fips").size()
    out = out.fillna(0).astype(int)
    not_urban = gaz[~gaz["GEOID"].isin(df["GEOID"])].groupby("state_fips").size()
    out[REASONS[0]] = not_urban.reindex(out.index).fillna(0).astype(int)
    for r in REASONS[1:]:
        out[r] = t[r].reindex(out.index).fillna(0).astype(int) if r in t else 0
    out["included (primary sample)"] = t.get("included", pd.Series(dtype=int)).reindex(
        out.index).fillna(0).astype(int)
    out.insert(0, "state", geo.lookup(pd.Series(out.index, index=out.index), "abbr"))
    out = out.reset_index(names="state_fips")
    total = out.drop(columns=["state_fips", "state"]).sum()
    out.loc[len(out)] = {"state_fips": "", "state": "Total", **total.to_dict()}
    return out


def cluster_sizes(p: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sizes = p.groupby("ua").agg(ua_name=("ua_name", "first"), n_tracts=("GEOID", "size"))
    sizes = sizes.sort_values("n_tracts", ascending=False).reset_index()
    n = sizes["n_tracts"]
    stats = {
        "n_uas": int(len(sizes)), "n_tracts": int(n.sum()), "min": int(n.min()),
        "p25": float(n.quantile(.25)), "median": float(n.median()), "p75": float(n.quantile(.75)),
        "max": int(n.max()), "mean": float(n.mean()),
        "top1_share": float(n.head(1).sum() / n.sum()),
        "top10_share": float(n.head(10).sum() / n.sum()),
        "n_uas_below_30": int((n < 30).sum()),
        "tracts_in_uas_below_30": int(n[n < 30].sum()),
    }
    return sizes, stats


def flow_diagram(flow: pd.DataFrame) -> None:
    plotting.apply_style()
    steps = flow[flow["step"].isin(["0", "1", "3", "3g", "4", "5"])].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.5)
    labels = {"0": "All 2010 tracts, 50 states + DC", "1": "In the SUHI database (50 states + DC)",
              "3": "Complete SUHI target and features", "3g": "With tract geometry",
              "4": "ACS 2015–2019 matched, universe ≥ 50", "5": "Primary analysis sample"}
    side = {"1": "Not urban (no SUHI value)",
            "3": "Missing SUHI fields", "3g": "Not in 2019 Gazetteer",
            "4": "ACS unmatched or\nuniverse < 50 persons"}
    for i, r in steps.iterrows():
        y = (len(steps) - 1 - i) * 1.5 + 0.35
        primary = r["step"] == "5"
        face = plotting.BLUE if primary else plotting.NEUTRAL
        ink = "white" if primary else plotting.INK
        ax.add_patch(FancyBboxPatch((0.3, y), 4.6, 0.8, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=face, edgecolor="none"))
        ax.text(2.6, y + 0.52, labels[r["step"]], ha="center", va="center", fontsize=8.5,
                color=ink)
        ax.text(2.6, y + 0.22, f"n = {int(r['n_tracts']):,}", ha="center", va="center",
                fontsize=9, color=ink, fontweight="bold")
        if i < len(steps) - 1:
            ax.annotate("", xy=(2.6, y - 0.68), xytext=(2.6, y - 0.02),
                        arrowprops={"arrowstyle": "-|>", "color": plotting.MUTED, "lw": 1})
        nxt = steps.iloc[i + 1]["step"] if i < len(steps) - 1 else None
        if nxt in side:
            removed = int(flow.loc[flow["step"] == nxt, "n_removed"].iloc[0])
            if nxt == "1":
                pr = int(flow.loc[flow["step"] == "2", "n_removed"].iloc[0])
                text = (f"{side[nxt]}\n− {removed:,}\n(Puerto Rico's {pr:,} SUHI tracts are "
                        "the sensitivity frame)")
            else:
                text = f"{side[nxt]}\n− {removed:,}"
            ax.annotate("", xy=(5.6, y - 0.35), xytext=(2.6, y - 0.35),
                        arrowprops={"arrowstyle": "-|>", "color": plotting.MUTED, "lw": 1})
            ax.text(5.75, y - 0.35, text, ha="left", va="center", fontsize=8,
                    color=plotting.INK_2)
    ax.set_title("Tract accounting: from all 2010 tracts to the analysis sample", fontsize=10)
    plotting.savefig(fig, OUT / "flow_diagram.png")


def main() -> None:
    df = validation.load_analysis()
    gaz = build.read_gazetteer()
    flow = pd.read_csv(paths.EXCLUSION_FLOW, dtype={"step": str})
    report.write_table(flow, OUT / "exclusion_flow", title="Exclusion flow",
                       note="n_removed = tracts dropped between the previous step and this one.")
    st = by_state(df, gaz)
    st.to_csv(OUT / "excluded_by_state_and_reason.csv", index=False)
    sizes, stats = cluster_sizes(df[df["in_primary"]])
    sizes.to_csv(OUT / "cluster_sizes.csv", index=False)
    report.write_json(stats, OUT / "cluster_size_summary.json")
    small = df[df["exclusion_reason"] == "ACS poverty universe < 50"]
    report.write_json({
        "n_universe_below_50": int(len(small)),
        "n_acs_unmatched_not_otherwise_excluded": int(
            (df["exclusion_reason"] == "ACS unmatched").sum()),
        "median_population_universe_below_50": float(small["population"].median()),
        "median_poverty_universe_below_50": float(small["pov_universe"].median()),
        "share_population_below_100": float((small["population"] < 100).mean()),
    }, OUT / "acs_exclusion_detail.json")
    flow_diagram(flow)
    write_readme()


def write_readme() -> None:
    flow = pd.read_csv(OUT / "exclusion_flow.csv", dtype={"step": str}).set_index("step")
    st = pd.read_csv(OUT / "excluded_by_state_and_reason.csv")
    cs = report.read_json(OUT / "cluster_size_summary.json")
    ax = report.read_json(OUT / "acs_exclusion_detail.json")
    tot = st[st["state"] == "Total"].iloc[0]
    rows = "\n".join(
        f"| {k} | {r.rule} | {int(r.n_tracts):,} | "
        f"{'' if np.isnan(r.n_removed) else f'{int(r.n_removed):,}'} |"
        for k, r in flow.iterrows())
    most_acs = st[st.state != "Total"].sort_values(REASONS[4], ascending=False).head(3)
    acs_states = ", ".join(f"{r.state} {int(r[REASONS[4]])}" for _, r in most_acs.iterrows())
    text = f"""# 1.3 Tracts used and excluded

## Purpose

Accounts for every tract: which were analysed, which were excluded and why, and how regional
differences and the clustering of tracts in cities are handled.

## Method

`tract_accounting.py` reports the sample-construction steps built once by `make build`
(`src/uhi_poverty/build.py`), breaks the exclusions down by state and reason, and describes how
tracts cluster in urbanized areas. Every merge is one-to-one on 2010 tract codes, and the counts
reconcile (the build fails otherwise).

## Results

[`results/exclusion_flow.md`](results/exclusion_flow.md), drawn in
[`results/flow_diagram.png`](results/flow_diagram.png):

| Step | Rule | Tracts | Removed |
|---|---|---:|---:|
{rows}

By reason, over the 50 states + DC and Puerto Rico
([`results/excluded_by_state_and_reason.csv`](results/excluded_by_state_and_reason.csv)):

| Reason | Tracts |
|---|---:|
| Not urban: no SUHI value (source limitation) | {int(tot[REASONS[0]]):,} |
| Territory (Puerto Rico; kept as the sensitivity frame) | {int(tot[REASONS[1]]):,} |
| Missing SUHI fields | {int(tot[REASONS[2]]):,} |
| Not in the 2019 Gazetteer (tract renumbered after 2010) | {int(tot[REASONS[3]]):,} |
| ACS unmatched or poverty universe < 50 persons | {int(tot[REASONS[4]]):,} |
| **Included (primary sample)** | **{int(tot['included (primary sample)']):,}** |

Of the ACS exclusions, {ax['n_universe_below_50']} tracts have a poverty universe below 50
persons (median total population {ax['median_population_universe_below_50']:.0f};
{report.fmt_pct(ax['share_population_below_100'])} have fewer than 100 residents);
{ax['n_acs_unmatched_not_otherwise_excluded']} more have no ACS row (the SUHI tracts without one
were already removed as renumbered after 2010). The states with the most ACS exclusions are
{acs_states}.

**Clustering** ([`results/cluster_sizes.csv`](results/cluster_sizes.csv)). The
{cs['n_tracts']:,} tracts are nested in {cs['n_uas']} urbanized areas. The median UA has
{cs['median']:.0f} tracts (interquartile range {cs['p25']:.0f}–{cs['p75']:.0f}; largest
{cs['max']:,}). The largest UA holds {report.fmt_pct(cs['top1_share'])} of all tracts and the ten
largest hold {report.fmt_pct(cs['top10_share'])}. {cs['n_uas_below_30']} UAs have fewer than 30
tracts ({cs['tracts_in_uas_below_30']:,} tracts in all).

## How limitations, regions and clustering are handled

- **Limitations.** Rural tracts have no SUHI value by construction: the database only covers
  tracts inside Census urbanized areas, so results describe urban tracts.
- **Regional differences.** No tract is dropped because of its region. Instead, regional
  transfer is tested by holding out whole Census regions and states (section 4.1).
- **Clustering.** Tracts are nested in {cs['n_uas']} UAs. This is handled by UA-grouped
  validation (leave-one-city-out is the headline) and by confidence intervals from a bootstrap
  that resamples UAs (sections 2.2 and 4.4). UAs with fewer than 30 tracts, or with only one
  class, stay in every pooled metric but get no per-city metric.
"""
    report.write_text(text, HERE / "README.md")


if __name__ == "__main__":
    main()
