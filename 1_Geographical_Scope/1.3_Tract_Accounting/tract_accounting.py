"""Section 1.3: which tracts are used, which are excluded, and why.

Writes `results/exclusion_flow.csv`, `results/excluded_by_state_and_reason.csv`,
`results/cluster_sizes.csv` and `results/flow_diagram.png`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from uhi_poverty import build, geo, paths, plotting, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
REASONS = ["not urban (no SUHI value; source limitation)", "missing SUHI fields", "missing geometry (not in 2019 Gazetteer)",
           "ACS unmatched or poverty universe < 50"]


def by_state(df: pd.DataFrame, gaz: pd.DataFrame) -> pd.DataFrame:
    gaz = gaz.assign(state_fips=gaz["GEOID"].str[:2])
    reason = df["exclusion_reason"].replace({"ACS unmatched": REASONS[3],
                                             "ACS poverty universe < 50": REASONS[3]})
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
    steps = flow[flow["step"].isin(["0", "1", "2", "2g", "3", "4"])].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.5)
    labels = {"0": "All 2010 tracts, 50 states + DC", "1": "In the SUHI database (50 states + DC)",
              "2": "Complete SUHI target and features", "2g": "With tract geometry",
              "3": "ACS 2015–2019 matched, universe ≥ 50", "4": "Primary analysis sample"}
    side = {"1": "Not urban (no SUHI value)",
            "2": "Missing SUHI fields", "2g": "Not in 2019 Gazetteer",
            "3": "ACS unmatched or\nuniverse < 50 persons"}
    for i, r in steps.iterrows():
        y = (len(steps) - 1 - i) * 1.5 + 0.35
        primary = r["step"] == "4"
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
    report.write_table(flow, OUT / "exclusion_flow")
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


if __name__ == "__main__":
    main()
