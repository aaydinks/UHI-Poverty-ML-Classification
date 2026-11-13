"""Section 2.2: the chi-squared test at the models' own median cut-offs.

Imports `High_UHI` and `High_Pov` from the built sample (defined in
`uhi_poverty.labels`), so the cut-offs are identical to the classifiers'.
Reports class counts, the 2×2 table, χ² with and without Yates' correction,
φ / Cramér's V, OR (Woolf CI), risk ratio and difference; dependence-aware
versions (CMH by UA with Breslow–Day, UA cluster-bootstrap OR, logistic
regression with UA-clustered SEs); and sensitivity analyses.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uhi_poverty import association, labels, paths, plotting, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)


def summary_row(name: str, x, y, ua, full: bool = True) -> dict:
    r = association.full(x, y, ua) if full else association.two_by_two(x, y)
    r["analysis"] = name
    return r


def sensitivity(df: pd.DataFrame, cuts: dict) -> pd.DataFrame:
    p = validation.frame("primary", df)
    lpi = validation.frame("lpi", df)
    rows = [summary_row("Primary: median splits", p["High_UHI"].astype(int),
                        p["High_Pov"].astype(int), p["ua"])]
    rows.append(summary_row("Poverty = HUD LPI below its median (LPI frame)",
                            lpi["High_UHI"].astype(int), lpi["High_Pov_LPI"].astype(int),
                            lpi["ua"]))
    rows.append(summary_row("Poverty = rate >= 20 % ('poverty area' cut-off)",
                            p["High_UHI"].astype(int), (p["pov_rate"] >= 0.20).astype(int),
                            p["ua"]))
    for q in (0.66, 0.75):
        cut = labels.quantile_cut(p["UHI_annual_day"], q)
        rows.append(summary_row(f"High SUHI = {int(q * 100)}th percentile ({cut:.2f} °C)",
                                (p["UHI_annual_day"] >= cut).astype(int),
                                p["High_Pov"].astype(int), p["ua"]))
    q75u = labels.quantile_cut(p["UHI_annual_day"], 0.75)
    q75p = labels.quantile_cut(p["pov_rate"], 0.75)
    rows.append(summary_row("Both at the 75th percentile",
                            (p["UHI_annual_day"] >= q75u).astype(int),
                            (p["pov_rate"] >= q75p).astype(int), p["ua"]))
    top10 = validation.top_uas(p, 10)
    q = p[~p["ua"].isin(top10)]
    rows.append(summary_row("Top-10 UAs by tract count excluded", q["High_UHI"].astype(int),
                            q["High_Pov"].astype(int), q["ua"]))
    return pd.DataFrame(rows)


def plot(sens: pd.DataFrame) -> None:
    plotting.apply_style()
    prim = sens.iloc[0]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    groups = [("National sample, median splits", prim)]
    width = 0.32
    for i, (_, r) in enumerate(groups):
        for j, (key, n_key, label, color) in enumerate((
                ("p_outcome_given_unexposed", ("n00", "n01"), "Low-SUHI tracts", plotting.BLUE),
                ("p_outcome_given_exposed", ("n10", "n11"), "High-SUHI tracts", plotting.ORANGE))):
            pr = r[key]
            n = r[n_key[0]] + r[n_key[1]]
            se = np.sqrt(pr * (1 - pr) / n)
            x = i + (j - 0.5) * width
            ax.bar(x, pr * 100, width=width * 0.9, color=color, label=label if i == 0 else None)
            ax.errorbar(x, pr * 100, yerr=1.96 * se * 100, color=plotting.INK, elinewidth=1,
                        capsize=0)
            ax.text(x, pr * 100 + 3, f"{pr:.1%}", ha="center", fontsize=8, color=plotting.INK)
    ax.set_xticks([0], [f"{name}\nOR {r['odds_ratio']:.2f} · V {r['cramers_v']:.3f}"
                        for name, r in groups])
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(0, 80)
    ax.set_ylabel("Share in the high-poverty class (%)")
    ax.set_title("P(high poverty | SUHI class)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    plotting.note(fig, "High poverty = poverty rate at or above the national median. "
                       "Error bars: 95 % binomial CI.", y=-0.08)
    plotting.savefig(fig, OUT / "conditional_probabilities.png")


def main() -> None:
    df = validation.load_analysis()
    cuts = report.read_json(paths.MEDIANS_JSON)["cuts"]
    sens = sensitivity(df, cuts)
    prim = sens.iloc[0].to_dict()
    sens.drop(columns=["expected"]).to_csv(OUT / "sensitivity.csv", index=False)
    results = {"primary": prim, "cut_values": cuts}
    report.write_json(results, OUT / "chi_squared_results.json")
    plot(sens)


if __name__ == "__main__":
    main()
