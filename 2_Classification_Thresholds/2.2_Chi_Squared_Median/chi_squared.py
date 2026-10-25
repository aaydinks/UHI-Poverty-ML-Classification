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


def contingency_md(r: dict, n_uhi: dict, n_pov: dict) -> str:
    t = np.array([[r["n00"], r["n01"]], [r["n10"], r["n11"]]], dtype=float)
    e = np.array(r["expected"])
    n = t.sum()
    rows = [
        ("Low SUHI (High_UHI = 0)", 0),
        ("High SUHI (High_UHI = 1)", 1),
    ]
    lines = ["| | Low poverty (High_Pov = 0) | High poverty (High_Pov = 1) | Total |",
             "|---|---:|---:|---:|"]
    for name, i in rows:
        cells = [f"{int(t[i, j]):,}<br>row {t[i, j] / t[i].sum():.1%} · col {t[i, j] / t[:, j].sum():.1%}"
                 f"<br>expected {e[i, j]:,.1f}" for j in (0, 1)]
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {int(t[i].sum()):,} |")
    lines.append(f"| Total | {int(t[:, 0].sum()):,} | {int(t[:, 1].sum()):,} | {int(n):,} |")
    return (
        "# Contingency table: High_UHI × High_Pov (primary sample)\n\n"
        "Both labels are national median splits of the primary analysis sample, identical to the "
        "classifiers' labels.\n\n"
        f"Class counts: High_UHI {n_uhi['n_pos']:,} positive / {n_uhi['n_neg']:,} negative; "
        f"High_Pov {n_pov['n_pos']:,} positive / {n_pov['n_neg']:,} negative.\n\n"
        + "\n".join(lines) + "\n\nrow = share of the row (SUHI class); col = share of the column "
        "(poverty class); expected = count under independence.\n"
    )


def summary_row(name: str, x, y, ua, full: bool = True) -> dict:
    r = association.full(x, y, ua) if full else association.two_by_two(x, y)
    r["analysis"] = name
    return r


def sensitivity(df: pd.DataFrame, cuts: dict) -> pd.DataFrame:
    p = validation.frame("primary", df)
    lpi = validation.frame("lpi", df)
    pr = validation.frame("pr", df)
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
    rows.append(summary_row("Puerto Rico included (same cut values)",
                            pr["High_UHI"].astype(int), pr["High_Pov"].astype(int), pr["ua"]))
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
    p = validation.frame("primary", df)
    cuts = report.read_json(paths.MEDIANS_JSON)["cuts"]
    sens = sensitivity(df, cuts)
    prim = sens.iloc[0].to_dict()
    report.write_text(contingency_md(prim, labels.class_counts(p["High_UHI"]),
                                     labels.class_counts(p["High_Pov"])),
                      OUT / "contingency_table.md")
    sens.drop(columns=["expected"]).to_csv(OUT / "sensitivity.csv", index=False)
    results = {"primary": prim, "cut_values": cuts}
    report.write_json(results, OUT / "chi_squared_results.json")
    report.write_text(results_md(prim, sens), OUT / "chi_squared_results.md")
    plot(sens)
    write_readme()


def results_md(prim: dict, sens: pd.DataFrame) -> str:
    f = report.fmt_num
    srows = "\n".join(
        f"| {r.analysis} | {r.n:,} | {f(r.odds_ratio, 2)} [{f(r.or_ci_low, 2)}, {f(r.or_ci_high, 2)}] | "
        f"{f(r.cmh_or, 2)} [{f(r.cmh_or_ci_low, 2)}, {f(r.cmh_or_ci_high, 2)}] | "
        f"[{f(r.or_boot_ci_low, 2)}, {f(r.or_boot_ci_high, 2)}] | {f(r.cramers_v)} | {report.fmt_p(r.p)} |"
        for r in sens.itertuples())
    return f"""# Chi-squared results

## Primary: High_UHI × High_Pov, national median splits (n = {prim['n']:,})

| Statistic | Value |
|---|---|
| Odds ratio (Woolf 95 % CI) | {f(prim['odds_ratio'], 3)} [{f(prim['or_ci_low'], 3)}, {f(prim['or_ci_high'], 3)}] |
| Odds ratio, UA cluster-bootstrap 95 % CI (B = {prim['B']:,}) | [{f(prim['or_boot_ci_low'], 3)}, {f(prim['or_boot_ci_high'], 3)}] |
| Cochran–Mantel–Haenszel OR by UA (95 % CI) | {f(prim['cmh_or'], 3)} [{f(prim['cmh_or_ci_low'], 3)}, {f(prim['cmh_or_ci_high'], 3)}] ({prim['n_strata_informative']} informative UAs) |
| Breslow–Day test of equal ORs across UAs | χ² = {f(prim['breslow_day_chi2'], 1)}, p {report.fmt_p(prim['breslow_day_p'])} |
| Logistic regression OR, UA-clustered SEs | {f(prim['logit_or'], 3)} [{f(prim['logit_or_ci_low'], 3)}, {f(prim['logit_or_ci_high'], 3)}] ({prim['n_clusters']} clusters) |
| P(high poverty \\| high SUHI) vs P(high poverty \\| low SUHI) | {report.fmt_pct(prim['p_outcome_given_exposed'])} vs {report.fmt_pct(prim['p_outcome_given_unexposed'])} |
| Risk ratio (95 % CI) | {f(prim['risk_ratio'], 3)} [{f(prim['rr_ci_low'], 3)}, {f(prim['rr_ci_high'], 3)}] |
| Risk difference (95 % CI) | {f(prim['risk_difference'], 3)} [{f(prim['rd_ci_low'], 3)}, {f(prim['rd_ci_high'], 3)}] |
| φ = Cramér's V | {f(prim['cramers_v'], 3)} |
| χ² (df = 1), uncorrected | {f(prim['chi2'], 1)}, p {report.fmt_p(prim['p'])} |
| χ² (df = 1), Yates-corrected | {f(prim['chi2_yates'], 1)}, p {report.fmt_p(prim['p_yates'])} |

## Sensitivity

| Analysis | n | OR [Woolf CI] | CMH OR by UA [CI] | OR cluster-bootstrap CI | Cramér's V | χ² p |
|---|---:|---:|---:|---:|---:|---:|
{srows}
"""


def write_readme() -> None:
    r = report.read_json(OUT / "chi_squared_results.json")
    s = pd.read_csv(OUT / "sensitivity.csv")
    prim = r["primary"]
    f = report.fmt_num
    pct = (prim["odds_ratio"] - 1) * 100
    lo_pct, hi_pct = (prim["or_boot_ci_low"] - 1) * 100, (prim["or_boot_ci_high"] - 1) * 100
    s_or = s.set_index("analysis")["odds_ratio"]
    s_lo, s_hi = s_or.drop(s_or.index[0]).min(), s_or.drop(s_or.index[0]).max()
    text = f"""# 2.2 Chi-squared test at the models' median

## Purpose

Tests the association between high SUHI and high poverty at the same cut-offs the classifiers
use, with the full 2×2 table, effect sizes with confidence intervals, and an account of the
spatial dependence between tracts in the same city.

## Method

`chi_squared.py` takes `High_UHI` and `High_Pov` from the built sample, where
`src/uhi_poverty/labels.py` defines them, so the cut-offs are exactly the classifiers'
national medians. It reports:

- class counts, the 2×2 table with row and column percentages and expected counts
  ([`results/contingency_table.md`](results/contingency_table.md));
- χ² (df = 1) with and without Yates' correction, φ / Cramér's V, the odds ratio with a Woolf
  CI, and the risk ratio and risk difference;
- dependence-aware versions: a Cochran–Mantel–Haenszel common OR stratified by UA with the
  Breslow–Day test, a UA cluster-bootstrap CI for the OR (B = 1,000), and a logistic regression
  with UA-clustered standard errors;
- sensitivity analyses: the LPI-based poverty label, a 20 % "poverty area" cut-off, SUHI at the
  66th and 75th percentiles, both outcomes at the 75th percentile, Puerto Rico included, and the
  ten largest UAs excluded.

## Results

**Primary sample** (n = {prim['n']:,} tracts; [`results/chi_squared_results.md`](results/chi_squared_results.md)).
High-SUHI tracts have **{pct:.0f} % higher odds** of being high-poverty tracts than low-SUHI
tracts: OR = {f(prim['odds_ratio'], 2)} (95 % CI {f(prim['or_ci_low'], 2)}–{f(prim['or_ci_high'], 2)};
UA cluster-bootstrap CI {f(prim['or_boot_ci_low'], 2)}–{f(prim['or_boot_ci_high'], 2)}, i.e.
{lo_pct:.0f}–{hi_pct:.0f} % higher odds). In probabilities, {report.fmt_pct(prim['p_outcome_given_exposed'])}
of high-SUHI tracts are high-poverty against {report.fmt_pct(prim['p_outcome_given_unexposed'])}
of low-SUHI tracts (risk difference {f(prim['risk_difference'], 3)}, risk ratio
{f(prim['risk_ratio'], 2)}). Cramér's V = {f(prim['cramers_v'], 3)}; χ²(1) =
{f(prim['chi2'], 1)}, p {report.fmt_p(prim['p'])}.

**Dependence.** Within urbanized areas the association is larger, not smaller: the CMH OR
stratified by UA is {f(prim['cmh_or'], 2)} ({f(prim['cmh_or_ci_low'], 2)}–{f(prim['cmh_or_ci_high'], 2)}).
The Breslow–Day test rejects a common OR across UAs (p {report.fmt_p(prim['breslow_day_p'])}), so
the strength of the association varies from city to city. The logistic regression with
UA-clustered standard errors gives OR {f(prim['logit_or'], 2)}
({f(prim['logit_or_ci_low'], 2)}–{f(prim['logit_or_ci_high'], 2)}).

**Sensitivity.** Across the sensitivity analyses the OR ranges from {f(s_lo, 2)} to
{f(s_hi, 2)} ([`results/sensitivity.csv`](results/sensitivity.csv)); none reverses the direction.

Figure: [`results/conditional_probabilities.png`](results/conditional_probabilities.png).
"""
    report.write_text(text, HERE / "README.md")


if __name__ == "__main__":
    main()
