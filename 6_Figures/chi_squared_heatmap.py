"""Chi-squared heatmap: the SUHI × poverty 2×2 table on the primary sample.

Counts and percentages of n, a title with the statistics under it, and notes, with
the numbers of section 2.2: national median splits of the 54,243-tract primary
sample, the classifiers' own labels. The title leads with the effect size. Every
number is read from `2.2/results/chi_squared_results.json`; nothing is
re-estimated here.

`panel()` draws one heatmap figure into a Figure or SubFigure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from uhi_poverty import paths, plotting, report

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)
SOURCE = paths.ROOT / "2_Classification_Thresholds" / "2.2_Chi_Squared_Median" / "results"
SOURCE_JSON = SOURCE / "chi_squared_results.json"
FIGSIZE = (6.2, 7.4)


def panel(fig, counts: np.ndarray, row_labels: list[str], col_labels: list[str], title: str,
          stats: list[str], notes: list[str]) -> None:
    """The heatmap layout: title, statistics, 2×2 heatmap of counts and % of n, notes."""
    n = counts.sum()
    fig.text(0.5, 0.985, title, ha="center", va="top", fontsize=11, fontweight="bold",
             color=plotting.INK)
    y = 0.94
    for line in stats:
        fig.text(0.5, y, line, ha="center", va="top", fontsize=12, fontweight="bold",
                 color=plotting.INK)
        y -= 0.037
    ax = fig.add_axes([0.16, 0.2, 0.76, 0.64])
    norm = Normalize(vmin=0, vmax=counts.max())
    cmap = plotting.sequential_cmap()
    ax.imshow(counts, cmap=cmap, norm=norm, aspect="auto")
    for i in range(2):
        for j in range(2):
            r, g, b, _ = cmap(norm(counts[i, j]))
            dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5
            color = plotting.SURFACE if dark else plotting.INK
            ax.text(j, i - 0.08, f"{int(counts[i, j]):,}", ha="center", va="center",
                    fontsize=17, fontweight="bold", color=color)
            ax.text(j, i + 0.13, f"({counts[i, j] / n:.1%})", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)
    ax.set_xticks([0, 1], col_labels, fontsize=10)
    ax.set_yticks([0, 1], row_labels, fontsize=10, rotation=90, va="center")
    ax.tick_params(length=4)
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    y = 0.115
    for line in notes:
        fig.text(0.5, y, line, ha="center", va="top", fontsize=7.5, color=plotting.INK_2)
        y -= 0.025


def load() -> dict:
    return report.read_json(SOURCE_JSON)


def numbers(r: dict) -> pd.DataFrame:
    """Every number the figure shows, with where it comes from."""
    p, cuts = r["primary"], r["cut_values"]
    n = p["n"]
    src = paths.rel(SOURCE_JSON)
    rows = [
        ("Low SUHI, low poverty", p["n00"], p["n00"] / n),
        ("Low SUHI, high poverty", p["n01"], p["n01"] / n),
        ("High SUHI, low poverty", p["n10"], p["n10"] / n),
        ("High SUHI, high poverty", p["n11"], p["n11"] / n),
    ]
    out = [{"quantity": f"cell: {k}", "value": v, "share_of_n": s, "source": src}
           for k, v, s in rows]
    for q, v in [("n", n), ("odds_ratio", p["odds_ratio"]),
                 ("or_boot_ci_low", p["or_boot_ci_low"]),
                 ("or_boot_ci_high", p["or_boot_ci_high"]), ("bootstrap_B", p["B"]),
                 ("cramers_v", p["cramers_v"]), ("chi2", p["chi2"]), ("df", p["df"]),
                 ("p", p["p"]), ("cmh_or_by_ua", p["cmh_or"]),
                 ("cmh_or_ci_low", p["cmh_or_ci_low"]), ("cmh_or_ci_high", p["cmh_or_ci_high"]),
                 ("uhi_cut_degC", cuts["High_UHI"]["value"]),
                 ("poverty_rate_cut", cuts["High_Pov"]["value"])]:
        out.append({"quantity": q, "value": v, "share_of_n": np.nan, "source": src})
    return pd.DataFrame(out)


def texts(r: dict) -> dict:
    p, cuts = r["primary"], r["cut_values"]
    f = report.fmt_num
    uhi_cut, pov_cut = cuts["High_UHI"]["value"], cuts["High_Pov"]["value"]
    p_txt = "p < 0.001" if p["p"] < 0.001 else f"p = {p['p']:.3f}"
    return {
        "title": f"High-SUHI tracts have {f(p['odds_ratio'], 2)} times the odds of high poverty",
        "stats": [f"OR = {f(p['odds_ratio'], 2)} (95 % CI {f(p['or_boot_ci_low'], 2)}–"
                  f"{f(p['or_boot_ci_high'], 2)})    Cramér's V = {f(p['cramers_v'], 3)}",
                  f"χ²(1) = {p['chi2']:,.0f}, {p_txt}    n = {p['n']:,}"],
        "rows": [f"Low SUHI\n(< {uhi_cut:.2f} °C)", f"High SUHI\n(≥ {uhi_cut:.2f} °C)"],
        "cols": [f"Low poverty (< {pov_cut:.1%})", f"High poverty (≥ {pov_cut:.1%})"],
        "notes": [
            "Percentages are of n. Thresholds are the national medians of the primary sample "
            "(the classifiers' labels);",
            f"poverty = ACS 2015–2019 poverty rate. CI: {p['B']:,} bootstrap resamples of "
            f"the {p['n_clusters']} urbanized areas (UAs).",
            f"Stratified by UA, the Cochran–Mantel–Haenszel OR is {f(p['cmh_or'], 2)} "
            f"({f(p['cmh_or_ci_low'], 2)}–{f(p['cmh_or_ci_high'], 2)}).",
        ],
    }


def draw(fig, r: dict | None = None) -> None:
    """The heatmap into a Figure or SubFigure."""
    r = load() if r is None else r
    p = r["primary"]
    t = texts(r)
    counts = np.array([[p["n00"], p["n01"]], [p["n10"], p["n11"]]], dtype=float)
    panel(fig, counts, t["rows"], t["cols"], t["title"], t["stats"], t["notes"])


def write_readme(r: dict) -> None:
    nums = pd.read_csv(OUT / "chi_squared_heatmap.csv")
    p = nums.set_index("quantity")["value"]
    n_clusters = r["primary"]["n_clusters"]
    cells = nums[nums.quantity.str.startswith("cell: ")]
    f = report.fmt_num
    rows = "\n".join(f"| {q.removeprefix('cell: ')} | {int(v):,} | {report.fmt_pct(s)} |"
                     for q, v, s in zip(cells.quantity, cells.value, cells.share_of_n,
                                        strict=True))
    text = f"""# Chi-squared heatmap

`chi_squared_heatmap.py` draws the 2×2 heatmap of SUHI class against poverty class. It
reads every number from section 2.2
([`{paths.rel(SOURCE_JSON)}`](../{paths.rel(SOURCE_JSON)})) and does not re-estimate
anything.

- **Sample and thresholds.** The table is the primary national sample (n = {int(p['n']):,} tracts in
  {n_clusters} urbanized areas), split at the national medians that define the classifiers'
  labels.
- **Framing.** The title leads with the effect size. The statistics sit directly under it
  in bold: the odds ratio with its UA cluster-bootstrap CI and Cramér's V, then χ² with its
  degrees of freedom, its p-value and n.
- **Dependence.** The CI resamples urbanized areas, and a footnote gives the
  Cochran–Mantel–Haenszel OR stratified by UA.

## Numbers shown

| Cell | Count | Share of n |
|---|---:|---:|
{rows}

| Statistic | Value |
|---|---|
| Odds ratio (95 % UA cluster-bootstrap CI, B = {int(p['bootstrap_B']):,}) | {f(p['odds_ratio'], 2)} ({f(p['or_boot_ci_low'], 2)}–{f(p['or_boot_ci_high'], 2)}) |
| Cramér's V | {f(p['cramers_v'], 3)} |
| χ²(1) | {p['chi2']:,.1f}, p {report.fmt_p(p['p'])} |
| CMH OR by UA (95 % CI) | {f(p['cmh_or_by_ua'], 2)} ({f(p['cmh_or_ci_low'], 2)}–{f(p['cmh_or_ci_high'], 2)}) |
| n | {int(p['n']):,} |

All values: [`results/chi_squared_heatmap.csv`](results/chi_squared_heatmap.csv).
Figure: [`results/chi_squared_heatmap.png`](results/chi_squared_heatmap.png).
"""
    report.write_text(text, HERE / "README_chi_squared_heatmap.md")


def main() -> None:
    plotting.apply_style()
    r = load()
    numbers(r).to_csv(OUT / "chi_squared_heatmap.csv", index=False)
    fig = plt.figure(figsize=FIGSIZE)
    draw(fig, r)
    plotting.savefig(fig, OUT / "chi_squared_heatmap.png")
    write_readme(r)


if __name__ == "__main__":
    main()
