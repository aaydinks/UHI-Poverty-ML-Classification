"""Section 2.1: the national medians, class counts, and the precedent for a median split.

Writes `results/median_values.json` (national cut values and class counts for
`High_UHI`, `High_Pov` and `High_Pov_LPI`, plus the 66th/75th-percentile
sensitivity cuts) and the README.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from uhi_poverty import labels, paths, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)


def main() -> None:
    built = report.read_json(paths.MEDIANS_JSON)
    df = validation.load_analysis()
    p = validation.frame("primary", df)
    lpi = validation.frame("lpi", df)
    sens = {}
    for q in (0.50, 0.66, 0.75):
        u_cut = labels.quantile_cut(p["UHI_annual_day"], q)
        r_cut = labels.quantile_cut(p["pov_rate"], q)
        sens[f"p{int(q * 100)}"] = {
            "UHI_annual_day_cut": u_cut,
            "High_UHI": labels.class_counts(labels.high_uhi(p["UHI_annual_day"], u_cut)),
            "pov_rate_cut": r_cut,
            "High_Pov": labels.class_counts(labels.high_pov(p["pov_rate"], r_cut)),
        }
    out = {
        "primary_n": len(p),
        "cuts": built["cuts"],
        "class_counts": {
            "High_UHI": labels.class_counts(p["High_UHI"]),
            "High_Pov": labels.class_counts(p["High_Pov"]),
            "High_Pov_LPI": labels.class_counts(lpi["High_Pov_LPI"]),
        },
        "percentile_sensitivity": sens,
    }
    report.write_json(out, OUT / "median_values.json")
    rows = []
    for name, c in out["class_counts"].items():
        cut = out["cuts"][name]
        rows.append({"label": name, "rule": f"{cut['column']} {cut['rule']}",
                     "cut": cut["value"], "n": c["n"], "positives": c["n_pos"],
                     "negatives": c["n_neg"], "share_positive": c["share_pos"]})
    report.write_table(pd.DataFrame(rows), OUT / "class_counts", digits=4,
                       title="National cut values and class counts")
    write_readme()


def write_readme() -> None:
    m = report.read_json(OUT / "median_values.json")
    f = report.fmt_num
    c, cc, s = m["cuts"], m["class_counts"], m["percentile_sensitivity"]
    rows = "\n".join(
        f"| `{k}` | `{c[k]['column']}` {c[k]['rule']} | {f(c[k]['value'], 4)} | {cc[k]['n']:,} | "
        f"{cc[k]['n_pos']:,} | {cc[k]['n_neg']:,} |" for k in ("High_UHI", "High_Pov", "High_Pov_LPI"))
    srows = "\n".join(
        f"| {q[1:]}th | {f(v['UHI_annual_day_cut'], 3)} °C | {v['High_UHI']['n_pos']:,} / "
        f"{v['High_UHI']['n_neg']:,} | {report.fmt_pct(v['pov_rate_cut'])} | "
        f"{v['High_Pov']['n_pos']:,} / {v['High_Pov']['n_neg']:,} |" for q, v in s.items())
    text = f"""# 2.1 Median split and its precedent

## Purpose

Sets one cut-off for each outcome, gives its rationale, and reports the class counts.

## Method

Both outcomes are split once, at the national median of the primary analysis sample, in
`src/uhi_poverty/labels.py`; every model and the chi-squared test (section 2.2) import the same
labels. `medians.py` records the cut values and class counts in
[`results/median_values.json`](results/median_values.json), with the 66th and 75th percentiles
as sensitivity cut-points.

## Results

Primary analysis sample, n = {m['primary_n']:,} tracts
([`results/class_counts.md`](results/class_counts.md)):

| Label | Rule | Cut | n | Positive | Negative |
|---|---|---:|---:|---:|---:|
{rows}

The high-poverty median is a tract poverty rate of {report.fmt_pct(c['High_Pov']['value'])}. The
LPI label is the robustness outcome; because a low index means more poverty, its positive class
is the tracts *below* the LPI median.

Sensitivity cut-points (positives / negatives):

| Percentile | UHI cut | High_UHI | Poverty-rate cut | High_Pov |
|---|---:|---:|---:|---:|
{srows}

## Precedent

Three studies of urban heat split a heat measure at, or close to, its median, two of them
alongside a median-type split of a poverty or income measure, as here:

1. **Madrigano J, Ito K, Johnson S, Kinney PL, Matte T.** A case-only study of vulnerability
   to heat wave–related mortality in New York City (2000–2011). *Environ Health Perspect*
   2015;123(7):672–678. doi:10.1289/ehp.1408178.
   Census tracts in New York City, with daytime summer surface temperature from a Landsat
   thermal scene. Every neighbourhood modifier, surface temperature and the share of
   households on public assistance among them, is dichotomised at its median (high = at or
   above the median); the 75th percentile is a sensitivity cut. This is the closest design:
   the same unit (tracts), satellite surface temperature, and a median split of both heat
   and a poverty measure. It differs in using the citywide median of tract surface
   temperature rather than the national median of urban-minus-rural SUHI.
2. **Goggins WB, Chan EYY, Ng E, Ren C, Chen L.** Effect modification of the association
   between short-term meteorological factors and mortality by urban heat islands in Hong
   Kong. *PLoS ONE* 2012;7(6):e38551. doi:10.1371/journal.pone.0038551.
   An urban heat island index for 248 planning units is split at its Hong Kong-wide median
   (3.67) into "hot" and "cool" areas and crossed with a split of median household income
   that divides deaths roughly in half, a 2×2 design of high/low heat island by high/low
   socioeconomic status like the one in section 2.2. The index is modelled from urban form
   (sky-view factor, greenery, wind), not satellite LST.
3. **Smargiassi A, Goldberg MS, Plante C, Fournier M, Baudouin Y, Kosatsky T.** Variation of
   daily warm season mortality as a function of micro-urban heat islands. *J Epidemiol
   Community Health* 2009;63(8):659–664. doi:10.1136/jech.2008.078147.
   Montreal deaths assigned the satellite surface temperature at the residence, split into
   two strata at the 50th, 66th and 75th percentiles (the main analysis used the 75th), with
   dwelling value split at the 25th, 33rd and 50th percentiles.

**How it maps here.** Surface UHI intensity has no external standard comparable to a clinical
threshold, so a distribution-based cut-point is the defensible choice, as in these studies,
and the median is the cut they use for both heat and socioeconomic status. A median split
gives two classes of equal size for both outcomes, which keeps accuracy-type metrics
interpretable and lets the same cut serve the classifiers and the association test.
Following the same practice, the 66th and 75th percentiles are reported as sensitivity
cut-points (section 2.2).
"""
    report.write_text(text, HERE / "README.md")


if __name__ == "__main__":
    main()
