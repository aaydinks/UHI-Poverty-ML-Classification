"""Section 2.1: the national medians and class counts.

Writes `results/median_values.json` (national cut values and class counts for
`High_UHI`, `High_Pov` and `High_Pov_LPI`, plus the 66th/75th-percentile
sensitivity cuts, and the Spearman correlation between the Low Poverty Index and the
poverty rate).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

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
        # direction check: the LPI falls as the poverty rate rises
        "lpi_vs_poverty_rate": {"n": len(lpi), "spearman_rho": float(spearmanr(lpi["LPI"],
                                                                              lpi["pov_rate"])[0])},
    }
    report.write_json(out, OUT / "median_values.json")
    rows = []
    for name, c in out["class_counts"].items():
        cut = out["cuts"][name]
        rows.append({"label": name, "rule": f"{cut['column']} {cut['rule']}",
                     "cut": cut["value"], "n": c["n"], "positives": c["n_pos"],
                     "negatives": c["n_neg"], "share_positive": c["share_pos"]})
    report.write_table(pd.DataFrame(rows), OUT / "class_counts")


if __name__ == "__main__":
    main()
