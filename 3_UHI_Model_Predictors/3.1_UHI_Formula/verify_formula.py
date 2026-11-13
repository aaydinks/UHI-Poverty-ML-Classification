"""Section 3.1: check every difference identity in the SUHI file.

- the 6 tract SUHI variants against `LST_urb_*_CT − LST_rur_*`;
- the 6 `_city` variants against `LST_urb_*_all − LST_rur_*`;
- `DelNDVI_*` (tract and city) and `DelDEM` against their urban − rural pairs;
- and, for contrast, the tract SUHI against the `_CT_act` LST the documentation
  calls ideal, which the file's `UHI_*` columns do not use.

Writes `results/identity_checks.csv`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from uhi_poverty import paths, report

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)

WINDOWS = [("annual", "day", ""), ("annual", "night", ""), ("summer", "day", "_summer"),
           ("summer", "night", "_summer"), ("winter", "day", "_winter"),
           ("winter", "night", "_winter")]


def identities() -> list[tuple[str, str, str, str]]:
    """(group, target, urban term, rural term)."""
    out = []
    for season, tod, suf in WINDOWS:
        rur = f"LST_rur_{tod}{suf}"
        out.append(("tract SUHI", f"UHI_{season}_{tod}", f"LST_urb_{tod}_CT{suf}", rur))
    for season, tod, suf in WINDOWS:
        rur = f"LST_rur_{tod}{suf}"
        out.append(("city SUHI", f"UHI_{season}_{tod}_city", f"LST_urb_{tod}_all{suf}", rur))
    for season, suf in (("annual", ""), ("summer", "_summer"), ("winter", "_winter")):
        out.append(("tract dNDVI", f"DelNDVI_{season}", f"NDVI_urb_CT{suf}", f"NDVI_rur{suf}"))
        out.append(("city dNDVI", f"DelNDVI_{season}_city", f"NDVI_urb_all{suf}", f"NDVI_rur{suf}"))
    out.append(("dDEM", "DelDEM", "DEM_urb_CT", "DEM_rur"))
    for season, tod, suf in WINDOWS:
        rur = f"LST_rur_{tod}{suf}"
        out.append(("contrast: _CT_act", f"UHI_{season}_{tod}", f"LST_urb_{tod}_CT_act{suf}", rur))
    return out


def check(df: pd.DataFrame, target: str, urb: str, rur: str) -> dict:
    sub = df[[target, urb, rur]].dropna()
    diff = sub[urb] - sub[rur]
    err = (sub[target] - diff).abs()
    flipped = (sub[target] + diff).abs()
    return {"n": len(sub), "max_abs_error": float(err.max()), "mean_abs_error": float(err.mean()),
            "share_exact_1e-6": float((err < 1e-6).mean()),
            "pearson_r": float(np.corrcoef(sub[target], diff)[0, 1]),
            "max_abs_error_if_rural_minus_urban": float(flipped.max())}


def main() -> None:
    df = pd.read_csv(paths.SUHI_CSV)
    rows = []
    for group, target, urb, rur in identities():
        rows.append({"group": group, "target": target, "identity": f"{urb} − {rur}",
                     **check(df, target, urb, rur)})
    t = pd.DataFrame(rows)
    t["holds"] = t["max_abs_error"] < 1e-6
    report.write_table(t, OUT / "identity_checks")
    # scientific notation for the tiny errors in the Markdown version
    md = t.copy()
    for c in ("max_abs_error", "mean_abs_error", "max_abs_error_if_rural_minus_urban"):
        md[c] = md[c].map(lambda v: f"{v:.1e}")


if __name__ == "__main__":
    main()
