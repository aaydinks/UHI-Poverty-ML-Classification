"""Section 3.1: check every difference identity in the SUHI file.

- the 6 tract SUHI variants against `LST_urb_*_CT − LST_rur_*`;
- the 6 `_city` variants against `LST_urb_*_all − LST_rur_*`;
- `DelNDVI_*` (tract and city) and `DelDEM` against their urban − rural pairs;
- and, for contrast, the tract SUHI against the `_CT_act` LST the documentation
  calls ideal, which the file's `UHI_*` columns do not use.

Writes `results/identity_checks.{csv,md}` and the README.
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
    report.write_table(t, OUT / "identity_checks", digits={"max_abs_error": 2,
                       "mean_abs_error": 2, "pearson_r": 6, "share_exact_1e-6": 4,
                       "max_abs_error_if_rural_minus_urban": 2},
                       title="Difference identities in the SUHI file",
                       note="Errors in the variable's units (°C, NDVI, metres); all 55,871 rows "
                            "with the three columns present.")
    # scientific notation for the tiny errors in the Markdown version
    md = t.copy()
    for c in ("max_abs_error", "mean_abs_error", "max_abs_error_if_rural_minus_urban"):
        md[c] = md[c].map(lambda v: f"{v:.1e}")
    report.write_text("# Difference identities in the SUHI file\n\n" + report.md_table(md, 6)
                      + "\nErrors are in the variable's units (°C, NDVI, metres).\n",
                      OUT / "identity_checks.md")
    write_readme()


def write_readme() -> None:
    t = pd.read_csv(OUT / "identity_checks.csv")
    main_ = t[t.group != "contrast: _CT_act"]
    act = t[t.group == "contrast: _CT_act"]
    ann = t[(t.target == "UHI_annual_day") & (t.group == "tract SUHI")].iloc[0]
    act_ann = act[act.target == "UHI_annual_day"].iloc[0]
    n_hold = int(main_["holds"].sum())
    groups = "\n".join(
        f"| {g} | {len(s)} | {int(s['holds'].sum())} | {s['max_abs_error'].max():.1e} | "
        f"{s['pearson_r'].min():.6f} |" for g, s in main_.groupby("group", sort=False))
    text = f"""# 3.1 The exact UHI formula

## Purpose

This section states the exact definition of the urban heat island target and checks it in the
data, so that the model for high UHI can exclude every input the target is built from.

## The formula

```
UHI_annual_day = LST_urb_day_CT − LST_rur_day
```

Definitions, paraphrased from the dataset documentation (Chakraborty, Hsu, Sheriff & Manya,
*United States Surface Urban Heat Island database*, Mendeley Data v3, 2020,
doi:10.17632/x9mv4krnm2.3, CC BY 4.0):

- `LST_urb_day_CT`: annual daytime land-surface temperature, the spatial mean of the pixels that
  intersect the census tract, clipped to the urbanized area (suffix `_CT`).
- `LST_rur_day`: the rural reference, the spatial mean of the non-urban, non-water pixels in
  the urbanized area's region of interest. There is one value per UA.
- `_all`: the spatial mean of all pixels intersecting the urbanized area (one value per UA);
  `_CT_act`: the spatial mean of all available pixels intersecting the tract, without clipping.

The documentation names `LST_urb_CT_act − LST_rur` as the ideal tract configuration and
`LST_urb_all − LST_rur` as the ideal UA-level one. The file's tract `UHI_*` columns use `_CT`,
not `_CT_act`: the `_CT_act` difference misses `UHI_annual_day` by up to
{act_ann.max_abs_error:.2f} °C, while the `_CT` difference matches it to
{ann.max_abs_error:.0e} °C.

## Method

`verify_formula.py` recomputes every difference variable in the file from its urban and rural
parts and records the maximum absolute error, the mean absolute error and Pearson r
([`results/identity_checks.md`](results/identity_checks.md)).

## Results

| Group | Identities | Hold (error < 1e-6) | Largest error | Smallest r |
|---|---:|---:|---:|---:|
{groups}

{n_hold} of {len(main_)} identities hold to floating-point precision over all
{int(ann.n):,} tracts with a value. In particular `UHI_annual_day` equals
`LST_urb_day_CT − LST_rur_day` with a maximum error of {ann.max_abs_error:.1e} °C
(r = {ann.pearson_r:.6f}).

Every `UHI_*` column is therefore an urban-minus-rural LST difference, and every `_city`
column is a UA-level constant built the same way. A model of `High_UHI` that sees any `LST_*`
or `UHI_*` column can reconstruct, or closely approximate, its own target, so section 3.2
removes both families from the UHI model.
"""
    report.write_text(text, HERE / "README.md")


if __name__ == "__main__":
    main()
