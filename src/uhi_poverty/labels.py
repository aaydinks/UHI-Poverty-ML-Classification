"""The one set of labels every model and the chi-squared test share.

- `High_UHI = UHI_annual_day >= median` (national, primary sample)
- `High_Pov = ACS poverty rate >= median` (national, primary sample)
- `High_Pov_LPI = LPI < LPI median` (LPI robustness frame). HUD's Low Poverty
  Index rises as poverty falls, so a *low* index marks the poorer tracts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def medians(df: pd.DataFrame, primary: pd.Series, lpi_frame: pd.Series) -> dict:
    return {
        "UHI_annual_day": float(df.loc[primary, "UHI_annual_day"].median()),
        "pov_rate": float(df.loc[primary, "pov_rate"].median()),
        "LPI": float(df.loc[lpi_frame, "LPI"].median()),
    }


def quantile_cut(values: pd.Series, q: float) -> float:
    return float(values.quantile(q))


def high_uhi(uhi: pd.Series, cut: float) -> pd.Series:
    return (uhi >= cut).astype("Int8").where(uhi.notna())


def high_pov(rate: pd.Series, cut: float) -> pd.Series:
    return (rate >= cut).astype("Int8").where(rate.notna())


def high_pov_lpi(lpi: pd.Series, cut: float) -> pd.Series:
    """Low index = poorer, so the high-poverty class is LPI below the median."""
    return (lpi < cut).astype("Int8").where(lpi.notna())


def add_labels(df: pd.DataFrame, cuts: dict) -> pd.DataFrame:
    out = df.copy()
    out["High_UHI"] = high_uhi(out["UHI_annual_day"], cuts["UHI_annual_day"])
    out["High_Pov"] = high_pov(out["pov_rate"], cuts["pov_rate"])
    out["High_Pov_LPI"] = high_pov_lpi(out["LPI"], cuts["LPI"])
    return out


def assert_orientation(df: pd.DataFrame) -> dict:
    """Positives of both poverty labels must have the higher mean ACS poverty rate."""
    checks = {}
    for label in ("High_Pov", "High_Pov_LPI"):
        sub = df[df[label].notna() & df["pov_rate"].notna()]
        pos = float(sub.loc[sub[label] == 1, "pov_rate"].mean())
        neg = float(sub.loc[sub[label] == 0, "pov_rate"].mean())
        if not pos > neg:
            raise AssertionError(f"{label}: positives are not the poorer tracts ({pos} <= {neg})")
        checks[label] = {"mean_pov_rate_pos": pos, "mean_pov_rate_neg": neg}
    sub = df[df["High_UHI"].notna()]
    pos = float(sub.loc[sub["High_UHI"] == 1, "UHI_annual_day"].mean())
    neg = float(sub.loc[sub["High_UHI"] == 0, "UHI_annual_day"].mean())
    if not pos > neg:
        raise AssertionError("High_UHI positives are not the hotter tracts")
    checks["High_UHI"] = {"mean_uhi_pos": pos, "mean_uhi_neg": neg}
    return checks


def class_counts(series: pd.Series) -> dict:
    s = series.dropna().astype(int)
    return {"n": int(len(s)), "n_pos": int(s.sum()), "n_neg": int((1 - s).sum()),
            "share_pos": float(s.mean()) if len(s) else np.nan}
