"""Section 1.2: national maps of surface urban heat on 2010 tract geometry.

F1  tract choropleth of `UHI_annual_day`, conterminous U.S. in Albers equal-area,
    with Alaska, Hawaii and Puerto Rico insets; diverging scale centred at 0 °C
    with robust symmetric limits (2nd–98th percentile).
F2  one point per UA at its centroid, coloured by median tract SUHI, sized by tracts.
F3  zoom panels for the largest UA in each Census region.
F4  tract-level summer daytime urban LST (`LST_urb_day_CT_summer`).
Also writes `results/ua_summary.csv` (feeds F2) and `results/map_coverage.json`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

from uhi_poverty import geo, maps, paths, plotting, report, validation

HERE = Path(__file__).resolve().parent
OUT = paths.results_dir(__file__)


def suhi_frame() -> pd.DataFrame:
    """Every SUHI tract with a daytime SUHI value (50 states + DC + Puerto Rico)."""
    df = validation.load_analysis()
    return df[df["UHI_annual_day"].notna()].copy()


def robust_limit(values: pd.Series) -> float:
    lo, hi = np.percentile(values, [2, 98])
    return float(max(abs(lo), abs(hi)))


def colorbar(fig, mappable, rect, label):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_label(label, color=plotting.INK_2, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)
    return cb


def f1(gdf, states, lim) -> None:
    fig = plt.figure(figsize=(10, 6.6))
    axes = maps.national_axes(fig)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    maps.draw(axes, gdf, states, {"cmap": plotting.suhi_cmap(), "norm": norm},
              column="UHI_annual_day")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plotting.suhi_cmap())
    colorbar(fig, sm, [0.56, 0.06, 0.38, 0.025],
             "Daytime surface UHI intensity, annual (°C): tract LST − rural reference LST")
    axes["conus"].set_title("Surface urban heat island intensity by census tract, 2010 tracts",
                            loc="left", fontsize=11)
    plotting.note(fig, f"{len(gdf):,} SUHI tracts with geometry. Colour limits ±{lim:.1f} °C "
                       "(2nd–98th percentile, symmetric); values beyond are clipped. Grey = no "
                       "SUHI value (not urban). Source: Chakraborty et al. (2020).", y=0.0)
    plotting.savefig(fig, OUT / "F1_national_tract_uhi.png")


def ua_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("ua")
    out = pd.DataFrame({
        "ua_name": g["ua_name"].first(),
        "n_tracts": g.size(),
        "median_uhi_annual_day": g["UHI_annual_day"].median(),
        "mean_uhi_annual_day": g["UHI_annual_day"].mean(),
        "ua_uhi_annual_day_city": g["UHI_annual_day_city"].first(),
        "lat": g["lat"].mean(),
        "lon": g["lon"].mean(),
    }).reset_index()
    out["primary_state"] = out["ua"].map(geo.ua_primary_state(df["ua"], df["state_fips"]))
    out["region"] = geo.lookup(out["primary_state"], "region")
    out["in_primary_sample"] = out["primary_state"] != geo.PUERTO_RICO
    return out.sort_values("n_tracts", ascending=False).reset_index(drop=True)


def f2(ua: pd.DataFrame, states, lim) -> None:
    import geopandas as gpd

    pts = gpd.GeoDataFrame(ua, geometry=gpd.points_from_xy(ua["lon"], ua["lat"]), crs="EPSG:4269")
    pts["state_fips"] = pts["primary_state"]
    pts = pts.sort_values("n_tracts", ascending=False)
    fig = plt.figure(figsize=(10, 6.6))
    axes = maps.national_axes(fig)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    sizes = 4 + 140 * np.sqrt(pts["n_tracts"] / pts["n_tracts"].max())
    for key, ax in axes.items():
        fips = None if key == "conus" else key
        crs = maps.CONUS_CRS if key == "conus" else maps.NON_CONUS[key]
        st = maps._part(states, fips, crs)
        st.plot(ax=ax, color=plotting.NEUTRAL, linewidth=0)
        st.boundary.plot(ax=ax, color="white", linewidth=0.5)
        p = maps._part(pts, fips, crs)
        if len(p):
            ax.scatter(p.geometry.x, p.geometry.y, s=sizes.loc[p.index], c=p["median_uhi_annual_day"],
                       cmap=plotting.suhi_cmap(), norm=norm, edgecolors=plotting.SURFACE,
                       linewidths=0.6, zorder=3)
        if key == "conus":
            ax.set_xlim(-2.4e6, 2.3e6)
            ax.set_ylim(2.6e5, 3.2e6)
        ax.set_aspect("equal")
    axes["02"].set_title("Alaska", fontsize=7, loc="left", color=plotting.INK_2, fontweight="normal")
    axes["15"].set_title("Hawaii", fontsize=7, loc="left", color=plotting.INK_2, fontweight="normal")
    axes["72"].set_title("Puerto Rico\n(sensitivity frame)", fontsize=7, loc="left",
                         color=plotting.INK_2, fontweight="normal")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plotting.suhi_cmap())
    colorbar(fig, sm, [0.56, 0.06, 0.38, 0.025], "Median tract daytime SUHI in the UA (°C)")
    axes["conus"].set_title(f"Surface urban heat by urbanized area ({len(ua):,} UAs)",
                            loc="left", fontsize=11)
    plotting.note(fig, "One point per urbanized area at the mean of its tract centroids; area "
                       "proportional to its number of tracts.", y=0.0)
    plotting.savefig(fig, OUT / "F2_national_ua_uhi.png")


def f3(gdf, df, lim) -> list[dict]:
    p = df[~df["territory"]]
    counts = p.groupby(["region", "ua"]).size().rename("n").reset_index()
    largest = counts.sort_values(["region", "n"], ascending=[True, False]).drop_duplicates("region")
    order = [r for r in geo.REGIONS if r in set(largest["region"])]
    largest = largest.set_index("region").loc[order].reset_index()
    names = df.groupby("ua")["ua_name"].first()
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.4))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    panels = []
    for ax, row in zip(axes.flat, largest.itertuples(), strict=True):
        geoids = set(df.loc[df["ua"] == row.ua, "GEOID"])
        sub = gdf[gdf["GEOID"].isin(geoids)].to_crs(maps.CONUS_CRS)
        sub.plot(ax=ax, column="UHI_annual_day", cmap=plotting.suhi_cmap(), norm=norm,
                 linewidth=0.05, edgecolor=plotting.SURFACE, rasterized=True)
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_title(f"{row.region}: {names[row.ua]}\n{row.n:,} tracts", fontsize=9)
        panels.append({"region": row.region, "ua": row.ua, "ua_name": names[row.ua],
                       "n_tracts": int(row.n)})
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plotting.suhi_cmap())
    colorbar(fig, sm, [0.3, 0.04, 0.4, 0.015], "Daytime surface UHI intensity (°C)")
    fig.suptitle("Largest urbanized area in each Census region", x=0.02, ha="left",
                 fontweight="bold", fontsize=11)
    plotting.savefig(fig, OUT / "F3_metro_panels_uhi.png")
    return panels


def f4(gdf, states) -> None:
    v = gdf["LST_urb_day_CT_summer"]
    lo, hi = np.percentile(v, [2, 98])
    fig = plt.figure(figsize=(10, 6.6))
    axes = maps.national_axes(fig)
    norm = Normalize(vmin=lo, vmax=hi)
    cmap = plotting.sequential_cmap()
    from matplotlib.colors import LinearSegmentedColormap

    warm = LinearSegmentedColormap.from_list(
        "warm", ["#fde3d9", "#f6b49a", "#ef8a62", "#e34948", "#b52a2a", "#7a1414"], N=256)
    cmap = warm
    maps.draw(axes, gdf, states, {"cmap": cmap, "norm": norm}, column="LST_urb_day_CT_summer")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar(fig, sm, [0.56, 0.06, 0.38, 0.025], "Summer daytime land-surface temperature (°C)")
    axes["conus"].set_title("Summer daytime surface temperature by census tract (LST_urb_day_CT_summer)",
                            loc="left", fontsize=11)
    plotting.note(fig, f"Colour limits {lo:.1f}–{hi:.1f} °C (2nd–98th percentile).", y=0.0)
    plotting.savefig(fig, OUT / "F4_national_tract_summer_lst.png")


def main() -> None:
    plotting.apply_style()
    df = suhi_frame()
    shapes = maps.load_tracts()
    states = maps.load_states()
    gdf = shapes.merge(df[["GEOID", "UHI_annual_day", "LST_urb_day_CT_summer"]], on="GEOID",
                       how="inner", validate="one_to_one")
    lim = robust_limit(df["UHI_annual_day"])
    ua = ua_summary(df)
    ua.to_csv(OUT / "ua_summary.csv", index=False)
    f1(gdf, states, lim)
    f2(ua, states, lim)
    panels = f3(gdf, df, lim)
    f4(gdf, states)
    unmatched = sorted(set(df["GEOID"]) - set(shapes["GEOID"]))
    report.write_json({
        "suhi_tracts_with_value": int(len(df)),
        "matched_to_cb2019_geometry": int(len(gdf)),
        "unmatched": unmatched,
        "colour_limit_degC": lim,
        "share_negative_suhi": float((df["UHI_annual_day"] < 0).mean()),
        "n_uas": int(len(ua)),
        "metro_panels": panels,
    }, OUT / "map_coverage.json")
    write_readme()


def write_readme() -> None:
    c = report.read_json(OUT / "map_coverage.json")
    panels = ", ".join(f"{p['ua_name']} ({p['region']})" for p in c["metro_panels"])
    water = [g for g in c["unmatched"] if g[5:7] == "99"]
    other = [g for g in c["unmatched"] if g[5:7] != "99"]
    text = f"""# 1.2 National heat map

## Purpose

Maps surface urban heat across every urbanized area in the country, on the same 2010 tract
geometry as the data.

## Method

`make_heatmap.py` draws four figures from the SUHI database on 2019 cartographic boundary
tracts (1:500k), which use 2010 tract definitions, the same vintage as the data:

- [`results/F1_national_tract_uhi.png`](results/F1_national_tract_uhi.png): daytime annual SUHI
  intensity (`UHI_annual_day`) for every tract, conterminous U.S. in Albers equal-area
  (EPSG:5070), with Alaska (EPSG:3338), Hawaii and Puerto Rico insets. Puerto Rico is labelled
  as the sensitivity frame.
- [`results/F2_national_ua_uhi.png`](results/F2_national_ua_uhi.png): one point per urbanized
  area at its centroid, coloured by median tract SUHI and sized by tract count. Urban tracts are
  tiny at national scale, so this is the readable national view
  ([`results/ua_summary.csv`](results/ua_summary.csv) holds the values).
- [`results/F3_metro_panels_uhi.png`](results/F3_metro_panels_uhi.png): the largest UA in each
  Census region ({panels}).
- [`results/F4_national_tract_summer_lst.png`](results/F4_national_tract_summer_lst.png):
  tract-level summer daytime urban LST (`LST_urb_day_CT_summer`), one value per tract.

## Map details

- **Geometry.** The maps use 2019 cartographic boundaries, which carry 2010 tract codes, so the
  data join the polygons without tracts dropping out through later splits or renumbering. {c['matched_to_cb2019_geometry']:,} of the
  {c['suhi_tracts_with_value']:,} SUHI tracts with a value match a polygon. Of the
  {len(c['unmatched'])} that do not, {len(water)} are water-only tracts (tract code 99xxxx) that the
  cartographic file omits and {len(other)} were renumbered after 2010 ({', '.join(other)}).
- **Colour scale.** SUHI can be negative ({report.fmt_pct(c['share_negative_suhi'])} of tracts are
  cooler than their rural reference), so the scale is diverging, blue below and red above 0 °C,
  with a neutral grey midpoint at 0 °C. Its limits are symmetric, ±{c['colour_limit_degC']:.1f} °C,
  set by the 2nd–98th percentiles so a few extreme tracts do not wash out the rest.
- **Scope.** All {c['n_uas']:,} urbanized areas in the database are shown.
"""
    report.write_text(text, HERE / "README.md")


if __name__ == "__main__":
    main()
