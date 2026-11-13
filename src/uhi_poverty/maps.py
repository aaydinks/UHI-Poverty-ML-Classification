"""Boundary loading and the national map layout shared by sections 1.2 and 1.4.

Geometry is the 2019 cartographic boundary file (1:500k), which is drawn on 2010
tract definitions. The conterminous U.S. is drawn in Albers equal-area
(EPSG:5070), with insets for Alaska (EPSG:3338) and Hawaii.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from . import geo, paths, plotting

CONUS_CRS = "EPSG:5070"
AK_CRS = "EPSG:3338"
HI_CRS = "ESRI:102007"  # Hawaii Albers equal-area conic
NON_CONUS = {"02": AK_CRS, "15": HI_CRS}


def load_tracts() -> gpd.GeoDataFrame:
    frames = []
    for path in sorted(paths.TRACTS_DIR.glob("cb_2019_*_tract_500k.zip")):
        frames.append(gpd.read_file(f"zip://{path}", columns=["GEOID"]))
    gdf = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=frames[0].crs)
    gdf["GEOID"] = geo.normalize_geoid(gdf["GEOID"])
    gdf["state_fips"] = gdf["GEOID"].str[:2]
    return gdf[geo.in_study_area(gdf["GEOID"])].reset_index(drop=True)


def load_states() -> gpd.GeoDataFrame:
    s = gpd.read_file(f"zip://{paths.STATES_ZIP}", columns=["STATEFP", "STUSPS"])
    return s[s["STATEFP"].isin(list(geo.STATES))]


# Inset extents (lon/lat): the main Hawaiian islands, and Alaska without the far Aleutians.
INSET_BOX = {"15": (-160.6, 18.8, -154.6, 22.4), "02": (-170.0, 51.0, -129.0, 71.6)}


def _part(gdf: gpd.GeoDataFrame, fips: str | None, crs: str) -> gpd.GeoDataFrame:
    col = "state_fips" if "state_fips" in gdf.columns else "STATEFP"
    if fips is None:
        sub = gdf[~gdf[col].isin(list(NON_CONUS))]
    else:
        sub = gdf[gdf[col] == fips]
        if fips in INSET_BOX and len(sub):
            sub = gpd.clip(sub.to_crs("EPSG:4269"), box(*INSET_BOX[fips]))
    return sub.to_crs(crs)


def national_axes(fig):
    """Main CONUS axis plus Alaska and Hawaii insets."""
    main = fig.add_axes([0.0, 0.12, 1.0, 0.86])
    ak = fig.add_axes([0.0, 0.0, 0.2, 0.26])
    hi = fig.add_axes([0.2, 0.02, 0.14, 0.16])
    axes = {"conus": main, "02": ak, "15": hi}
    for ax in axes.values():
        ax.set_axis_off()
    return axes


def draw(axes: dict, tracts: gpd.GeoDataFrame, states: gpd.GeoDataFrame, plot_kw: dict,
         column: str | None = None) -> None:
    """Draw `tracts` (coloured by `column`, or `plot_kw['color']`) into the layout."""
    for key, ax in axes.items():
        fips = None if key == "conus" else key
        crs = CONUS_CRS if key == "conus" else NON_CONUS[key]
        t = _part(tracts, fips, crs)
        s = _part(states, fips, crs)
        s.plot(ax=ax, color=plotting.NEUTRAL, linewidth=0, zorder=0)
        if len(t):
            kw = dict(plot_kw)
            if column is not None:
                kw["column"] = column
            t.plot(ax=ax, linewidth=0, rasterized=True, zorder=1, **kw)
        s.boundary.plot(ax=ax, color="white", linewidth=0.4, zorder=2)
        if key == "conus":
            ax.set_xlim(-2.4e6, 2.3e6)
            ax.set_ylim(2.6e5, 3.2e6)
        ax.set_aspect("equal")
    axes["02"].set_title("Alaska", fontsize=7, loc="left", color=plotting.INK_2,
                         fontweight="normal")
    axes["15"].set_title("Hawaii", fontsize=7, loc="left", color=plotting.INK_2,
                         fontweight="normal")
