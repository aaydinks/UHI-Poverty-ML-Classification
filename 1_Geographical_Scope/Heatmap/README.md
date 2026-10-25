# 1.2 National heat map

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
  Census region (New York--Newark, NY--NJ--CT (Northeast), Chicago, IL--IN (Midwest), Miami, FL (South), Los Angeles--Long Beach--Anaheim, CA (West)).
- [`results/F4_national_tract_summer_lst.png`](results/F4_national_tract_summer_lst.png):
  tract-level summer daytime urban LST (`LST_urb_day_CT_summer`), one value per tract.

## Map details

- **Geometry.** The maps use 2019 cartographic boundaries, which carry 2010 tract codes, so the
  data join the polygons without tracts dropping out through later splits or renumbering. 55,629 of the
  55,649 SUHI tracts with a value match a polygon. Of the
  20 that do not, 11 are water-only tracts (tract code 99xxxx) that the
  cartographic file omits and 9 were renumbered after 2010 (04019002701, 04019002903, 04019410501, 04019410502, 04019410503, 04019470500, 06037930401, 36053940401, 36085008900).
- **Colour scale.** SUHI can be negative (23.0 % of tracts are
  cooler than their rural reference), so the scale is diverging, blue below and red above 0 °C,
  with a neutral grey midpoint at 0 °C. Its limits are symmetric, ±5.3 °C,
  set by the 2nd–98th percentiles so a few extreme tracts do not wash out the rest.
- **Scope.** All 497 urbanized areas in the database are shown.
