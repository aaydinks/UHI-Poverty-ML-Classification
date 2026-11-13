# 1.4 Coverage by state and urbanized area

## Purpose

Shows which states and cities are in the sample, how much of each is covered, and which are left
out.

## Method

`coverage.py` tabulates the analysis sample by state and by urbanized area (UA) and maps the
status of every 2010 tract.

- [`results/tracts_by_state.md`](results/tracts_by_state.md): all 51 jurisdictions, with the
  analysis n, SUHI n, all 2010 tracts, coverage %, and the share of tracts that are
  `High_UHI` and `High_Pov`.
- [`results/tracts_by_urban_area.csv`](results/tracts_by_urban_area.csv): every included UA,
  with its name (principal cities), primary state, every state it spans, n, % `High_UHI` and
  % `High_Pov`. [`results/major_urban_areas.md`](results/major_urban_areas.md) lists the 50
  largest.
- [`results/included_excluded.md`](results/included_excluded.md): jurisdictions and UAs left
  out, and why.
- [`results/coverage_map_tracts.png`](results/coverage_map_tracts.png): every 2010 tract
  coloured by status; [`results/coverage_by_state_map.png`](results/coverage_by_state_map.png):
  analysis tracts per state.

## Results

All 51 jurisdictions (50 states + DC) and 486 urbanized areas
are in the analysis sample. Coverage (analysis tracts / all 2010 tracts) is highest in
DC (100 %), RI (95 %), NJ (95 %) and lowest in
VT (16 %), MT (25 %), SD (28 %), reflecting how
urban each state is. The largest UAs are
New York--Newark, NY--NJ--CT (4,217), Los Angeles--Long Beach--Anaheim, CA (2,758), Chicago, IL--IN (2,059), Philadelphia, PA--NJ--DE--MD (1,395), Miami, FL (1,188); 69 UAs span more
than one state.

Tracts on the cartographic map, by status: Analysis sample: 54,243; Excluded: missing data: 728; Not urban (no SUHI value): 17,860.

**"Cities" means urbanized areas.** The SUHI database's unit is the Census Urbanized Area
(2010), a continuously built-up area named after its principal city or cities (for example
"Minneapolis--St. Paul, MN--WI"). Throughout, "cities" means UAs; leave-one-city-out
(section 4.1) holds out one UA at a time.
