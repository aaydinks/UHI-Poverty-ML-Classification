# 1.1 National focus

## Purpose

States the study area: which tracts, urbanized areas and states the national analysis covers.

## Method

Every analysis uses one national sample (`make build`), and `scope_summary.py` records its
extent in [`results/scope_summary.json`](results/scope_summary.json). Every join uses 2010
tract codes.

## Results

| | Count |
|---|---:|
| All 2010 census tracts, 50 states + DC | 73,056 |
| Tracts in the SUHI database, 50 states + DC | 54,998 |
| Urbanized areas (UAs) in the SUHI database, 50 states + DC | 486 |
| **Primary analysis sample** | **54,243 tracts in 486 UAs across 51 jurisdictions (50 states + DC)** |
| Share of all 2010 tracts in the 50 states + DC | 74.2 % |
| By Census region | Northeast 11,094, Midwest 11,349, South 18,652, West 13,148 |
| Robustness frame with HUD LPI | 54,237 |
