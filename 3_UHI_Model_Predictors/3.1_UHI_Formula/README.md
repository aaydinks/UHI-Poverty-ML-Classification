# 3.1 The exact UHI formula

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
12.77 °C, while the `_CT` difference matches it to
5e-08 °C.

## Method

`verify_formula.py` recomputes every difference variable in the file from its urban and rural
parts and records the maximum absolute error, the mean absolute error and Pearson r
([`results/identity_checks.md`](results/identity_checks.md)).

## Results

| Group | Identities | Hold (error < 1e-6) | Largest error | Smallest r |
|---|---:|---:|---:|---:|
| tract SUHI | 6 | 6 | 4.9e-07 | 1.000000 |
| city SUHI | 6 | 6 | 1.0e-08 | 1.000000 |
| tract dNDVI | 3 | 3 | 4.0e-07 | 1.000000 |
| city dNDVI | 3 | 3 | 2.7e-08 | 1.000000 |
| dDEM | 1 | 1 | 9.8e-07 | 1.000000 |

19 of 19 identities hold to floating-point precision over all
55,649 tracts with a value. In particular `UHI_annual_day` equals
`LST_urb_day_CT − LST_rur_day` with a maximum error of 5.0e-08 °C
(r = 1.000000).

Every `UHI_*` column is therefore an urban-minus-rural LST difference, and every `_city`
column is a UA-level constant built the same way. A model of `High_UHI` that sees any `LST_*`
or `UHI_*` column can reconstruct, or closely approximate, its own target, so section 3.2
removes both families from the UHI model.
