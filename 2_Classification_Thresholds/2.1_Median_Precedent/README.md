# 2.1 Median split and its precedent

## Purpose

Sets one cut-off for each outcome, gives its rationale, and reports the class counts.

## Method

Both outcomes are split once, at the national median of the primary analysis sample, in
`src/uhi_poverty/labels.py`; every model and the chi-squared test (section 2.2) import the same
labels. `medians.py` records the cut values and class counts in
[`results/median_values.json`](results/median_values.json), with the 66th and 75th percentiles
as sensitivity cut-points.

## Results

Primary analysis sample, n = 54,243 tracts
([`results/class_counts.md`](results/class_counts.md)):

| Label | Rule | Cut | n | Positive | Negative |
|---|---|---:|---:|---:|---:|
| `High_UHI` | `UHI_annual_day` >= median | 1.4476 | 54,243 | 27,122 | 27,121 |
| `High_Pov` | `pov_rate` >= median | 0.1050 | 54,243 | 27,122 | 27,121 |
| `High_Pov_LPI` | `LPI` < median (low index = poorer) | 55.0000 | 54,237 | 26,916 | 27,321 |

The high-poverty median is a tract poverty rate of 10.5 %. The
LPI label is the robustness outcome; because a low index means more poverty, its positive class
is the tracts *below* the LPI median.

Sensitivity cut-points (positives / negatives):

| Percentile | UHI cut | High_UHI | Poverty-rate cut | High_Pov |
|---|---:|---:|---:|---:|
| 50th | 1.448 °C | 27,122 / 27,121 | 10.5 % | 27,122 / 27,121 |
| 66th | 2.314 °C | 18,443 / 35,800 | 15.6 % | 18,443 / 35,800 |
| 75th | 2.843 °C | 13,561 / 40,682 | 19.7 % | 13,561 / 40,682 |

## Precedent

Three studies of urban heat split a heat measure at, or close to, its median, two of them
alongside a median-type split of a poverty or income measure, as here:

1. **Madrigano J, Ito K, Johnson S, Kinney PL, Matte T.** A case-only study of vulnerability
   to heat wave–related mortality in New York City (2000–2011). *Environ Health Perspect*
   2015;123(7):672–678. doi:10.1289/ehp.1408178.
   Census tracts in New York City, with daytime summer surface temperature from a Landsat
   thermal scene. Every neighbourhood modifier, surface temperature and the share of
   households on public assistance among them, is dichotomised at its median (high = at or
   above the median); the 75th percentile is a sensitivity cut. This is the closest design:
   the same unit (tracts), satellite surface temperature, and a median split of both heat
   and a poverty measure. It differs in using the citywide median of tract surface
   temperature rather than the national median of urban-minus-rural SUHI.
2. **Goggins WB, Chan EYY, Ng E, Ren C, Chen L.** Effect modification of the association
   between short-term meteorological factors and mortality by urban heat islands in Hong
   Kong. *PLoS ONE* 2012;7(6):e38551. doi:10.1371/journal.pone.0038551.
   An urban heat island index for 248 planning units is split at its Hong Kong-wide median
   (3.67) into "hot" and "cool" areas and crossed with a split of median household income
   that divides deaths roughly in half, a 2×2 design of high/low heat island by high/low
   socioeconomic status like the one in section 2.2. The index is modelled from urban form
   (sky-view factor, greenery, wind), not satellite LST.
3. **Smargiassi A, Goldberg MS, Plante C, Fournier M, Baudouin Y, Kosatsky T.** Variation of
   daily warm season mortality as a function of micro-urban heat islands. *J Epidemiol
   Community Health* 2009;63(8):659–664. doi:10.1136/jech.2008.078147.
   Montreal deaths assigned the satellite surface temperature at the residence, split into
   two strata at the 50th, 66th and 75th percentiles (the main analysis used the 75th), with
   dwelling value split at the 25th, 33rd and 50th percentiles.

**How it maps here.** Surface UHI intensity has no external standard comparable to a clinical
threshold, so a distribution-based cut-point is the defensible choice, as in these studies,
and the median is the cut they use for both heat and socioeconomic status. A median split
gives two classes of equal size for both outcomes, which keeps accuracy-type metrics
interpretable and lets the same cut serve the classifiers and the association test.
Following the same practice, the 66th and 75th percentiles are reported as sensitivity
cut-points (section 2.2).
