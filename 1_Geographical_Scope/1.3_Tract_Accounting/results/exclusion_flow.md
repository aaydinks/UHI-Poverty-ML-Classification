# Exclusion flow

| step | rule | n_tracts | n_removed | note |
|---|---|---:|---:|---|
| 0 | All 2010 tracts, 50 states + DC (2019 Gazetteer; coverage universe) | 73,056 | – | Puerto Rico adds 945 tracts |
| 1 | In the SUHI database, 50 states + DC (urbanized-area tracts only) | 54,998 | 18,067.000 | Gazetteer tracts with no SUHI value: not urban (source limitation); 9 SUHI codes are absent from the 2019 Gazetteer |
| 2 | 50 states + DC (Puerto Rico held out as the sensitivity frame) | 54,998 | 873.000 | 873 Puerto Rico SUHI tracts moved to the sensitivity frame |
| 3 | Complete SUHI target and features | 54,761 | 237.000 | missing SUHI fields |
| 3g | …with Gazetteer geometry (water_frac, log_aland) | 54,752 | 9.000 | not in the 2019 Gazetteer |
| 4 | Matched to ACS 2015–2019 with poverty universe >= 50 | 54,243 | 509.000 | ACS unmatched or small universe |
| 5 | Primary analysis sample | 54,243 | – | – |
| 5a | …with HUD LPI present (robustness frame) | 54,237 | 6.000 | no LPI value |
| 5b | …plus Puerto Rico (sensitivity frame) | 55,095 | – | +852 Puerto Rico tracts |

n_removed = tracts dropped between the previous step and this one.
