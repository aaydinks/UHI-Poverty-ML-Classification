# Share of validation tracts with a nearest neighbour in training

| task | scheme | scheme_label | n_validation | share_with_train_neighbour | mean_train_neighbours |
|---|---|---|---:|---:|---:|
| A | random5 | Random 5-fold (reference) | 54,243 | 1.000 | 6.398 |
| A | county5 | County-grouped 5-fold | 54,243 | 0.153 | 0.410 |
| A | ua5 | UA-grouped 5-fold | 54,243 | 0.036 | 0.090 |
| A | loco | Leave-one-city-out | 54,243 | 0.045 | 0.114 |
| A | state | Leave-one-state-out (UA-buffered) | 54,243 | 0.001 | 0.002 |
| A | region | Leave-one-region-out (UA-buffered) | 54,243 | 0.000 | 0.000 |
| A | major_ua | Each of the 20 largest UAs held out | 23,777 | 0.020 | 0.058 |
| B | random5 | Random 5-fold (reference) | 54,243 | 1.000 | 6.396 |
| B | county5 | County-grouped 5-fold | 54,243 | 0.156 | 0.412 |
| B | ua5 | UA-grouped 5-fold | 54,243 | 0.037 | 0.093 |
| B | loco | Leave-one-city-out | 54,243 | 0.045 | 0.114 |
| B | state | Leave-one-state-out (UA-buffered) | 54,243 | 0.001 | 0.002 |
| B | region | Leave-one-region-out (UA-buffered) | 54,243 | 0.000 | 0.000 |
| B | major_ua | Each of the 20 largest UAs held out | 23,777 | 0.020 | 0.058 |
