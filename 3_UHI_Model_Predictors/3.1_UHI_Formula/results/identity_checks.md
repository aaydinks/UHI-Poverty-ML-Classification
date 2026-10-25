# Difference identities in the SUHI file

| group | target | identity | n | max_abs_error | mean_abs_error | share_exact_1e-6 | pearson_r | max_abs_error_if_rural_minus_urban | holds |
|---|---|---|---:|---|---|---:|---:|---|---|
| tract SUHI | UHI_annual_day | LST_urb_day_CT − LST_rur_day | 55,649 | 5.0e-08 | 3.2e-09 | 1.000000 | 1.000000 | 1.9e+01 | yes |
| tract SUHI | UHI_annual_night | LST_urb_night_CT − LST_rur_night | 55,649 | 5.0e-08 | 1.6e-09 | 1.000000 | 1.000000 | 1.8e+01 | yes |
| tract SUHI | UHI_summer_day | LST_urb_day_CT_summer − LST_rur_day_summer | 55,649 | 1.0e-08 | 3.2e-09 | 1.000000 | 1.000000 | 2.3e+01 | yes |
| tract SUHI | UHI_summer_night | LST_urb_night_CT_summer − LST_rur_night_summer | 55,649 | 4.0e-08 | 3.4e-09 | 1.000000 | 1.000000 | 1.3e+01 | yes |
| tract SUHI | UHI_winter_day | LST_urb_day_CT_winter − LST_rur_day_winter | 55,649 | 4.4e-07 | 1.9e-09 | 1.000000 | 1.000000 | 1.5e+01 | yes |
| tract SUHI | UHI_winter_night | LST_urb_night_CT_winter − LST_rur_night_winter | 55,649 | 4.9e-07 | 6.7e-10 | 1.000000 | 1.000000 | 2.2e+01 | yes |
| city SUHI | UHI_annual_day_city | LST_urb_day_all − LST_rur_day | 55,871 | 9.0e-09 | 3.3e-09 | 1.000000 | 1.000000 | 5.0e+00 | yes |
| city SUHI | UHI_annual_night_city | LST_urb_night_all − LST_rur_night | 55,871 | 9.0e-09 | 1.7e-09 | 1.000000 | 1.000000 | 1.8e+00 | yes |
| city SUHI | UHI_summer_day_city | LST_urb_day_all_summer − LST_rur_day_summer | 55,871 | 9.0e-09 | 3.1e-09 | 1.000000 | 1.000000 | 6.5e+00 | yes |
| city SUHI | UHI_summer_night_city | LST_urb_night_all_summer − LST_rur_night_summer | 55,871 | 1.0e-08 | 3.7e-09 | 1.000000 | 1.000000 | 2.0e+00 | yes |
| city SUHI | UHI_winter_day_city | LST_urb_day_all_winter − LST_rur_day_winter | 55,871 | 1.0e-08 | 2.1e-09 | 1.000000 | 1.000000 | 4.2e+00 | yes |
| city SUHI | UHI_winter_night_city | LST_urb_night_all_winter − LST_rur_night_winter | 55,871 | 8.0e-09 | 6.6e-10 | 1.000000 | 1.000000 | 2.2e+00 | yes |
| tract dNDVI | DelNDVI_annual | NDVI_urb_CT − NDVI_rur | 55,853 | 5.1e-08 | 3.1e-10 | 1.000000 | 1.000000 | 1.5e+00 | yes |
| city dNDVI | DelNDVI_annual_city | NDVI_urb_all − NDVI_rur | 55,871 | 1.0e-09 | 4.0e-10 | 1.000000 | 1.000000 | 2.7e-01 | yes |
| tract dNDVI | DelNDVI_summer | NDVI_urb_CT_summer − NDVI_rur_summer | 55,853 | 4.9e-08 | 2.5e-10 | 1.000000 | 1.000000 | 1.4e+00 | yes |
| city dNDVI | DelNDVI_summer_city | NDVI_urb_all_summer − NDVI_rur_summer | 55,871 | 1.0e-09 | 2.5e-10 | 1.000000 | 1.000000 | 2.5e-01 | yes |
| tract dNDVI | DelNDVI_winter | NDVI_urb_CT_winter − NDVI_rur_winter | 55,853 | 4.0e-07 | 3.0e-10 | 1.000000 | 1.000000 | 1.2e+00 | yes |
| city dNDVI | DelNDVI_winter_city | NDVI_urb_all_winter − NDVI_rur_winter | 55,871 | 2.7e-08 | 2.4e-10 | 1.000000 | 1.000000 | 2.7e-01 | yes |
| dDEM | DelDEM | DEM_urb_CT − DEM_rur | 55,854 | 9.8e-07 | 3.4e-08 | 1.000000 | 1.000000 | 1.6e+03 | yes |
| contrast: _CT_act | UHI_annual_day | LST_urb_day_CT_act − LST_rur_day | 55,649 | 1.3e+01 | 1.6e-01 | 0.724505 | 0.972994 | 2.0e+01 | no |
| contrast: _CT_act | UHI_annual_night | LST_urb_night_CT_act − LST_rur_night | 55,649 | 7.7e+00 | 6.2e-02 | 0.724541 | 0.958874 | 1.7e+01 | no |
| contrast: _CT_act | UHI_summer_day | LST_urb_day_CT_act_summer − LST_rur_day_summer | 55,649 | 1.2e+01 | 2.2e-01 | 0.724523 | 0.976423 | 2.3e+01 | no |
| contrast: _CT_act | UHI_summer_night | LST_urb_night_CT_act_summer − LST_rur_night_summer | 55,649 | 6.2e+00 | 6.3e-02 | 0.724595 | 0.972137 | 1.2e+01 | no |
| contrast: _CT_act | UHI_winter_day | LST_urb_day_CT_act_winter − LST_rur_day_winter | 55,649 | 9.0e+00 | 1.0e-01 | 0.724613 | 0.970031 | 1.8e+01 | no |
| contrast: _CT_act | UHI_winter_night | LST_urb_night_CT_act_winter − LST_rur_night_winter | 55,649 | 1.0e+01 | 7.0e-02 | 0.724613 | 0.962447 | 2.2e+01 | no |

Errors are in the variable's units (°C, NDVI, metres).
