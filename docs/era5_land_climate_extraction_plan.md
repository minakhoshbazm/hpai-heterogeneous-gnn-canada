# ERA5-Land Climate Extraction Implementation Plan

## 1. Purpose
This plan describes how to extract monthly CSD-level climate features from ERA5-Land for the HPAI project. No extraction has been run yet; the document only prepares the workflow.

## 2. Temporal Scope
- Use the CFIA label period as the required date range. The script should read the label table and detect the minimum and maximum dates automatically before extraction.
- Temporal unit is **year** and **month**.
- Prediction framing: features from month t are used to predict outbreak risk in month t+1.
- No future-month leakage: only data from month t or earlier may be used for the prediction of month t+1.

## 3. Spatial Scope
- All 5,161 Census Subdivisions (CSDs) are included.
- CSD boundaries are the spatial aggregation units.
- CSD identifiers (`csd_uid`) must be preserved as strings with leading zeros.
- CRS handling:
  - CSD boundaries may be stored in the project CRS.
  - Geometries must be reprojected or exported correctly for Earth Engine raster aggregation.
  - All CRS handling steps should be logged.

## 4. ERA5-Land Variables (V1)
- **climate_tmean_c**: Monthly mean 2 m temperature, converted from Kelvin to degrees C.
- **climate_tmin_c**: Monthly minimum 2 m temperature, converted from Kelvin to degrees C.
- **climate_tmax_c**: Monthly maximum 2 m temperature, converted from Kelvin to degrees C.
- **climate_precip_mm**: Monthly total precipitation, converted to millimeters. Status: Resolved. Use `ECMWF/ERA5_LAND/MONTHLY_AGGR` (`total_precipitation_sum`) for V1 climate extraction.
- *Optional if feasible*:
  - **climate_snow_depth_cm**: Monthly mean snow depth, converted to centimetres.
  - **climate_freeze_thaw_days**: Number of days in the month where daily minimum temperature is below 0 C and daily maximum temperature is above 0 C.

## 5. Recommended Workflow (Google Earth Engine)
1. Prepare or upload a CSD boundary asset that includes the `csd_uid` property.
2. For each month in the label period:
   - Load the pre-aggregated monthly collection: `ECMWF/ERA5_LAND/MONTHLY_AGGR`.
   - Filter the collection to the target month.
   - For each variable, select the corresponding monthly band:
     - temperature mean (`temperature_2m`), minimum (`temperature_2m_min`), and maximum (`temperature_2m_max`)
     - precipitation total (`total_precipitation_sum`)
     - optional snow depth mean from monthly snow depth if available
   - Then spatially summarize the resulting monthly image over each CSD, usually using a spatial mean. Do not spatially sum precipitation across pixels unless a physically area-weighted total is explicitly intended.
   - Export the per-CSD results as a table (CSV) for that month.
3. After all months are exported, combine the monthly CSV files into a single table using a simple concatenation script (Python or shell).
4. Perform local validation of the combined CSV (row counts, ranges, missing values).

All units and temporal-versus-spatial aggregation choices must be documented in the extraction script.

*Note*: Compute Canada is intended for downstream model training and analysis, not for the initial raw ERA5 hourly processing which is handled in Earth Engine.

## 6. Proposed Output
Future output file:
```
data/processed/features/csd_monthly_climate_features.csv
```
**Required columns**:
- csd_uid
- year
- month
- climate_tmean_c
- climate_tmin_c
- climate_tmax_c
- climate_precip_mm
- climate_snow_depth_cm (optional)
- climate_freeze_thaw_days (optional)

## 7. QA and Validation Checks
- Row count equals 5,161 CSDs x number of months.
- No duplicate `csd_uid`/`year`/`month` rows.
- All `csd_uid` values are strings (preserve leading zeros).
- No missing climate values unless explicitly documented.
- Temperature values are within plausible physical limits (e.g., -50 C to 50 C).
- Precipitation values are non-negative and within plausible monthly ranges. Implausibly large values should stop the extraction workflow.
- Snow depth values are non-negative.
- Monthly coverage matches the label period.
- Ensure that climate data from month t+1 is never used when predicting month t+1.
- The final CSV must join cleanly to the static node feature table and the monthly label table.

## 8. Risks and Decisions Needed Before Extraction
- Collection Selection: Resolved. Use `ECMWF/ERA5_LAND/MONTHLY_AGGR` for temperature and precipitation V1 extraction.
- Planned Bands:
  - `temperature_2m`
  - `temperature_2m_min`
  - `temperature_2m_max`
  - `total_precipitation_sum`
- Decide whether to include the optional snow depth variable in V1.
- Decide whether freeze-thaw days are part of V1 or a later V1.1 extension.
- Choose between exporting one CSV per month versus a single combined table.
- Verify Earth Engine export quota and strategy (e.g., Drive vs. Cloud Storage).

## 9. Recommended V1 Scope
- Use monthly pre-aggregated data (`MONTHLY_AGGR`) for both temperature (mean, min, max) and precipitation (sum).
- Add snow depth if it can be derived without excessive complexity.
- Defer freeze-thaw day calculation to V1.1 if hourly/daily aggregation proves difficult.

## 10. Smoke Test Findings & Action Plan
- **Temperature Smoke Test**: Passed. For CSD 1001101, January 2022, ERA5-Land hourly temperature features looked plausible:
  - tmean: about -0.64 C
  - tmin: about -10.29 C
  - tmax: about 10.37 C
- **Precipitation Aggregation**: Resolved via comparison test (CSD 1001101, January 2022):
  - Method A (Monthly `total_precipitation_sum`): 254.55 mm
  - Method B (Hourly `total_precipitation_hourly` monthly sum): 254.55 mm
  - Method C (Incorrect Hourly `total_precipitation` simple sum): 3156.53 mm (anomalous double-counting bug)
- **10-CSD Monthly Aggregate Smoke Test**:
  - A 10-CSD smoke test using `ECMWF/ERA5_LAND/MONTHLY_AGGR` ran successfully, producing exactly 10 rows.
  - Climate fields (`climate_tmean_c`, `climate_tmin_c`, `climate_tmax_c`, `climate_precip_mm`) were present.
  - The GEE CSV output still included `.geo` with empty MultiPoint geometry and `system:index`, which should be dropped during local post-processing.
  - One coastal/island CSD, Les Iles-de-la-Madeleine (`csd_uid` 2401023), returned missing climate values.
  - This indicates the full pipeline needs a missing-value fallback strategy for small/coastal/island CSDs, likely centroid sampling or buffered geometry fallback.
- **Conclusion**:
  - Use `ECMWF/ERA5_LAND/MONTHLY_AGGR` for V1 climate extraction.
  - Do not use raw hourly `total_precipitation` simple sum.
  - `total_precipitation_hourly` is valid but more computationally expensive; keep as fallback or validation method.
  - Do not scale to full national extraction until the coastal/island missing-value fallback handling is designed and smoke-tested.
