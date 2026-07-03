# Climate Features Source Decision Document

## 1. Comparison of Candidate Data Sources

### 1.1 ERA5-Land
- Spatial resolution: 0.1 degree (about 9 km) grid (regular latitude-longitude).
- Temporal resolution: Hourly data; can be aggregated to monthly.
- Canada coverage: Global product; fully covers all Canadian territories.
- Variables available:
  - Mean temperature derived by averaging 2 m temperature over the month
  - Minimum temperature (t2m_min) - derived from hourly values
  - Maximum temperature (t2m_max) - derived from hourly values
  - Total precipitation (tp)
  - Snow depth (sd) and snowfall (sf) - separate variables
  - Freeze-thaw indicator (derived from temperature crossing 0 C)
- Ease of access: Available via the Copernicus Climate Data Store (CDS) API and also through Google Earth Engine; exact collection ID should be verified before implementation.
- Suitability for monthly CSD aggregation: Hourly data can be reduced to monthly statistics per CSD polygon using GEE reducers or local raster aggregation tools.
- Advantages:
  - Consistent global coverage and a single source for all variables.
  - High temporal density (hourly) enables accurate derivation of extremes and freeze-thaw days.
  - Direct GEE integration allows cloud-based processing without moving large files.
- Limitations:
  - Coarser spatial resolution (about 9 km) may miss fine-scale variation, especially in complex terrain.
  - Requires API key and registration for CDS download; GEE access may have usage quotas.
- Expected computational burden: Moderate to high if processing locally (hourly rasters for the whole country); low to moderate when using GEE reducers (cloud-based, parallel).
- Reproducibility: High - the same CDS version and GEE collection are versioned; scripts can be archived and re-run.
- Workflow suitability:
  - Compute Canada: Feasible but may require large storage for hourly files; best to use GEE to pre-aggregate and then download monthly summaries.
  - Google Earth Engine: Very suitable - one-step reducers over CSD polygons.

### 1.2 Daymet
- Spatial resolution: 1 km (approx 30 arc-seconds) grid, higher than ERA5-Land.
- Temporal resolution: Daily values; easily aggregated to monthly.
- Canada coverage: Covers North America (including all of Canada) but may have gaps at the extreme northern latitudes.
- Variables available:
  - Minimum temperature (tmin)
  - Maximum temperature (tmax)
  - Precipitation (prcp)
  - Shortwave radiation (srad) - not snowfall; Daymet provides snow water equivalent (swe) instead.
  - Vapor pressure (vp)
  - Day length (dayl)
  - Snow water equivalent (swe)
  - Mean temperature can be derived as (tmin + tmax) / 2
- Ease of access: Distributed via the NASA/ORNL DAAC; can be accessed via HTTP download, Python `daymet` package, or through Google Earth Engine; exact collection ID should be verified before implementation.
- Suitability for monthly CSD aggregation: Daily rasters can be summed or averaged by month over each CSD polygon with simple reducers.
- Advantages:
  - Finer spatial resolution (1 km) captures more local variation.
  - Daily data already matches a convenient temporal step for monthly aggregation.
  - Straightforward download; no registration needed for most public mirrors.
- Limitations:
  - Slightly narrower geographic extent; extreme northern CSDs may lack data.
  - Fewer climate variables than ERA5-Land for this use case, especially no direct snow depth or snowfall.
  - Requires handling of many daily files if processing locally; GEE reduces this burden.
- Expected computational burden: Moderate - daily rasters for a year are ~365 files per variable; GEE can aggregate efficiently.
- Reproducibility: High - same dataset version (e.g., Daymet v4) can be referenced; GEE assets are stable.
- Workflow suitability:
  - Compute Canada: Manageable; daily files can be streamed and aggregated on-the-fly.
  - Google Earth Engine: Suitable - GEE hosts the Daymet collection, enabling direct polygon reducers.

## 2. Recommended V1 Choice
**Recommendation: Use ERA5-Land as the primary source for V1 climate features.**

Why:
- ERA5-Land provides a single, globally consistent dataset that includes all required variables (including snow depth and snowfall) without needing additional proxies.
- The hourly temporal resolution simplifies the creation of derived metrics such as freeze-thaw days and snow-related flags.
- Direct access via GEE means we can perform the entire CSD-level monthly aggregation in the cloud, avoiding large downloads and storage concerns on Compute Canada.
- While the spatial resolution is coarser than Daymet, the V1 goal is simplicity and reproducibility; about 9 km resolution is sufficient for a national-scale risk model.
- Daymet can be retained as a fallback or sensitivity analysis source if higher spatial detail is later required.

## 3. Planned V1 Climate Features
| Feature name | Description | Units |
|--------------|-------------|-------|
| climate_tmean_c | Monthly mean temperature (average of 2 m temperature) | degrees C |
| climate_tmin_c | Monthly minimum temperature (minimum of 2 m temperature) | degrees C |
| climate_tmax_c | Monthly maximum temperature (maximum of 2 m temperature) | degrees C |
| climate_precip_mm | Total monthly precipitation (sum of precipitation) | mm |
| climate_snow_depth_cm *(if feasible)* | Monthly average snow depth | cm |
| climate_freeze_thaw_days *(if feasible)* | Number of days in the month where temperature crossed 0 C (freeze-thaw events) | days |

All feature names follow the `climate_` prefix and use SI units where applicable.

## 4. Validation Plan
- Temporal coverage: Features will be generated for every month that overlaps the CFIA outbreak label period (e.g., 2015-2024).
- Row completeness: Ensure exactly one row per CSD-month; missing rows will be flagged.
- No leakage: For predicting outbreak in month t+1, climate summaries should use data from month t or earlier only; future months are never referenced.
- Units documentation: Units are explicitly recorded in the feature table header and in this decision document.
- Missing data checks: Identify CSD-months with missing raster coverage; impute with NaN and log occurrences.
- Extreme value checks: Flag values outside plausible physical ranges (e.g., temperature < -50 C or > 50 C, precipitation < 0).
- CRS handling: CSD boundaries are maintained in the project CRS where appropriate and reprojected as needed for raster aggregation. CRS handling and area-weighting assumptions must be documented during implementation.
- Joinability: The final CSV will contain csd_uid, year, month columns, guaranteeing a direct join to the static node feature table and the monthly label table.

## 5. Proposed Output
Future output file:
```
data/processed/features/csd_monthly_climate_features.csv
```
Key columns:
- csd_uid
- year
- month
- climate_tmean_c
- climate_tmin_c
- climate_tmax_c
- climate_precip_mm
- climate_snow_depth_cm (optional)
- climate_freeze_thaw_days (optional)
