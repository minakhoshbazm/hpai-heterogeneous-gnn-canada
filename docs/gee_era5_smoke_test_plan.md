# GEE ERA5-Land Smoke Test Plan

## 1. Purpose
Before running the full national monthly climate extraction we will execute a small Google Earth Engine (GEE) smoke test. The test confirms that we can:
- Identify the correct ERA5-Land collection ID.
- Read the CSD boundary asset with the required `csd_uid` property.
- Access the needed climate bands (2 m temperature, precipitation, optional snow depth).
- Apply the correct unit conversions (Kelvin to degrees C, mm for precipitation, cm for snow depth).
- Use the appropriate reducers for monthly summarization.
- Export the results in a CSV format that matches the planned schema.
- Preserve the join keys (`csd_uid`, `year`, `month`).

## 2. Inputs Needed
- A CSD boundary file (GeoJSON or Shapefile) that includes a string property `csd_uid`.
- A small subset of 5-10 CSDs selected from different provinces to exercise geometry handling.
- One test month for winter (e.g., January 2022) and optionally a spring month (e.g., April 2022) to verify temperature and precipitation patterns.
- Access to a Google Earth Engine account with the project set to the repository's GEE project.

## 3. Boundary Preparation
- Keep the `csd_uid` attribute as a string; do not convert to integer or drop leading zeros.
- Simplify geometries only if necessary for performance, but log the simplification tolerance used.
- Do not upload any private outbreak coordinate data; only public CSD outlines are uploaded.
- Record the CRS of the uploaded asset (usually EPSG:4326) and any re-projection steps.

## 4. ERA5-Land Collection Verification
- Before any extraction, verify the exact GEE collection ID for ERA5-Land (e.g., `ECMWF/ERA5_LAND/HOURLY`).
- List available bands and note the units:
  - `temperature_2m` (Kelvin)
  - `total_precipitation` (meters per hour or accumulated per hour)
  - `snow_depth` (meters, if present)
- Confirm the temporal resolution (hourly) and how to aggregate to monthly values.
- Do not assume the collection ID; query the GEE catalog in a short script to print it.

## 5. Smoke Test Design
1. Load the CSD boundary asset and filter to the selected test CSDs.
2. For the chosen test month:
   - Filter the ERA5-Land collection to that month.
   - Aggregate hourly images to obtain:
     - climate_tmean_c: temporal monthly mean of hourly 2 m temperature, then spatial mean over CSD.
     - climate_tmin_c: temporal monthly minimum of hourly 2 m temperature, then spatial mean over CSD.
     - climate_tmax_c: temporal monthly maximum of hourly 2 m temperature, then spatial mean over CSD.
     - climate_precip_mm: temporal monthly total precipitation, then spatial mean over CSD unless an area-weighted total is explicitly intended.
     - Optional climate_snow_depth_cm: temporal monthly mean of hourly snow depth, then spatial mean over CSD, converted to centimeters.
   - Reduce each aggregated image over the CSD polygons using `ee.Reducer.mean()` (or appropriate reducers for the temporal summaries). For precipitation use a spatial sum only if an area-weighted total is required; otherwise keep the mean.
   - Export the per-CSD table to Google Drive (or Cloud Storage) as a CSV.
3. Expected output path:
   `results/smoke_tests/era5_land_climate_smoke_test.csv`

Smoke-test outputs under results/smoke_tests/ are local QA artifacts and should not be committed unless explicitly approved.

## 6. QA Checks for Smoke Test
- Row count exactly matches the number of test CSDs.
- All `csd_uid` values are present and remain strings.
- Temperature columns (`climate_tmean_c`, `climate_tmin_c`, `climate_tmax_c`) fall within plausible limits (e.g., -50 C to 50 C).
- Precipitation column (`climate_precip_mm`) is non-negative.
- If snow depth is included, values are non-negative.
- No missing values unless explicitly documented.
- `year` and `month` columns correctly reflect the test month.
- The CSV can be read locally with pandas and matches the schema defined for the full extraction.

## 7. Decision After Smoke Test
- **If the smoke test passes**: proceed to a pilot extraction for a single province or a single year, then scale to the full national monthly extraction.
- **If the smoke test fails**: identify the failure point (collection ID, band name, unit conversion, geometry, export method) and fix it before attempting a larger pilot.

## 8. Safety Rules
- Do not upload any raw or private outbreak coordinate data to GEE.
- Do not run any model training or full extraction until the smoke test is approved.
- Do not modify or commit processed data files.
- Only scripts, documentation, and the smoke-test plan should be added to version control.
- All steps must be logged so that the smoke test can be reproduced or audited.
