# Implementation Plan: ERA5-Land National One-Year Climate Pilot

## 1. Purpose of the National Pilot
Before launching the full multi-year, national-scale extraction of ERA5-Land monthly climate features, we will conduct a one-year national pilot for all 5,161 Census Subdivisions (CSDs) across Canada.

This national pilot is required to:
* Verify that Google Earth Engine (GEE) can successfully handle the spatial reduction of all 5,161 CSD polygons on a monthly iteration loop without memory timeouts.
* Estimate GEE processing runtimes and export file sizes to plan the full multi-year workflow.
* Identify and compile a complete list of all coastal, island, and remote CSDs that experience grid-masking nulls.
* Quantify national missingness as missing CSD-month records to establish baseline requirements for a future fallback strategy.
* Validate the schema, data types, and join compatibility at a national scale.

---

## 2. Scope and Source
* **Spatial Units:** All 5,161 Census Subdivisions (CSDs) in the uploaded GEE asset:
  `projects/avianinfluenza/assets/canada_csd_2021_boundaries`
* **Temporal Period:** All months of the calendar year 2022 (months 1-12).
* **Expected Records:** Exactly **61,932 rows** (5,161 CSDs * 12 months).
* **Source Dataset:** `ECMWF/ERA5_LAND/MONTHLY_AGGR` in Google Earth Engine.

---

## 3. Target Variables
The script will select and aggregate the following monthly bands:
* `climate_tmean_c` (derived from `temperature_2m`, converted from Kelvin to Celsius)
* `climate_tmin_c` (derived from `temperature_2m_min`, converted from Kelvin to Celsius)
* `climate_tmax_c` (derived from `temperature_2m_max`, converted from Kelvin to Celsius)
* `climate_precip_mm` (derived from `total_precipitation_sum`, converted from meters to millimeters)
* `climate_missing_flag` (1 if any of the above variables are null, 0 otherwise)

---

## 4. Pipeline Workflow
1. **GEE Script Execution:**
   - Filter the boundary asset to load all 5,161 CSD features.
   - Run the monthly loop over the 12 months of 2022.
   - Reduce the converted climate bands using `ee.Reducer.mean()`.
   - Use the `missingAsNumber` helper function to flag nulls.
   - Drop feature geometry using `ee.Feature(null, dictionary)`.
   - Export the collection to Google Drive as a single CSV file named `era5_land_climate_pilot_2022_national`.
2. **Local Post-Processing:**
   - Move the raw exported CSV to the local directory:
     `data/raw/gee_exports/`
   - Execute a Python post-processing script to clean, validate, and drop GEE system columns (`system:index` and `.geo`).
   - Write the finalized CSV to the destination:
     `data/processed/features/pilot_csd_monthly_climate_features_2022_national.csv`

*Note: No fallback values should be filled during this national pilot. Fallback design and implementation will proceed only after national missingness has been quantified and audited.*

---

## 5. Local Validation and QA Checks
The post-processing script must enforce the following validation rules:
1. **Row Count:** Must equal exactly 61,932.
2. **Unique CSDs:** Must contain exactly 5,161 unique `csd_uid` values.
3. **Temporal Coverage:** Months must span 1 to 12.
4. **Duplicates:** 0 duplicate `csd_uid`/`year`/`month` rows.
5. **Schema Compliance:** Output columns must match the required V1 feature schema.
6. **Flag Consistency:** `climate_missing_flag` must equal 1 if and only if any of the climate values are missing (null).
7. **Precipitation check:** All non-null `climate_precip_mm` values must be non-negative.
8. **Temperature order:** All non-null values must satisfy `climate_tmin_c` <= `climate_tmean_c` <= `climate_tmax_c`.
9. **Missing CSDs Log:** Print the list of all unique `csd_uid` codes with missing values to identify coastal boundaries.
