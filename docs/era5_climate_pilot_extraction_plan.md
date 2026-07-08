# Implementation Plan: ERA5-Land Monthly Climate Pilot Extraction

## 1. Pilot Scope and Objectives
The purpose of this pilot extraction is to validate the ERA5-Land monthly aggregation pipeline at a small scale before launching the full national extraction. This pilot will help identify pipeline bottlenecks, verify the monthly iteration loop in Google Earth Engine (GEE), and measure feature missingness in coastal regions.

### Scope:
* **Spatial Units:** 10 representative Census Subdivisions (CSDs):
  - 1001101, 1101002, 1201001, 1301001, 2401023, 3501005, 4601035, 4701001, 4801003, 5929028
  These CSDs span multiple provinces and include coastal/island geometries (e.g., csd_uid 2401023).
* **Temporal Period:** All months of the calendar year 2022 (12 months).
* **Data Source:** GEE collection `ECMWF/ERA5_LAND/MONTHLY_AGGR`.
* **Climate Variables:**
  - `climate_tmean_c` (derived from `temperature_2m`)
  - `climate_tmin_c` (derived from `temperature_2m_min`)
  - `climate_tmax_c` (derived from `temperature_2m_max`)
  - `climate_precip_mm` (derived from `total_precipitation_sum`)

---

## 2. Expected Output Schema
The final cleaned CSV output of the pilot will be saved at:
`data/processed/features/pilot_csd_monthly_climate_features.csv`

The file must contain exactly 120 rows (10 CSDs * 12 months) and have the following columns:
* `csd_uid` (String, preserving leading zeros)
* `year` (Integer)
* `month` (Integer)
* `climate_tmean_c` (Float, Celsius)
* `climate_tmin_c` (Float, Celsius)
* `climate_tmax_c` (Float, Celsius)
* `climate_precip_mm` (Float, Millimeters)
* `climate_missing_flag` (Integer, 0 or 1)

*Note: Earth Engine metadata columns (`system:index`, `.geo`) must be dropped during local post-processing.*

---

## 3. GEE Export Strategy
To prevent out-of-memory errors and manage GEE limits:
* **GEE Task Export:** The pilot script will iterate through the target months of 2022 and map the spatial reduction over the 10 CSDs.
* **Geometry Discarding:** Features will have their geometry explicitly set to null before export using `ee.Feature(null, feature.toDictionary([...]))`.
* **Export Destination:** Google Drive as a single CSV file containing all 12 months of pilot data.
* **Comparison for Scaling:** For the full national extraction, we will decide based on the pilot runtime whether to export one combined file, one file per year, or one file per month.

---

## 4. Local Post-Processing Plan
Once the exported CSV is downloaded locally:
* **Cleaning Script:** A lightweight Python script will read the raw CSV.
* **Drop EE Metadata:** Remove `system:index` and `.geo` columns.
* **Leading Zeros:** Ensure `csd_uid` is treated as a string to preserve leading zeros.
* **Missing Value Documentation:** If any CSD-month records contain null values, they will be logged and `climate_missing_flag` will be set to 1. No fallback values will be filled during this pilot.
* **Note on Fallback Handling:** We will not run fallback centroid/buffer aggregation in GEE during the pilot. Missing values will only be quantified and recorded. The actual fallback logic will be implemented and smoke-tested before scaling to the full national extraction. The pilot is allowed to have missing values; missingness is an expected diagnostic output, not an automatic failure.

---

## 5. QA Checks
The post-processed pilot CSV must satisfy the following checks:
1. **Row Count:** Exactly 120 rows.
2. **Column Constraints:** No `system:index` or `.geo` columns.
3. **Leading Zeros:** Check that Newfoundland and Labrador CSDs (starting with 10) have 7-character string identifiers.
4. **Physical Ranges:**
   - Temperature values: -50 C to 45 C
   - Precipitation values: Non-negative and less than 1000 mm (if present)
5. **Completeness Check:** Report the count and percentage of missing values (especially for CSD 2401023).

---

## 6. Runtime and Logging Notes
* Log the execution duration of the GEE task.
* Estimate total export size and scale it linearly to calculate the burden for the full 5,161 CSDs over the entire label period (approx. 2015-2024).
* Log any GEE timeout or API exceptions.

---

## 7. Decision Criteria for Scaling
We will proceed to the full national extraction under these conditions:
1. The GEE loop executes without timeouts.
2. The schema joins cleanly with the static features table.
3. Feature values are within physically plausible bounds (ignoring missing fields).
4. Missing values are successfully quantified and recorded.
5. The post-pilot plan for small/coastal CSD fallback handling (an ordered fallback strategy for small/coastal CSDs) is designed, implemented in the GEE scripts, and smoke-tested.
