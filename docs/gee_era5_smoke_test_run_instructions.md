# Running the ERA5-Land GEE Smoke Test

## Prerequisites
- A Google Earth Engine (GEE) account with access to the project where you will store assets.
- The CSD boundary asset uploaded to GEE (must contain a `csd_uid` property).
- The `scripts/gee/era5_land_climate_smoke_test.js` file saved in this repository.

## Step-by-step instructions
1. **Open the Earth Engine Code Editor**
   - Go to https://code.earthengine.google.com/ and sign in with your GEE account.

2. **Create a new script**
   - Click **New > Script**.
   - Give the script a name, e.g., `era5_land_smoke_test`.

3. **Copy the template code**
   - Open the file `scripts/gee/era5_land_climate_smoke_test.js` in your local repository.
   - Copy the entire contents of the file.
   - Paste the code into the GEE Code Editor.

4. **Replace placeholder values**
   - `CSD_ASSET_ID`: set to the full asset path of your CSD boundary collection, e.g., `users/yourname/csd_boundaries`.
   - `TEST_CSD_UIDS`: replace the placeholder strings with the `csd_uid` values of 5-10 CSDs you want to test (keep them as quoted strings in the array).
   - `TEST_YEAR` and `TEST_MONTH`: set the year and month you wish to test (e.g., `2022` and `1` for January 2022).
   - `EXPORT_DESCRIPTION` and `EXPORT_FOLDER`: optionally change if you want a different description or Google Drive folder.
   - Verify the ERA5-Land collection ID. The template uses `ECMWF/ERA5_LAND/HOURLY`; confirm that this collection exists in GEE before running.
   - Verify band names (`temperature_2m`, `total_precipitation`, `snow_depth`). If a band is missing, comment out the corresponding section of the script.

5. **Run the script**
   - Click **Run** (or press Ctrl+Enter).
   - The console will print the number of test CSDs and the first feature to confirm that the CSD asset loaded correctly.
   - It will also print the first ERA5-Land image and its band names.

6. **Check the export task**
   - In the **Tasks** tab (right-hand side), you should see a task named `era5_land_climate_smoke_test`.
   - Click **Run** on the task. The default settings are fine for a smoke test.
   - The task will export a CSV to the Google Drive folder you specified.

7. **Verify the output**
   - Open your Google Drive and navigate to the folder `EXPORT_FOLDER` (default `hpai_gee_exports`).
   - Locate the CSV file; its name will contain the description, year and month, e.g., `era5_land_climate_smoke_test_2022_1.csv`.
   - Open the CSV (e.g., with Excel or pandas) and verify that:
     - There are rows equal to the number of test CSDs.
     - Columns include `csd_uid`, `year`, `month`, and the climate variables (`climate_tmean_c`, `climate_tmin_c`, `climate_tmax_c`, `climate_precip_mm`, `climate_snow_depth_cm`).
     - Values are within plausible ranges (e.g., temperature between -50 and 50, precipitation non-negative).

## Current Smoke-Test Finding
- **Boundary asset test**: Passed (the uploaded CSD asset loaded correctly, preserved `csd_uid`, and returned the expected CSD count).
- **Temperature smoke test**: Passed. For CSD 1001101, January 2022, ERA5-Land hourly temperature features looked plausible:
  - tmean: about -0.64 C
  - tmin: about -10.29 C
  - tmax: about 10.37 C
- **Precipitation aggregation**: Unresolved. Precipitation using simple hourly sum produced:
  - precip_sum_mm: about 3166.5 mm
  - A diagnostic showed: `precip_min_mm = 0`, `precip_mean_mm` about 4.26, `precip_max_mm` about 41.2, `precip_sum_mm` about 3166.5.
- **Conclusion**: Do not use simple sum of `total_precipitation` from the hourly ERA5-Land GEE collection for monthly precipitation until the accumulation semantics are verified.
- **Action / Recommendations**:
  1. Verify/use ERA5-Land monthly aggregated collection if available,
  2. Verify correct accumulated precipitation differencing method,
  3. Or exclude precipitation from V1 smoke extraction until corrected.
## What NOT to do
- Do not upload any private outbreak coordinate data to GEE or Google Drive.
- Do not modify any processed data files in the repository.
- Do not commit the exported CSV; it is a local QA artifact.
- Do not run a full national extraction until the smoke test is approved.

## Reporting results
- Summarize any errors or unexpected values you observe.
- If the smoke test succeeds, note the confirmation and proceed to the pilot extraction step outlined in `docs/gee_era5_smoke_test_plan.md`.
- If there are issues, record the error messages and adjust the placeholders or band names accordingly before re-running.

---
*This document is intended as a quick guide for running the ERA5-Land GEE smoke test. No code execution occurs on the local machine; all processing happens within Google Earth Engine.*

