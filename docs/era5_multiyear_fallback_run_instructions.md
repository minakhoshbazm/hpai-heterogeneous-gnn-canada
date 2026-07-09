# ERA5-Land Multi-Year Fallback GEE Run Instructions

This document outlines the steps to run GEE extraction scripts for the years 2023, 2024, and 2025, using the validated hierarchical fallback cascade.

## 1. GEE Execution Steps

Run the following scripts separately in the Google Earth Engine Code Editor:
1. [era5_land_climate_national_2023_with_fallback.js](file:///d:/gdrive/models/hpai-heterogeneous-gnn-canada/scripts/gee/era5_land_climate_national_2023_with_fallback.js)
2. [era5_land_climate_national_2024_with_fallback.js](file:///d:/gdrive/models/hpai-heterogeneous-gnn-canada/scripts/gee/era5_land_climate_national_2024_with_fallback.js)
3. [era5_land_climate_national_2025_with_fallback.js](file:///d:/gdrive/models/hpai-heterogeneous-gnn-canada/scripts/gee/era5_land_climate_national_2025_with_fallback.js)

For each execution, verify the printed console diagnostics:
* **Total CSD count:** Should print `5161`.
* **Actual output row count:** Should print `61932`.
* **Monthly image counts:** Should verify exactly `1` image per month.
* **Method counts histogram:** Should show reasonable distributions (predominantly `polygon_mean`, with a small percentage of `buffer_25km` and `buffer_100km`).
* **First output feature properties:** Verify that all 13 properties are listed correctly.

---

## 2. Exporting and CSV Integration

1. In the **Tasks** tab, click **Run** for each year's task to launch the exports:
   - Task 2023: `era5_land_climate_national_2023_with_fallback`
   - Task 2024: `era5_land_climate_national_2024_with_fallback`
   - Task 2025: `era5_land_climate_national_2025_with_fallback`
2. Configure each task to export as a **CSV** into your Drive folder `hpai_gee_exports`.
3. Once completed, download and save the files to their respective workspace paths:
   - `data/raw/gee_exports/era5_land_climate_national_2023_with_fallback.csv`
   - `data/raw/gee_exports/era5_land_climate_national_2024_with_fallback.csv`
   - `data/raw/gee_exports/era5_land_climate_national_2025_with_fallback.csv`

---

## 3. Post-Processing & QA

To clean and validate the multi-year exports, you can reuse the post-processing script pattern (`scripts/postprocess_era5_2022_with_fallback.py`) on each downloaded file.
