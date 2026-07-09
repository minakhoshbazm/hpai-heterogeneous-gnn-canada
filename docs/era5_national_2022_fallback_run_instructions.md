# ERA5-Land National 2022 Fallback Run Instructions

This document provides instructions for running the GEE extraction script to retrieve monthly ERA5-Land climate features for all 5,161 Canadian Census Subdivisions (CSDs) in 2022, using the V1 fallback cascade.

## 1. How to Run in Google Earth Engine (GEE)

1. Open the [Google Earth Engine Code Editor](https://code.google.com/editor/).
2. Create a new script or copy the contents of [era5_land_climate_national_2022_with_fallback.js](file:///d:/gdrive/models/hpai-heterogeneous-gnn-canada/scripts/gee/era5_land_climate_national_2022_with_fallback.js) into the editor.
3. Click **Run** to execute the script in the console.
4. Verify the printed metrics in the Console tab:
   - **Total CSD count**: Should print `5161`.
   - **Monthly image count in 2022**: Should print `12`.
   - **Expected row count**: Should print `61932`.
   - **Actual output row count**: Should print `61932`.
5. Locate the **Tasks** tab in the right-hand panel of the GEE interface.
6. Find the task named `era5_land_climate_national_2022_with_fallback` and click **Run**.
7. In the configuration dialog:
   - Keep the Task Name as `era5_land_climate_national_2022_with_fallback`.
   - Select **Drive** as the destination.
   - Enter `hpai_gee_exports` (or your preferred folder name) as the Drive folder.
   - Set the File Format to **CSV**.
8. Click **Run** in the dialog to start the export process.

---

## 2. Expected Dataset Properties

* **Expected Row Count:** Exactly `61932` rows (5,161 CSDs * 12 months).
* **Expected Methods (`climate_extraction_method`):**
  - `polygon_mean` (for inland CSDs with valid polygon data)
  - `buffer_25km` (first fallback)
  - `buffer_50km` (second fallback)
  - `buffer_100km` (third fallback)
  - `missing` (if still unresolved after the 100 km buffer; sets `climate_missing_flag = 1`)

---

## 3. CSV Integration

1. Once the export completes in GEE, download the exported CSV file from your Google Drive folder (`hpai_gee_exports/era5_land_climate_national_2022_with_fallback.csv`).
2. Move/save this CSV to the following workspace path:
   `data/raw/gee_exports/era5_land_climate_national_2022_with_fallback.csv`

---

## 4. Post-Processing & QA

A dedicated local python post-processing and validation script will be created to run QA checks on the exported CSV file after download. These checks will verify:
- The exact total row count (`61,932`).
- The presence of all expected fallback methods.
- Value range consistency (e.g. non-negative precipitation, tmin <= tmean <= tmax).
- Re-calculation of missing flags and matching fallback parameters.
