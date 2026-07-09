# Dynamic World Province-Year Extraction GEE Run Instructions

This document provides instructions for running the reusable GEE script to extract monthly Dynamic World land cover features at the CSD level for a specific province and year.

## 1. GEE Script Usage

The script [`dynamic_world_province_year_export.js`](file:///d:/gdrive/models/hpai-heterogeneous-gnn-canada/scripts/gee/dynamic_world_province_year_export.js) is designed to run on a single province and year at a time to prevent Google Earth Engine execution timeouts.

### Steps:
1. Open the [Google Earth Engine Code Editor](https://code.google.com/editor/).
2. Copy the contents of [`dynamic_world_province_year_export.js`](file:///d:/gdrive/models/hpai-heterogeneous-gnn-canada/scripts/gee/dynamic_world_province_year_export.js) and paste it into the editor.
3. Modify the **Parameters** block at the top of the script as desired. For example:
   ```javascript
   var PROVINCE_UID = '35'; // Ontario
   var YEAR = 2022;
   var EXPORT_DESCRIPTION = 'dynamic_world_province_35_2022';
   ```
4. Click **Run** in the toolbar.
5. Verify the console outputs:
   - **PROVINCE_UID**: Displays the current target province ID.
   - **YEAR**: Displays the current target year.
   - **Selected CSD count**: Count of CSDs found in this province.
   - **Expected row count text**: Selected CSD count * 12.
   - **Export description**: Displays the GEE export task name.
6. Navigate to the **Tasks** tab in the right-hand panel, click **Run** for the task, and configure the export to **Drive**, folder `hpai_gee_exports`, as a **CSV**.
7. Once the task finishes, download the resulting CSV and place it in the following workspace directory:
   `data/raw/gee_exports/dynamic_world_province_<PROVINCE_UID>_<YEAR>.csv`

---

## 2. Extraction Roadmap

Ontario 2022 (`province_u = "35"`, `YEAR = 2022`) has already been run and validated as a pilot. Run the remaining provinces and years (2022-2025) as needed:
* **Ontario (35):** 2023, 2024, 2025
* **Quebec (24):** 2022, 2023, 2024, 2025
* **Other Provinces/Territories:** Run for years 2022-2025 using their respective numeric `province_u` codes (e.g., Alberta = '48', British Columbia = '59', Manitoba = '46', etc.).
