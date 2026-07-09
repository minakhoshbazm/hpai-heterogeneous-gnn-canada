# Dynamic World National 2022 and Province Pilot Extraction GEE Run Instructions

This document provides instructions for running GEE scripts to extract monthly Dynamic World land cover probabilities and candidate seasonal pond proxy metrics at the CSD level.

## 1. Ontario 2022 Province Pilot Run

To mitigate timeout issues, a province-specific pilot script is available. It extracts data for Ontario CSDs (province_u = "35") at 30m scale with optimized tile scaling.

### Steps:
1. Open the [Google Earth Engine Code Editor](https://code.google.com/editor/).
2. Copy the contents of [dynamic_world_2022_province_pilot.js](file:///d:/gdrive/models/hpai-heterogeneous-gnn-canada/scripts/gee/dynamic_world_2022_province_pilot.js) into the editor.
3. Click **Run** in the toolbar.
4. Verify the console outputs:
   - **Selected province**: Ontario (province_u = "35")
   - **CSD count for Ontario**: Verify CSD size
   - **Expected row count text**: CSD count * 12
   - **Year**: 2022
5. Under the **Tasks** tab, click **Run** on `dynamic_world_ontario_2022_pilot`.
6. Configure the task to export to **Drive**, inside the `hpai_gee_exports` folder as a **CSV**.
7. Click **Run** to start the job.
8. Once completed, download the CSV and save it to:
   `data/raw/gee_exports/dynamic_world_ontario_2022_pilot.csv`

---

## 2. National 2022 Extraction (Single Job)

Instructions for the full national extraction (5,161 CSDs, 61,932 expected rows):

### Steps:
1. Copy the contents of [dynamic_world_national_2022_export_only.js](file:///d:/gdrive/models/hpai-heterogeneous-gnn-canada/scripts/gee/dynamic_world_national_2022_export_only.js) into the Code Editor.
2. Click **Run** and verify the printed diagnostics.
3. In the **Tasks** tab, run `dynamic_world_national_2022`.
4. Export as **CSV** to your Google Drive `hpai_gee_exports` folder.
5. Once completed, download and save the CSV to:
   `data/raw/gee_exports/dynamic_world_national_2022.csv`
