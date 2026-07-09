# Dynamic World CSD-Level Monthly Extraction Smoke Test Instructions

This document provides instructions for running the GEE smoke test script to extract monthly Dynamic World land cover features and candidate seasonal pond proxy metrics for a subset of Canadian CSDs.

## 1. How to Run in Google Earth Engine (GEE)

1. Open the [Google Earth Engine Code Editor](https://code.google.com/editor/).
2. Copy the contents of the [dynamic_world_smoke_test.js](file:///d:/gdrive/models/hpai-heterogeneous-gnn-canada/scripts/gee/dynamic_world_smoke_test.js) script and paste it into the editor.
3. Click **Run** in the toolbar.
4. Verify the following printed diagnostics in the Console panel:
   - **Selected CSD count**: Should print `6`.
   - **Expected row count**: Should print `18` (6 CSDs * 3 months).
   - **Actual output row count**: Verify it prints `18`.
   - Examine the **First output feature** properties to ensure all expected metrics (e.g. `dw_water_mean`, `dw_seasonal_pond_proxy`, etc.) are computed.
5. In the **Tasks** tab in the right-hand panel, find the task named `dynamic_world_smoke_test` and click **Run**.
6. In the dialog, set:
   - Destination to **Drive**.
   - Folder to `hpai_gee_exports`.
   - File format to **CSV**.
7. Click **Run** to start the export.

---

## 2. Integration and Download

1. Once the GEE task finishes running, download `dynamic_world_smoke_test.csv` from your Google Drive.
2. Save/move the CSV file to the following workspace path:
   `data/raw/gee_exports/dynamic_world_smoke_test.csv`
3. A follow-up post-processing QA validation script will be run locally to verify value ranges, columns, and data properties.
