# Running the GEE ERA5-Land Fallback Smoke Test

## 1. Purpose
This document provides instructions for running the GEE ERA5-Land fallback smoke test script. The goal is to compare five different spatial extraction methods (polygon mean, centroid point sample, and buffers of 25 km, 50 km, and 100 km) on a select subset of 11 CSDs (8 known missing coastal CSDs and 3 inland control CSDs) for January and July 2022.

---

## 2. Instructions for Running the GEE Script

1. **Open the GEE Code Editor:**
   - Go to https://code.earthengine.google.com/ and log in.

2. **Paste the Script:**
   - Copy the contents of the local file `scripts/gee/era5_fallback_smoke_test.js`.
   - Paste the code into a new script in the GEE Code Editor.

3. **Execute the Script:**
   - Click **Run** (or press Ctrl+Enter).
   - In the **Console** tab on the right, verify the printed metadata:
     - Selected CSD count should show `11`.
     - Expected row count should display `110`.
     - Actual output row count should display `110`.
     - Inspect the printed first 20 rows. Observe how different `method_name` entries change the `method_missing_flag` for coastal CSDs like Les Iles-de-la-Madeleine (`csd_uid` 2401023) or Burgeo (`csd_uid` 1003028).

4. **Start the GEE Task Export:**
   - Go to the **Tasks** tab on the right side of the Code Editor.
   - You should see a task named `era5_fallback_smoke_test`.
   - Click **Run** next to the task name.
   - Confirm the export parameters in the pop-up window:
     - Drive folder: `hpai_gee_exports` (or your preferred folder).
     - File name prefix: `era5_fallback_smoke_test`.
     - File format: `CSV`.
   - Click **Run** in the pop-up window to begin the GEE task.
   - Monitor the task until it completes (showing a blue checkmark).

---

## 3. Post-Export Actions

1. **Download the Exported CSV:**
   - Once the task has completed, navigate to your Google Drive folder (`hpai_gee_exports`).
   - Download the file `era5_fallback_smoke_test.csv`.

2. **Move to Project Workspace:**
   - Move the downloaded raw CSV directly into your local repository at:
     `data/raw/gee_exports/era5_fallback_smoke_test.csv`
   - **Do not edit the raw CSV file manually.**
   - **Do not interpret fallback choices or select a V1 fallback level yet.** 
   - A local Python QA script will be created to read this CSV, analyze the performance of each buffer size, and programmatically determine the optimal fallback settings.
