# Running the ERA5-Land GEE National 2022 Pilot Extraction

## 1. Purpose
This document provides instructions for running the ERA5-Land monthly climate features national pilot extraction script in Google Earth Engine (GEE). This pilot processes all 5,161 Canadian Census Subdivisions (CSDs) across all 12 months of the year 2022.

---

## 2. Instructions for Running the Script

1. **Access the GEE Code Editor:**
   - Log in to your Google Earth Engine account at: https://code.earthengine.google.com/

2. **Load the Boundary Asset:**
   - Verify that the CSD boundary asset is uploaded and accessible at:
     `projects/avianinfluenza/assets/canada_csd_2021_boundaries`

3. **Copy and Paste the Script:**
   - Open the local file `scripts/gee/era5_land_climate_national_2022.js`.
   - Copy the entire contents of the script.
   - Paste the code into a new script in the GEE Code Editor.

4. **Execute the Script:**
   - Click **Run** (or press Ctrl+Enter).
   - In the **Console** tab on the right, verify the printed output:
     - Total CSD count (Expected: 5161) should display `5161`.
     - Expected row count should display `61932`.
     - Actual output row count should show `61932`.
     - Inspect the printed first 10 rows in the console to verify that the fields are structured correctly.

5. **Start GEE Task Export:**
   - Go to the **Tasks** tab on the right side of the Code Editor.
   - You should see a task named `era5_land_climate_national_2022`.
   - Click **Run** next to the task name.
   - Confirm the export parameters in the pop-up window:
     - Drive folder: `hpai_gee_exports` (or your preferred folder).
     - File name prefix: `era5_land_climate_national_2022`.
     - File format: `CSV`.
   - Click **Run** in the pop-up window to begin the GEE task.
   - Monitor the task until it completes (showing a blue checkmark).

---

## 3. Post-Export Actions

1. **Download the Exported CSV:**
   - Once the task has completed, navigate to your Google Drive folder (`hpai_gee_exports`).
   - Download the file `era5_land_climate_national_2022.csv`.

2. **Move to Project Workspace:**
   - Move the downloaded raw CSV directly into your local repository at:
     `data/raw/gee_exports/era5_land_climate_national_2022.csv`
   - **Do not edit the raw CSV file manually.** Any manual intervention can corrupt string identifiers (such as stripping leading zeros from `csd_uid`) or break character encoding.

3. **Next Steps:**
   - Local post-processing and QA validation scripts will be created next to process this raw export.
