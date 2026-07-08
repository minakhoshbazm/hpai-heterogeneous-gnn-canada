# Running the ERA5-Land GEE Pilot Climate Extraction

## 1. Purpose
This document provides instructions for running the ERA5-Land monthly climate features pilot extraction script in Google Earth Engine (GEE). This is a pilot run for 10 representative Census Subdivisions (CSDs) across all 12 months of the year 2022. It is not the full national extraction.

---

## 2. Instructions for Running the Script

1. **Access the GEE Code Editor:**
   - Log in to your Google Earth Engine account at: https://code.earthengine.google.com/

2. **Load the CSD Boundary Asset:**
   - Ensure you have access to the CSD boundary asset: `projects/avianinfluenza/assets/canada_csd_2021_boundaries`.

3. **Copy and Paste the Script:**
   - Open the local file `scripts/gee/era5_land_climate_pilot_10csd_2022.js`.
   - Copy the entire contents of the script.
   - Paste the code into a new script in the GEE Code Editor.

4. **Execute the Script:**
   - Click **Run** (or press Ctrl+Enter).
   - In the **Console** tab on the right, verify the printed output:
     - Selected CSD count should be 10.
     - Monthly output row count (Expected: 120) should show 120.
     - Inspect the printed first 10 rows to verify that climate values are populated (or appropriately null/missing).

5. **Start GEE Task Export:**
   - Go to the **Tasks** tab on the right side of the Code Editor.
   - You should see a task named `era5_land_climate_pilot_10csd_2022`.
   - Click **Run** next to the task name.
   - Confirm the parameters in the pop-up window:
     - Drive folder: `hpai_gee_exports` (or your preferred folder).
     - File name prefix: `era5_land_climate_pilot_10csd_2022`.
     - File format: `CSV`.
   - Click **Run** in the pop-up window to begin the GEE task.
   - Monitor the task until it completes (showing a blue checkmark).

---

## 3. Expected Output and Validation

### Expected Row Count:
* Exactly **120 rows** (10 CSDs * 12 months).

### Expected Output Columns in the Exported CSV:
* `csd_uid`: CSD unique identifier (preserved as 7-character string with leading zeros).
* `csd_name`: Name of the CSD.
* `province_u`: Province unique code.
* `province_n`: Province name abbreviation.
* `year`: 2022.
* `month`: Calendar month (1 to 12).
* `climate_tmean_c`: Mean 2 m temperature (Celsius).
* `climate_tmin_c`: Minimum 2 m temperature (Celsius).
* `climate_tmax_c`: Maximum 2 m temperature (Celsius).
* `climate_precip_mm`: Total precipitation (Millimeters).
* `climate_missing_flag`: Missing value flag (1 if any climate value is null, 0 otherwise).

---

## 4. Known Issue: Small/Coastal/Island CSDs
* A known issue is that small, coastal, or island CSDs (such as Les Iles-de-la-Madeleine, `csd_uid` 2401023) may return missing (null) values for climate variables due to GEE grid masking or reducer scale boundaries.
* The script handles this by identifying these features and setting `climate_missing_flag = 1`. 
* Do not scale the pipeline to full national extraction until a fallback strategy (e.g., centroid buffers) is implemented and smoke-tested to address these null rows.

---

## 5. Local Post-Processing Checklist
After downloading the exported CSV from Google Drive:
1. **Drop Metadata:** Remove the `system:index` and `.geo` columns if present (GEE may export them even when geometry is set to null).
2. **Preserve leading zeros:** Ensure `csd_uid` is read as a string.
3. **Audit Missingness:** Verify that the rows where `climate_missing_flag = 1` correspond exactly to the expected coastal CSDs (e.g., `csd_uid` 2401023).
