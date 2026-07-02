# CSD Structural Features Plan

This document summarizes the accepted structural feature foundation for the HPAI heterogeneous graph neural network (GNN) model in Canada.

---

## 1. Primary Node Scale
* **Census Subdivisions (CSDs):** Act as the primary graph nodes (~5,161 subdivisions).
* **Census Divisions (CDs):** Remain strictly as fallback or aggregation scales only; they are not used as the primary modeling scale.

## 2. Existing Label Foundation
* **Spatial Mapping:** Canadian Food Inspection Agency (CFIA) poultry outbreak events have been successfully mapped to CSDs using exact CFIA coordinate points.
* **CSD-Month Aggregation:** CSD-month labels serve as the current official target label foundation.
* **Current Status:** All downstream modeling remains paused until structural and environmental feature layers are complete.

## 3. Road Network Features
* **NRN Data Coverage:** National Road Network (NRN) ZIP files are downloaded, verified, and successfully processed for all outbreak-affected provinces (BC, AB, SK, MB, ON, QC, NB, NS, PEI, NL). Territory road ZIPs (Yukon (60), NWT (61), Nunavut (62)) were not processed. Territory CSDs are present in the output, but their road lengths/densities are set to `0.0` and centroid distances are `NaN`.
* **Feature Extraction Methodology:**
  * **Lengths:** Computed by intersecting (splitting) the `ROADSEG` line geometries with CSD polygons in the projected Statistics Canada coordinate system **`EPSG:3347`**.
  * **Distances:** Shortest distance from the CSD centroid to the nearest major road or highway segment in the province, computed using spatial indexing in **`EPSG:3347`**.
  * **Area Sensitivity Flag:** Very small CSDs (e.g., land area < 1.0 km²) can produce extremely high road-density ratios. These are flagged with `small_area_flag` so they can be considered in sensitivity analysis.
* **Output Schema (`csd_road_features.csv`):**
  * `csd_uid` (Primary Key, e.g., `"3520005"`)
  * `csd_area_km2` (CSD land area in km² computed from `EPSG:3347`)
  * `all_road_length_km` (Total length in km of all roads in the CSD)
  * `all_road_density_km_per_km2` (Ratio of total road length to CSD area)
  * `major_road_length_km` (Total length in km of major roads)
  * `major_road_density_km_per_km2` (Ratio of major road length to CSD area)
  * `highway_length_km` (Total length in km of highways)
  * `highway_density_km_per_km2` (Ratio of highway road length to CSD area)
  * `distance_to_nearest_major_road_km` (Shortest distance from CSD centroid to the nearest major road segment in the province)
  * `distance_to_nearest_highway_km` (Shortest distance from CSD centroid to the nearest highway segment in the province)
  * `major_road_proportion` (Ratio of major road length to total road length)
  * `highway_proportion` (Ratio of highway length to total road length)
  * `road_data_coverage` (QA Column: `"processed"` for target provinces, `"not_processed_territory"` for Yukon, NWT, Nunavut)
  * `small_area_flag` (QA Column: boolean, `true` if `csd_area_km2 < 1.0` else `false`)

## 4. Poultry Exposure (Census of Agriculture)
* **StatCan Table:** Statistics Canada Table 32-10-0375-01 has been downloaded and inspected.
* **Resolution:** The dataset provides Census Consolidated Subdivision (CCS) level poultry production and farms-reporting proxy values rather than exact high-resolution inventory counts or density.
* **Mapping Rule:** CCS-level values will be mapped to their member CSDs using `ccs_` prefixes.
* **No Down-Scaling:** CCS-level values must not be area-weighted down to member CSD boundaries to avoid introducing spatial errors.
* **Pending Verification:** The exact semantic meanings of StatCan `E` and `F` `STATUS` suppression codes remain pending clarification.

## 5. CSD-to-CCS Mapping Lookup
* **Lookup Table:** A lookup file is located at `data/processed/lookups/csd_to_ccs_lookup.csv`.
* **Dimension:** It maps all 5,161 CSDs to their corresponding 1,757 parent CCSs.
* **Application:** This lookup table will be used to resolve and attach the parent-CCS agricultural poultry proxies directly to the corresponding CSD nodes.

## 6. Global Livestock Environmental Atlas (GLW) Features
* **GLW Chicken (2020):** Aligned with 2020 FAOSTAT, downloaded and verified at `data/raw/glw/GLW4-2020.D-DA.CHK.tif`.
* **GLW Duck (2020):** Availability is unconfirmed; the GLW 2015 duck density raster (`5_Dk_2015_Da.tif`) is established as the fallback source.
* **GLW Turkey (2020):** Unconfirmed and not included in V1.
* **Spatial Resolution & Scale:** GLW data is distributed at a coarse global resolution of 5 arc-minutes (~10 km at the equator). 
* **Extraction Fallbacks:** Because the raster scale is coarse, small or coastal CSD polygons may not overlap any valid grid cell centers. For these CSDs, the extraction pipeline falls back to **centroid sampling** or **zero fallback** (if the centroid is also ocean/nodata).
* **QA Flags:** The output dataset includes QA columns indicating the extraction method used (`zonal`, `centroid_fallback`, or `zero_fallback`) and the number of valid pixels intersected (`valid_pixel_count`). These QA flags must be analyzed during downstream GNN modeling to assess sensitivity to boundary misalignment.
* **Chicken 2015 Baseline:** Due to unit discrepancies (which result in significantly larger raw values for GLW 2015 chicken counts compared to GLW 2020 density), GLW 2015 chicken stats are excluded from the V1 modeling dataset and saved in a separate sensitivity file (`csd_glw_chicken_2015_sensitivity.csv`).
* **Feature Schema:**
  * **V1 Table (`csd_glw_poultry_density.csv`):** `glw2020_chicken_density_mean`, `glw2020_chicken_density_max`, `glw2020_chicken_density_p95`, `glw2015_duck_density_mean`, `glw2015_duck_density_max`, `glw2015_duck_density_p95` (birds/km²), plus their respective `_extraction_method` and `_valid_pixel_count` QA columns.
  * **Sensitivity Table (`csd_glw_chicken_2015_sensitivity.csv`):** `glw2015_chicken_mean`, `glw2015_chicken_max`, `glw2015_chicken_p95` (raw count scale), plus its `_extraction_method` and `_valid_pixel_count` QA columns.

## 7. Data Privacy and Repository Integrity
* **No Commits:** Do not commit raw CFIA data, exact coordinates, private coordinate-based mapping quality assurance charts, model weight checkpoints, or large raw datasets.
* **Git Configuration:** Raw datasets remain explicitly ignored via `.gitignore`.
