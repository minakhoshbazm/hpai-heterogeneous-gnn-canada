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
* **NRN Data Coverage:** National Road Network (NRN) ZIP files are downloaded and verified for all outbreak-affected provinces:
  * British Columbia (BC)
  * Alberta (AB)
  * Ontario (ON)
  * Quebec (QC)
  * Saskatchewan (SK)
  * Manitoba (MB)
  * Nova Scotia (NS)
  * Newfoundland & Labrador (NL)
  * New Brunswick (NB)
  * Prince Edward Island (PEI)
* **Layer Availability:** The `ROADSEG` layer exists in each province's dataset.
* **Attribute Schema:** The key columns `ROADCLASS` and `TRAFFICDIR` are available.
* **Feature Extraction:** Road density and network features will be extracted geometrically using the projected Statistics Canada coordinate system **`EPSG:3347`** at a later stage.

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
* **GLW Chicken (2020):** Source has been officially confirmed on the FAO site; however, the actual download and metadata verification remain pending.
* **GLW Duck (2020):** Availability is unconfirmed; the GLW 2015 duck density raster is established as the fallback source.
* **GLW Turkey (2020):** Availability is unconfirmed; turkeys will not be included in the V1 model unless an official, reliable source is found.
* **Role:** GLW density rasters will ultimately provide continuous, CSD-level density-like poultry exposure features.

## 7. Data Privacy and Repository Integrity
* **No Commits:** Do not commit raw CFIA data, exact coordinates, private coordinate-based mapping quality assurance charts, model weight checkpoints, or large raw datasets.
* **Git Configuration:** Raw datasets remain explicitly ignored via `.gitignore`.
