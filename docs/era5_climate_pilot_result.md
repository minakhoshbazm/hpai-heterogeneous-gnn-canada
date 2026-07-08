# ERA5-Land Monthly Climate Pilot Extraction Results

## 1. Purpose of the Pilot
This pilot was conducted to validate the monthly climate extraction pipeline using Google Earth Engine (GEE). The objectives were to verify the monthly loops, ensure proper unit conversions, drop unnecessary geometry data, handle missing values via a diagnostic flag, and identify potential schema or encoding issues prior to scaling up.

---

## 2. Pilot Scope
* **Spatial Units:** 10 representative Canadian Census Subdivisions (CSDs):
  - 1001101, 1101002, 1201001, 1301001, 2401023, 3501005, 4601035, 4701001, 4801003, 5929028
* **Temporal Scope:** All 12 months of the year 2022.
* **Total Expected Records:** 120 CSD-month rows (10 CSDs * 12 months).

---

## 3. Data Source
* **Collection ID:** `ECMWF/ERA5_LAND/MONTHLY_AGGR` in Google Earth Engine.
* **Resolution:** Approximately 9 km (0.1 degree).

---

## 4. Extraction Variables
The following properties were selected for export and conversion:
* `csd_uid` (String identifier)
* `csd_name` (Name of the CSD)
* `province_u` (Province code)
* `province_n` (Province name)
* `year` (2022)
* `month` (1-12)
* `climate_tmean_c` (Celsius, converted from Kelvin)
* `climate_tmin_c` (Celsius, converted from Kelvin)
* `climate_tmax_c` (Celsius, converted from Kelvin)
* `climate_precip_mm` (Millimeters, converted from meters)
* `climate_missing_flag` (1 if any variable was null, 0 otherwise)

---

## 5. QA Result Summary
The exported CSV output was evaluated against the pilot QA criteria:
* **Record Count:** Exactly 120 rows.
* **CSD Coverage:** 10 unique CSDs across months 1-12.
* **Duplicates:** 0 duplicate CSD-month rows.
* **Completeness:** 
  - 108 complete rows (with fully populated climate features).
  - 12 missing rows (containing null climate variables).
  - **Location of Missingness:** All 12 missing rows belong to CSD `2401023` (Les Iles-de-la-Madeleine).
* **Precipitation Check:** No negative values; precip_mm values are positive or zero.
* **Temperature Check:** Order is physically valid across all populated rows: `climate_tmin_c` <= `climate_tmean_c` <= `climate_tmax_c`.
* **Encoding Artifacts:** CSD names containing accents (e.g., Les Iles-de-la-Madeleine) exhibited encoding artifacts in the exported CSV. Downstream joins must strictly rely on the unique `csd_uid` property instead of `csd_name`.

---

## 6. Interpretation and Key Findings
1. **Extraction Pipeline Validity:** The monthly aggregation pipeline is computationally sound and successfully extracts valid climate statistics for inland CSDs.
2. **Missingness in Small/Coastal/Island Regions:** The missingness encountered in Les Iles-de-la-Madeleine (`csd_uid` 2401023) confirms that water-dominated or small land-area CSDs fail polygon mean reductions due to ERA5-Land grid masking. 
3. **Data Integrity:** Unique numeric and string identifiers (`csd_uid`) remain stable, while names are subject to character encoding corruption. Using `csd_uid` as the primary join key is mandatory.
4. **Quantification of Missingness:** The percentage of CSDs with missing values must be quantified nationally before final extraction to scale the impact of fallback mechanisms.

---

## 7. Next Steps and Decisions
1. **Implement Local Post-Processing Script:** Develop a Python script to clean the raw exported CSV, drop GEE system columns (`system:index`, `.geo`), enforce string formatting for `csd_uid` (preserving leading zeros), and audit `climate_missing_flag`.
2. **Expand the Pilot Scope:** Before executing the full national multi-year extraction, run a larger pilot (e.g., one complete province or one full year nationally) to evaluate processing time, final file size, and the proportion of missing coastal cells.
3. **Design Fallback Handling:** Develop and test the ordered centroid-buffer fallback strategy (such as a 100 km centroid buffer mean) to handle small/coastal CSDs where the polygon mean returns nulls.
