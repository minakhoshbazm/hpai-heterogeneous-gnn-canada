# Decision Note: ERA5-Land Climate Fallback Method Selection

## 1. Fallback Smoke-Test Summary
To select the optimal fallback strategy for small, coastal, and island CSDs with missing ERA5-Land climate values, we conducted a GEE fallback smoke test.
* **Scope:** 11 CSDs (8 known missing CSDs, 3 controls).
* **Temporal Period:** January and July 2022.
* **Tested Methods:** Polygon mean, centroid sample, 25 km centroid buffer mean, 50 km centroid buffer mean, and 100 km centroid buffer mean.
* **Total Records:** 110 expected rows.
* **Status:** The run completed successfully, and all local validation and QA checks passed.

---

## 2. Resolution Results
The counts of CSD-month combinations resolved at each extraction tier:
* **polygon_mean:** 6 CSD-months (3 controls * 2 months)
* **centroid_sample:** 0 CSD-months
* **buffer_25km:** 14 CSD-months (7 missing CSDs * 2 months)
* **buffer_50km:** 0 CSD-months
* **buffer_100km:** 2 CSD-months (1 missing CSD, Les Iles-de-la-Madeleine `csd_uid` 2401023, * 2 months)
* **unresolved:** 0 CSD-months

---

## 3. Interpretation and Key Findings
1. **Control Consistency:** For the inland controls, the polygon mean matches the centroid and 25 km buffer mean with modest deviations (e.g. less than 0.1 degrees C temperature difference). This indicates that control deviations were modest in this small smoke test.
2. **Buffer Efficacy:** The 25 km buffer successfully resolved 7 out of the 8 missing coastal CSDs. This indicates that a small expansion of the geometry is sufficient to overlap with land-masked ERA5-Land cells in most coastal cases.
3. **Island Constraints:** Les Iles-de-la-Madeleine (`csd_uid` 2401023) was the only CSD that required the 100 km buffer; 100 km was the first tested buffer distance that resolved this CSD in the smoke test.
4. **Resolution Rate:** 100% of the tested CSDs were resolved at or before the 100 km buffer level. No CSD-month remained unresolved.
5. **Smoothing Risk:** Larger buffers (50 km and 100 km) exhibit larger deviations from control polygon means due to spatial over-smoothing. Therefore, the smallest buffer that successfully resolves a feature should be prioritized.

---

## 4. V1 Fallback Decision
We recommend implementing a hierarchical, ordered cascade in the final climate extraction pipeline:
1. **polygon_mean:** (Default) Aggregates climate values over the CSD boundary.
2. **buffer_25km:** (First fallback) Used if the polygon mean is null.
3. **buffer_50km:** (Second fallback) Used if the 25 km buffer is null.
4. **buffer_100km:** (Third fallback) Used if the 50 km buffer is null.
5. **unresolved:** (missing flag = 1) Assigned if the 100 km buffer is still null.

### Specific Exclusions and Future Scope:
* **centroid_sample:** Will not be included in the V1 pipeline. It did not resolve any additional CSDs compared to the 25 km buffer and presents a higher risk of pointing to water-masked grid cells.
* **Nearest CSD and Parent CD Average:** These remain reserved as last-resort fallback options for future pipeline iterations (V1.1+) if subsequent multi-year runs reveal CSDs that are unresolved by the 100 km buffer. They are not part of the default V1 fallback cascade.

---

## 5. Required Tracking Columns
The final climate feature table must include these three columns to track fallback provenance:
* `climate_extraction_method`: String indicating the source method (e.g., "polygon_mean", "buffer_25km", "buffer_50km", "buffer_100km", or "missing").
* `climate_fallback_distance_km`: Float indicating the buffer distance (0.0, 25.0, 50.0, or 100.0).
* `climate_missing_flag`: Integer (1 if the record remained unresolved, 0 otherwise).

---

## 6. Next Implementation Step
The next step is to create and validate a national 2022 climate extraction script that implements this V1 fallback cascade, validates output row counts and column schemas, and verifies that the 320 missing CSDs are successfully resolved.
