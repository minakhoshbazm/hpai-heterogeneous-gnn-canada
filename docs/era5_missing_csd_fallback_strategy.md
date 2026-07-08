# Plan: ERA5 Climate Missing-CSD Fallback Strategy

## 1. Background
The national 2022 ERA5-Land monthly climate pilot extraction completed successfully for all 5,161 Canadian Census Subdivisions (CSDs) across all 12 months. However, the pilot QA validation revealed a substantial subset of missing data:
* **Total Expected Records:** 61,932 rows (5,161 CSDs * 12 months).
* **Complete Records:** 58,092 rows.
* **Missing Records:** 3,840 rows (containing null values for tmean, tmin, tmax, and precip).
* **Missing CSDs:** Exactly **320 CSDs** were missing data for all 12 months of the year.

Spatial analysis of these 320 CSDs shows that missingness is concentrated in small, coastal, island, or remote regions (such as St. Shott's, Southern Harbour, Arnold's Cove, Wabana, and Burgeo). This is likely related to the interaction between small/coastal/island geometries, ERA5-Land grid placement, masking, and reducer scale behavior; the exact mechanism should be confirmed during fallback smoke tests.

---

## 2. Why Fallback is Needed
For downstream heterogeneous graph neural network (GNN) modeling and outbreak forecasting, leaving complete CSD temporal histories as missing is highly problematic:
* **GNN Node Feature Continuity:** Graph neural networks require fully populated node features across all space-time slices to perform spatial convolutions and message passing.
* **Predictive Performance:** Dropping or leaving 320 CSD nodes unpopulated removes important agricultural and remote regions from model inference.
* **Scientific Transparency:** While filling these gaps is necessary, the fallback values must be explicitly flagged rather than silently imputed. The GNN model and researchers must be able to distinguish between observations derived from direct polygon mean reduction versus fallback approximations.

---

## 3. Comparison of Candidate Fallback Methods

| Method | Advantages | Limitations | Scientific Risk |
| :--- | :--- | :--- | :--- |
| **A. Direct Polygon Mean** | Uses actual CSD boundaries; captures local geography. | Fails completely for small, narrow, or island CSDs. | None; this is the primary choice. |
| **B. Centroid Point Sample** | Zero-width extraction; fast to compute. | Centroid point may still fall in water or be masked out. | High if the centroid is on a boundary or water pixel. |
| **C. Centroid Buffer Mean (25 km)** | Smallest buffer; close proximity to CSD centroid. | May still return missing values for offshore/isolated islands. | Low local variation; representative of local climate. |
| **D. Centroid Buffer Mean (50 km)** | Medium buffer; covers larger coastal bays. | May pull in pixels from adjacent landmasses with different microclimates. | Moderate; could blur local topography (e.g., coastal vs inland). |
| **E. Centroid Buffer Mean (100 km)** | High success rate; successfully resolved island CSDs (e.g., CSD 2401023). | Large radius blurs local geographical variance. | High; can over-smooth temperature and precipitation. |
| **F. Nearest Valid Neighboring CSD** | Leverages valid CSD-level polygon values. | Neighbor may be far away or structurally different (e.g., mainland vs island). | Blurs boundary distinctions; assumes uniform spatial correlation. |
| **G. Parent Census Division (CD) Average** | Conceptually clean hierarchical group average. | CDs in northern/remote Canada are massive; blurs local variations. | High; CD averages can include wildly different climate zones. |

---

## 4. Recommended V1 Strategy
We recommend an ordered cascade strategy implemented in the extraction pipeline. For each CSD-month:
1. **Attempt Level 1: Direct Polygon Mean**
   - Perform standard spatial reduction over CSD boundary.
2. **Attempt Level 2: Centroid Point Sample**
   - If Level 1 is missing, extract climate values at the CSD centroid.
3. **Attempt Level 3: 25 km Centroid Buffer Mean**
   - If Level 2 is missing, generate a 25 km buffer around the centroid and compute the mean.
4. **Attempt Level 4: 50 km Centroid Buffer Mean**
   - If Level 3 is missing, generate a 50 km buffer around the centroid and compute the mean.
5. **Attempt Level 5: 100 km Centroid Buffer Mean**
   - If Level 4 is missing, generate a 100 km buffer around the centroid and compute the mean.
6. **Attempt Level 6: Nearest Valid CSD Climate Value**
   - If 100 km buffer still fails, use the nearest CSD with valid polygon-mean climate values for that month.
   - Record the source CSD UID in `climate_fallback_source_uid`.

7. **Attempt Level 7: Parent Census Division Average (Last-Resort)**
   - If nearest valid CSD is not acceptable or unavailable, assign the average value of valid CSDs within the same parent Census Division.
   - This is the last-resort fallback because CDs can be large and climatically heterogeneous.

---

## 5. Required Output Tracking Columns
The cleaned climate feature CSV must include the following metadata tracking columns:
* `climate_tmean_c`: Mean temperature (Celsius).
* `climate_tmin_c`: Minimum temperature (Celsius).
* `climate_tmax_c`: Maximum temperature (Celsius).
* `climate_precip_mm`: Precipitation (Millimeters).
* `climate_missing_flag`: Integer (0 if data was resolved, 1 if all fallback attempts failed and data remains missing).
* `climate_extraction_method`: String indicating the successful resolution level:
  - `"polygon_mean"`
  - `"centroid_sample"`
  - `"centroid_buffer_25km"`
  - `"centroid_buffer_50km"`
  - `"centroid_buffer_100km"`
  - `"nearest_valid_csd"`
  - `"parent_cd_average"`
  - `"missing"` (all failed)
* `climate_fallback_distance_km`: Numeric distance (0.0 for polygon mean/centroid, 25.0, 50.0, 100.0, or parent distance approximation).
* `climate_fallback_source_uid`: String CSD or CD identifier indicating the source of fallback values (e.g., the CSD itself, the nearest CSD UID, the parent CD UID, or `"none"`).

---

## 6. Validation Requirements
After fallback extraction is implemented, the post-processing script must enforce:
1. **Zero Unflagged Fallbacks:** Any row with fallback values must have `climate_extraction_method != "polygon_mean"`.
2. **Frequency Counts:** Print the total number of records resolved by each extraction method.
3. **Missing CSD Auditing:** Print the list of CSDs that fell back to Level 5 (100 km), Level 6 (Nearest Valid CSD), or Level 7 (Parent CD average).
4. **Comparison Check:** Compare fallback values against nearby valid polygon means to ensure no massive discontinuities.
5. **Extreme Bounds:** Check that fallback values do not introduce physically impossible values (e.g., tmax < tmin, or negative precipitation).
6. **Visual Mapping:** Export coordinates of fallback CSDs to verify that they are situated along coasts or islands.

---

## 7. Decision Before Implementation
Fallback implementation choices significantly affect downstream scientific interpretation and spatial correlations within the GNN.
* **This plan must be reviewed and approved by the core team before any coding or GEE script modifications are implemented.**
* Scaling to the full multi-year national extraction is blocked until the fallback GEE logic is designed, peer-reviewed, and smoke-tested.
