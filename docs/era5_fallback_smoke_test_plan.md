# Plan: GEE ERA5-Land Fallback Smoke Test

## 1. Purpose of the Smoke Test
The national 2022 pilot extraction identified 320 CSDs that returned missing values across all 12 months. This smoke test is designed to evaluate candidate fallback methods in Google Earth Engine (GEE) on a small subset of these missing CSDs to determine which fallback radius is the smallest sufficient method to resolve data, and to compare values against non-missing controls.

---

## 2. Test Scope and Selection
The smoke test will process the following features:

### Test CSDs (Known Missing):
1. **St. Shott's** (`csd_uid` 1001120)
2. **Southern Harbour** (`csd_uid` 1001267)
3. **Arnold's Cove** (`csd_uid` 1001270)
4. **Bryant's Cove** (`csd_uid` 1001375)
5. **Wabana** (`csd_uid` 1001494)
6. **Burgeo** (`csd_uid` 1003028)
7. **Ramea** (`csd_uid` 1003026)
8. **Les Iles-de-la-Madeleine** (`csd_uid` 2401023)

### Control CSDs (Known Complete):
1. **CSD 1001101** (Newfoundland control CSD)
2. **CSD 3501005** (Ontario control CSD)
3. **CSD 4801003** (Alberta control CSD)

### Test Time Periods:
* **January 2022** (Winter control)
* **July 2022** (Summer control)

---

## 3. Fallback Methods to Compare
For each selected CSD and target month, the GEE script will compute and compare five separate extraction methods:
1. **Direct Polygon Mean:** Spatial mean reduction using the original CSD boundary.
2. **Centroid Point Sample:** Point sample extracted at the CSD centroid.
3. **25 km Centroid Buffer Mean:** Spatial mean reduction over a 25 km radius buffer around the centroid.
4. **50 km Centroid Buffer Mean:** Spatial mean reduction over a 50 km radius buffer around the centroid.
5. **100 km Centroid Buffer Mean:** Spatial mean reduction over a 100 km radius buffer around the centroid.

*Note: Last-resort fallback levels (Nearest Valid CSD and Parent CD average) will not be modeled during this initial GEE buffer comparison test.*

---

## 4. Expected Output Schema
The GEE export task will generate a flat table (CSV) with one row per CSD-month-method combination (11 CSDs * 2 months * 5 methods = 110 expected rows). 

Columns must include:
* `csd_uid`: String CSD identifier.
* `csd_name`: Name of the CSD.
* `province_u`: Province code.
* `year`: 2022.
* `month`: month index (1 or 7).
* `method_name`: String name of the method (e.g., `"polygon_mean"`, `"centroid"`, `"buffer_25km"`, `"buffer_50km"`, `"buffer_100km"`).
* `climate_tmean_c`: Converted mean temperature (Celsius).
* `climate_tmin_c`: Converted min temperature (Celsius).
* `climate_tmax_c`: Converted max temperature (Celsius).
* `climate_precip_mm`: Converted precipitation (Millimeters).
* `method_missing_flag`: Integer (1 if the values are missing/null for this specific method, 0 otherwise).

---

## 5. Validation Questions
During local post-processing, we will analyze the output CSV to answer:
1. **Resolution Threshold:** Which method is the smallest-radius fallback that successfully resolves each of the 8 missing CSDs?
2. **Buffer Sufficiency:** Are the 25 km or 50 km buffers sufficient to resolve the majority of missing coastal CSDs, or is 100 km required?
3. **Over-Smoothing Risk:** Does the 100 km buffer produce suspiciously identical values for CSDs in the same region, indicating a loss of spatial variance?
4. **Control Divergence:** For the 3 control CSDs, how much do fallback values (centroid, buffers) diverge from their true polygon means?
5. **Selection Criteria:** Which buffer size strikes the best balance between resolution success rate and local spatial representative value?

---

## 6. Decision Rule
* The pipeline should select the **smallest-radius method** that successfully resolves all climate variables.
* The columns `climate_extraction_method` and `climate_fallback_distance_km` must be populated during final post-processing so downstream GNN models can dynamically adjust or distinguish fallback-derived features.
* The next step is to create the GEE smoke test script implementing this comparison before running the fallback extraction.
