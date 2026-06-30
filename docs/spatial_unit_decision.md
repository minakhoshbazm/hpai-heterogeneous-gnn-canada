# Spatial Unit Decision: HPAI Canada Modeling

Choosing the appropriate spatial unit of analysis involves balancing epidemiological relevance, spatial resolution of covariates, and label availability. The table below compares the candidates considered for this project.

## Spatial Unit Comparison

| Spatial Unit | Approximate Size / Count | Spatial Resolution | Data Availability | Scientific / Practical Value |
| :--- | :--- | :--- | :--- | :--- |
| **Province / Territory** | 13 units | Coarse | Extremely high (CFIA public reporting is at least province-level) | Low. Provinces are too large to identify regional biosecurity or local migratory stopover risks. |
| **Census Agricultural Region (CAR)** | ~80 units (Statistics Canada) | Medium | High. Agronomic and census data are natively compiled at this level. | Medium-High. Directly aligns with regional poultry counts and crop distributions. |
| **Census Division (CD)** | ~293 units (Statistics Canada) | Medium-Fine | High. Good compromise for localizing outbreaks while preserving farm density aggregates. | **High (Preferred)**. Offers a localized scale that matches both epidemiological action and ecological bird counts. |
| **Grid Cells (e.g., 10 km)** | Thousands of cells | Fine | High for remote sensing, but very low for agricultural poultry counts due to privacy constraints. | Low-Medium. Aggregating confidential farm records to small grids violates confidentiality and leads to extreme label sparsity. |

---

## Chosen Spatial Strategy

### Primary Unit: Census Division (CD) or Census Agricultural Region (CAR)
We prefer **Census Divisions (CDs)** (or **Census Agricultural Regions (CARs)**) as the primary spatial node in our GNN.
* **Rationale:** Outbreak events are concentrated locally (e.g., in Fraser Valley, BC, or Southern Ontario). A province-level model is too coarse to be useful, while a grid-based model suffers from extreme sparsity and privacy-driven data suppression. CDs provide a standard geographical boundary containing agricultural statistical aggregates (poultry farm density, animal population sizes) and allow alignment with eBird checklist locations and environmental anomaly grids.

* **Finalized Implementation:** We processed the raw 2021 Census Subdivision (CSD) boundaries shapefile (`lcsd000b21a_e.shp` containing 5,161 polygons) and dissolved them using the 4-digit parent CDUID prefix. This generated **exactly 293 Census Division polygons** saved in [census_divisions_wgs84.geojson](file:///d:/gdrive/models/hpai-heterogeneous-gnn-canada/data/processed/boundaries/census_divisions_wgs84.geojson) (for web sharing) and [census_divisions_epsg3347.gpkg](file:///d:/gdrive/models/hpai-heterogeneous-gnn-canada/data/processed/boundaries/census_divisions_epsg3347.gpkg) (for projected spatial analysis).

* **CRS Standards for Area Calculations:** To avoid geometric distortions when calculating area-based features (e.g., land area, farm density, road density), all vector metrics are calculated using the projected **`EPSG:3347` (NAD83 / Statistics Canada Lambert)** coordinate system.

### Falling Back to Provinces
* Public CFIA reports sometimes redact the exact municipality or Census Division of a poultry outbreak to protect commercial interests, reporting only the province. 
* **Fallback Rule:** If geocoding or spatial association of past poultry outbreaks to specific Census Divisions is restricted or incomplete, the project will fallback to **Province-level modeling** as Stage 1 and Stage 2 baselines, or utilize a multi-resolution hierarchy where regional predictions are constrained by provincial aggregates. In our current CFIA dataset, 632 out of 634 records contain coordinates, meaning we can bypass the province fallback for almost all records and use high-confidence spatial joins.

### Join and Modeling Policy: Use Stable IDs
* **Mandate Stable Identifiers:** To prevent errors caused by spelling variations, bilingual names (e.g., `Quebec` vs. `Quebec / Québec`), or translation changes, all database joins, feature groupings, and modeling keys must use stable numeric identifiers (such as `cd_uid` or `province_code`) rather than text display names.
* **Text Fields for Previews Only:** Fields like `province_name_en` or `province_name_bilingual` are intended purely for labels, map legends, and human-readable previews. They must never be used as keys for grouping, merging, or joining data.

---

## Hierarchical Validation: Census Subdivision (CSD) to Census Division (CD)

An administrative rollup validation audit was performed:
1. **Administrative Scale:**
   - There are 5,161 Census Subdivisions (CSDs) representing municipalities, towns, and local reserves in Statistics Canada's 2021 database.
   - There are 293 Census Divisions (CDs) representing counties or regional districts.
2. **Hierarchy Rule:**
   - The first 4 digits of a 7-digit CSD unique code (`CSDUID`) represent its parent Census Division unique code (`CDUID`).
3. **Validation Checkpoint Results:**
   - Evaluated all 5,161 CSD records against the 293 CD boundaries.
   - All CSDs mapped cleanly to their parent CDs with zero orphans (100% hierarchy matching coverage).
   - Validated that total land areas sum up correctly when dissolving CSD polygons to CDs.
   - A crosswalk lookup table was successfully generated and saved at [csd_to_cd_crosswalk.csv](file:///d:/gdrive/models/hpai-heterogeneous-gnn-canada/data/processed/boundaries/csd_to_cd_crosswalk.csv) to map municipality names and subdivisions to modeling node indices.
