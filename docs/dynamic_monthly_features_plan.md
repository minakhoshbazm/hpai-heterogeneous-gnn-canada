# Dynamic Monthly Features Plan

## 1. Purpose
The static CSD graph foundation is now complete, but HPAI risk changes over time.  To predict next-month outbreak risk we need **monthly dynamic predictors** that describe the evolving environmental, climatic, habitat, and wild-bird conditions for each Census Subdivision (CSD).

## 2. Why Monthly for V1?
- **Label sparsity**: Aggregating outbreak labels to the CSD-month level provides enough positive cases for training while keeping the dataset manageable.
- **Data alignment**: Most climate, satellite, land-cover, and bird-occurrence products are regularly released on a monthly schedule, making month the natural temporal grain.
- **Interpretability**: Monthly aggregates are easier to validate and explain in the first reproducible pipeline.
- **Future work**: A 14-day or bi-weekly window can be explored later (V2) if data quality and label density support it.

## 3. Target Temporal Unit
The modeling unit will be **CSD-month**.
Each row will contain:
- `csd_uid`
- `year`
- `month`
- dynamic feature values for that month
- **next-month** outbreak target (binary presence/absence)

> **Important**: No exact outbreak coordinates, raw private locations, or future information that would cause label leakage will be included.

## 4. Planned Monthly Feature Groups

### 4.1 Climate
Climate strongly influences virus persistence and bird migration.
Potential monthly features (computed from ERA5-Land or Daymet):
- mean temperature
- minimum temperature
- maximum temperature
- total precipitation
- snow cover or snow-water equivalent (if available)
- freeze-thaw indicator (optional)
- climate anomaly relative to long-term monthly averages (future extension)

### 4.2 Land-cover and surface-water
Land-cover and water affect habitat suitability for wild birds and proximity of poultry to water bodies.
Potential monthly features (Dynamic World, JRC Global Surface Water, GEE):
- water fraction per CSD-month
- cropland fraction
- wetland or herbaceous wetland fraction
- built-up / urban fraction
- monthly change in water area
- surface-water anomaly (future extension)

### 4.3 Seasonal and Permanent Small Water Bodies
Small ponds, seasonal wetlands, and flooded fields are key stopover sites for migratory waterfowl.
Distinguish:
- **Permanent water** – present most of the year
- **Seasonal water** – appears only in certain months (spring melt, flood events)
- **Small water body density** – count or area of ponds/wet patches inside or near each CSD (if feasible)

Potential features:
- permanent water fraction per CSD
- seasonal water fraction per CSD-month
- small-pond area or density (optional)
- monthly change in water area
- spring-melt or flooding indicator
- distance to nearest permanent or seasonal water body (future work)

**Recommended V1**: start with CSD-level monthly water fraction and a seasonal water occurrence proxy before attempting exact pond inventories.

### 4.4 Vegetation and Moisture
Vegetation health and surface moisture indicate habitat quality for birds.
Potential monthly features (Sentinel-2, GEE):
- NDVI mean, max, 95th percentile
- NDWI mean, max, 95th percentile
- vegetation anomaly (future)
- surface-moisture anomaly (future)

### 4.5 Wild Bird Abundance or Migration
Migratory waterfowl and other AIV reservoir species drive regional risk.
Potential monthly features (eBird Status & Trends, eBird BDS, GBIF):
- monthly abundance or presence of selected waterfowl species
- monthly abundance of AIV reservoir species groups
- migration intensity per CSD-month
- species richness or total waterfowl counts
- flyway or seasonal migration indicators

**Caveat**: Bird observations are subject to effort bias; documentation of observation effort and a simple bias-adjustment will be recorded.

## 5. Proposed Output Table
```
data/processed/features/csd_monthly_dynamic_features.csv
```
**Key columns**:
- `csd_uid`
- `year`
- `month`
- one column for each dynamic feature (e.g., `temp_mean`, `precip_total`, `water_frac`, `ndvi_mean`, `bird_abundance_waterfowl`, ...)

The table will have **one row per CSD-month**.  No private coordinates or future target information will be stored.

## 6. Validation Checks
- Exactly one row for each CSD-month combination.
- No duplicate CSD-month rows.
- All CSDs represented for each month, or missingness documented.
- Date range matches the CFIA label period.
- CRS and boundary consistency confirmed against the static CSD layer.
- Summary of missing values by feature group.
- Temporal coverage summary for each group.
- Verify no label leakage: all dynamic features are computed using data **available up to the prediction month**.
- Ability to join cleanly with static node features on `csd_uid`.
- Ability to join cleanly with the label table on `csd_uid`, `year`, `month`.

## 7. Modeling Note
During model training, the monthly dynamic table will be joined with:
- static CSD node features (already built)
- CSD spatial graph edges (rook, queen, nearest-neighbor fallback)
- next-month outbreak target label

The final modeling row thus represents:
```
CSD-month predictors  -->  next-month outbreak risk
```
Baseline statistical models (e.g., logistic regression) will be evaluated before scaling to a full spatiotemporal GNN.

## 8. Recommended First Implementation Step
1. **Climate** – widely available, naturally temporal, and straightforward to validate.
2. **Water / seasonal water proxies** – add surface-water context.
3. **Vegetation / moisture** – incorporate habitat quality.
4. **Wild bird abundance / migration** – enrich ecological signal.

Following this order ensures a solid, verifiable pipeline before adding more complex satellite-derived or species-specific products.

---
