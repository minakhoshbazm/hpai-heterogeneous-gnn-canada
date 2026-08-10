# Relational GNN Edge Relation Time-Safety Audit Report

## Summary of Audit Findings

| Relation Name | Source Columns | Temporal Nature | Construction Period | Used 2025 Data | Val Access Post 2024-09 | Time-Safety Status |
|---|---|---|---|---|---|---|
| **geographic_proximity** | CSD Centroids (EPSG:3347) | Static | Fixed 2021 Geometry | NO | NO | **VALID** |
| **poultry_exposure_similarity** | ccs_broilers_roasters_cornish_farms_reporting, ccs_broilers_roasters_cornish_production_kg, ccs_turkey_farms_reporting, ccs_turkey_production_kg, glw2020_chicken_density_mean, glw2020_chicken_density_max, glw2015_duck_density_mean, glw2015_duck_density_max | Static | Static Historical Ag | NO | NO | **VALID** |
| **shared_water_habitat (Primary Dynamic)** | dw_water_mean, dw_water_p95, dw_flooded_vegetation_mean, dw_seasonal_pond_proxy | Dynamic (Monthly) | Snapshot-Specific Expanding Window ($\le M$) | NO in train/val | NO | **STRICTLY TIME-SAFE** |
| **shared_water_habitat (Sensitivity)** | dw_water_mean, dw_water_p95, dw_flooded_vegetation_mean, dw_seasonal_pond_proxy | Dynamic Aggregated | Fixed Historical Reference (2022-01..2024-09) | NO | NO | **SENSITIVITY ANALYSIS** |

## Detailed Audit & Time-Safety Architecture

### 1. Geographic Proximity Edge Relation
- **Source**: CSD centroid coordinates from `data/processed/graph/csd_centroids_epsg3347.csv`.
- **Audit**: Centroids are fixed spatial coordinates. No temporal attributes or labels used. Fully time-safe.

### 2. Poultry Exposure Similarity Edge Relation
- **Source**: Static Census of Agriculture (2021) and Gridded Livestock of the World (GLW 2015/2020).
- **Audit**: Features have zero temporal variation across monthly rows ($std = 0.0$). Fully time-safe.

### 3. Shared Water Habitat Edge Relation (Primary Strict Time-Safe Architecture)
- **Source**: Dynamic World monthly land cover aggregated features (`dw_water_mean`, `dw_water_p95`, `dw_flooded_vegetation_mean`, `dw_seasonal_pond_proxy`).
- **Implementation**: Constructed **month-specific snapshot edge tensors** for each snapshot $M$. For snapshot $M$, water habitat similarity is computed using ONLY Dynamic World observations available in or before month $M$ (expanding historical mean up to month $M$).
- **Leakage Prevention**: Early training snapshots (e.g. 2022-01) never access future Dynamic World covariates (e.g. 2024-09). 2025 test snapshots use only observations up to the test month.

### 4. Shared Water Habitat Edge Relation (Sensitivity Analysis Benchmark)
- **Implementation**: Fixed historical reference period (2022-01..2024-09 mean).
- **Audit Note**: Audited as a sensitivity analysis benchmark. Noted that static aggregation across 2022-2024 allows early 2022 training snapshots to see 2024 covariates under that static graph assumption.

---
*Report generated automatically by `scripts/audit_relational_gnn_time_safety.py`.*
