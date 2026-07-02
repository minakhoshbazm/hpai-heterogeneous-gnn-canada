# Progress Report: CSD-Based Spatiotemporal Graph Foundation for HPAI Risk Modeling in Canada

## Executive Summary
- Moved from province‑level to Census Subdivision (CSD)‑level graph nodes.
- Mapped CFIA poultry outbreak labels to CSD‑month observations.
- Built static CSD node features from StatCan poultry proxy, GLW density, and road transport data.
- Constructed a spatial adjacency graph (rook, queen, nearest‑neighbor fallback edges).
- Project is now ready for dynamic monthly environmental feature integration and model training.

## 1. Project Goal
The aim is to create a reproducible machine‑learning pipeline that predicts, one month ahead, the risk of Highly Pathogenic Avian Influenza (HPAI) outbreaks in Canadian poultry.  The approach uses a graph where each node corresponds to a Census Subdivision (CSD).  Monthly environmental, land‑cover, and wild‑bird variables will later be added as dynamic node features.

## 2. Why Census Subdivisions?
- **Finer granularity** than provinces, allowing more local risk patterns to be captured.
- **Standard geographic unit** from Statistics Canada, with well‑defined boundaries.
- Enables aggregation to province or health‑district levels when needed while keeping a detailed base layer.
- Currently, CSD is the primary node scale for the graph.

## 3. What Has Been Completed
- CSD boundary shapefile prepared and normalised to EPSG:3347.
- CFIA poultry outbreak records linked to CSD‑month labels.
- Lookup table linking each CSD to its parent Census Consolidated Subdivision (CCS) built.
- StatCan poultry production proxy (broiler & turkey) aggregated from CCS to CSD.
- Global Livestock Watch (GLW) poultry density (chicken 2020, duck 2015 fallback) summarised for each CSD.
- NRN road network metrics (total length, density, centroid‑to‑road distance) extracted for each CSD.
- Spatial adjacency graph constructed using rook and queen contiguity plus nearest‑neighbor fallback edges for isolated polygons.
- Unified static node feature table created (`csd_static_structural_features.csv`).
- Full dataset inventory documented and committed.

## 4. Current Data Products
| Data product | Location | Rows | Columns / Details |
|--------------|----------|------|-------------------|
| Static structural feature table | `data/processed/features/csd_static_structural_features.csv` | 5,161 | 33 columns – metadata, poultry proxy, GLW density, road features, QA flags |
| Spatial adjacency edges | `data/processed/graph/csd_adjacency_edges.csv` | 10,662 (undirected) | Rook = 10,254, Queen = 378, Nearest‑Neighbor fallback = 30 |
| CCS‑level poultry proxy | `data/processed/poultry/ccs_poultry_proxy.csv` | 1,757 (CCS) | Production weights, farm counts |
| GLW poultry density | `data/processed/glw/csd_glw_poultry_density.csv` | 5,161 | Mean, max, 95th percentile for chicken & duck |
| Road transport features | `data/processed/roads/csd_road_features.csv` | 5,161 | Length, density, distance metrics for all/major/highway roads |
| CSD‑to‑CCS lookup | `data/processed/lookups/csd_to_ccs_lookup.csv` | 5,161 | Maps each CSD to its parent CCS |

## 5. Figures
### Figure 1 – Canada CSD overview
![Figure 1](file:///C:/Users/mkhos/.gemini/antigravity-ide/brain/482f25ab-1c90-49dc-ad78-3792b25d1c65/figure1_csd_map_1783025933141.png)
*Light‑gray CSD boundaries across Canada.*

### Figure 2 – Static feature schema
![Figure 2](file:///C:/Users/mkhos/.gemini/antigravity-ide/brain/482f25ab-1c90-49dc-ad78-3792b25d1c65/figure2_static_feature_schema_1783025946046.png)
*Each CSD node aggregates metadata, poultry proxy, GLW density, road features and QA flags; edges connect neighbouring nodes.*

### Figure 3 – Edge type counts
![Figure 3](file:///C:/Users/mkhos/.gemini/antigravity-ide/brain/482f25ab-1c90-49dc-ad78-3792b25d1c65/figure3_edge_counts_1783025957550.png)
*Bar chart of rook, queen and nearest‑neighbor fallback edges.*

## 6. Scientific Interpretation
The current work provides a **static graph foundation**.  All spatial relationships and structural covariates are in place, but temporal covariates (monthly climate, land‑cover, water indices, vegetation, and wild‑bird abundance) are not yet added.  This foundation is essential because the graph topology and static features define how future dynamic variables will be propagated through a spatiotemporal Graph Neural Network.

## 7. Data Privacy and Reproducibility
- Raw CFIA point data are excluded from the repository; only aggregated CSD‑month labels are stored.
- No exact outbreak coordinates appear in any figure or table.
- All processing scripts are version‑controlled; outputs are deterministic given the same inputs.
- The repository `.gitignore` prevents accidental inclusion of raw or proprietary data.

## 8. Next Steps
1. Generate monthly environmental layers (temperature, precipitation, vegetation indices).
2. Incorporate land‑cover and water‑body masks.
3. Add wild‑bird migration and abundance estimates.
4. Create temporal training/validation/test splits at the CSD‑month level.
5. Train baseline statistical models (e.g., logistic regression) for benchmarking.
6. Train and evaluate spatiotemporal GNN models using the static graph and dynamic features.

## 9. References & Resources
- Statistics Canada Census Subdivision (CSD) boundaries.
- CFIA HPAI outbreak reports (aggregated to CSD‑month).
- Global Livestock Watch (GLW) poultry density rasters.
- National Road Network (NRN) shapefiles.
- Python libraries: pandas, geopandas, networkx, PyG.

---
*Prepared by the project development team. All content is ASCII‑only and complies with data‑privacy constraints.*
