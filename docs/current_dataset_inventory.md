# CSD Dataset and Static Graph Inventory

---

## 1. Primary Spatial Unit
The **Census Subdivision (CSD)** is the primary spatial scale used in this project. Each CSD represents a municipality or equivalent area (e.g., Indian reserves, unorganized territories) as defined by Statistics Canada. In our graph model, CSDs act as the **nodes** (vertices), with a total of **5,161 CSD nodes** spanning all provinces and territories.

---

## 2. Current Target Labels
Historical poultry outbreaks documented by the Canadian Food Inspection Agency (CFIA) have been mapped to **CSD-month labels**. These labels record the presence or absence of HPAI outbreak events within each subdivision for each month.

---

## 3. Static Node Features
All static features have been unified into a single node feature table:
`data/processed/features/csd_static_structural_features.csv`

* **Row Count:** 5,161 (exactly one row per CSD node)
* **Column Count:** 33 columns
* **Column Groups:**
  1. **CSD Metadata:** Unique CSD IDs (`csd_uid`), municipal names (`csd_name`), and province markers (`province_uid`, `province_name`).
  2. **Parent-CCS Poultry Proxy:** Agricultural statistics aggregated from larger parent Census Consolidated Subdivisions (CCS) (broiler and turkey production and farms reporting).
  3. **GLW Poultry Density:** Zonal density metrics (mean, max, 95th percentile) for chickens (GLW 2020) and ducks (GLW 2015 fallback).
  4. **Road and Transport Exposure:** Total road lengths, densities, and centroid proximity metrics for all roads, major roads, and highways.
  5. **QA Flags and Safeguards:** Indicators describing zonal extraction pixel counts, road data coverage, and a sensitivity flag for CSDs under 1.0 km2.
* **Missing Value Notes:**
  * **Poultry Proxies:** Missing entries in broiler/turkey columns represent data suppression, not available values, or records not reported at that geography. These missing values should not be treated as zero unless later verified.
  * **Road Features:** The 107 missing values in the nearest distance columns correspond exactly to CSDs in the three northern territories (Yukon, NWT, Nunavut) which were not processed for road segments.
* **Exclusion Policy:** To protect data privacy and model hygiene, this static feature table strictly excludes:
  * Exact latitude and longitude coordinates.
  * Spatial geometries (polygons).
  * Outbreak event labels.
  * Target modeling columns.

---

## 4. Graph Edges
The edge connections are saved in the adjacency edge list:
`data/processed/graph/csd_adjacency_edges.csv`

The graph contains **10,662 undirected edges** (normalized such that `source_uid < target_uid`). Connections are classified into:
1. **Rook Adjacency (10,254 edges):** CSDs that share a boundary length greater than 0 meters.
2. **Queen Adjacency (378 edges):** CSDs that touch only at a corner point (shared boundary length is 0 meters).
3. **Nearest-Neighbor Fallback (30 edges):** Added to connect individual isolated coastal, island, or remote CSD polygons to their geographically nearest neighbor, avoiding isolated graph nodes where possible. Fallback edges are weaker than true land-border adjacency and are flagged separately in the edge list (using `edge_type='nearest_neighbor_fallback'` and `adjacency_method='nearest_centroid'`) so GNN models can handle them uniquely.

---

## 5. Poultry Exposure Data
Two distinct types of poultry exposure data are included to capture agricultural density:
1. **Census of Agriculture (CCS-level) Poultry Production Proxy:** Farm counts and production weights mapped from larger parent Census Consolidated Subdivisions (CCS) to CSDs. These represent official CCS-level agricultural production/farms-reporting proxy values. Note that these are not exact farm locations and do not constitute true CSD-level inventory counts.
2. **GLW Raster-based Poultry Density Features:** Zonal density summaries (birds/km2) extracted from global modeling grids (FAO GLW 2020 chicken and 2015 duck density) directly to CSD boundaries. These represent high-resolution spatial variations.

---

## 6. Road Features
CSD-level road features include lengths, densities, proportions, and centroid-to-road distance metrics for all roads, major roads, and highways.
* **Processed Provinces:** NRN road segments were processed for all 10 outbreak-affected provinces: **British Columbia (BC), Alberta (AB), Saskatchewan (SK), Manitoba (MB), Ontario (ON), Quebec (QC), New Brunswick (NB), Nova Scotia (NS), Prince Edward Island (PEI), and Newfoundland & Labrador (NL)**.
* **Territories:** Road ZIPs for Yukon, NWT, and Nunavut were not processed (marked as `not_processed_territory`).

---

## 7. Files Intentionally Not Tracked by Git
To maintain repository hygiene and respect data licenses:
* All raw input files (boundary shapes, NRN ZIPs, GLW rasters) under `data/raw/` are excluded.
* All processed output CSVs, shapefiles, and summary JSON files under `data/processed/` are excluded.
* These exclusions are enforced via the root `.gitignore` file.

---

## 8. What is Ready
The **static spatial graph foundation** is fully constructed, unit-tested, and validated. This includes:
* The 5,161-row static feature table.
* The 10,662-edge spatial adjacency list.
* Complete documentation mapping the CRS (`EPSG:3347`) and schemas.

---

## 9. What is Not Ready Yet
* **Dynamic Environmental Features:** Monthly climate, land cover/water indexes, vegetation indexes, and wild bird abundance models are not yet processed.
* **Temporal Model Training:** The final dynamic spatio-temporal GNN modeling pipeline remains paused until the temporal features are assembled.
