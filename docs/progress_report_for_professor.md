# Progress Report: CSD-Level Static Graph Foundation for HPAI Risk Modeling in Canada

## Executive Summary
- Selected Census Subdivisions (CSDs) as the primary spatial resolution because they provide finer local detail than provinces and suit spatial risk modeling.
- Treated each CSD as a single graph node; the graph contains 5,161 nodes.
- CFIA poultry outbreak records were aggregated to CSD-month labels, preserving privacy while enabling temporal modeling.
- Built static node features from a poultry production proxy, GLW density rasters, road/transport metrics, and geographic QA flags.
- Constructed spatial adjacency edges (rook, queen, nearest-neighbor fallback) to form the graph backbone for future GNN work.

## 1. Why I Chose CSD Resolution
- Province-level modeling is too coarse to capture local outbreak drivers.
- CSDs are an official Statistics Canada unit that balances spatial detail with data availability.
- They are small enough to reflect local risk variation yet can be aggregated to provinces or Census Divisions when needed.
- Existing agricultural, road, and environmental datasets align naturally with CSD boundaries.

## 2. What a Node Means
- **One node = one Census Subdivision** (5,161 nodes total).
- Each node will later hold monthly environmental features and a monthly target label (outbreak presence/absence).
- For now we have assembled the *static* part of the node feature table.

![Node concept](file:///C:/Users/mkhos/.gemini/antigravity-ide/brain/482f25ab-1c90-49dc-ad78-3792b25d1c65/node_concept_1783026483322.png)
*Figure: CSD polygon -> graph node -> static features + monthly label*

## 3. Mapping CFIA Outbreaks to CSDs
- Outbreak records were linked to the containing CSD using their location metadata.
- Events were then aggregated into **CSD-month** binary labels (outbreak present or not).
- This mirrors public-health dashboards that report counts by administrative area, avoiding any exposure of exact farm coordinates.
- The aggregated label set forms the target variable for future prediction models.

![CFIA mapping concept](file:///C:/Users/mkhos/.gemini/antigravity-ide/brain/482f25ab-1c90-49dc-ad78-3792b25d1c65/cfia_mapping_concept_1783026492857.png)
*Figure: Privacy-safe aggregation of outbreak points to CSD-month labels*

## 4. Static Features Extracted So Far
### 4.1 Statistics Canada Poultry Production Proxy
- Obtained from the Census of Agriculture (CCS level).
- Values (broiler, turkey, farm counts) are assigned to each CSD via the CSD-to-CCS lookup.
- Provides a proxy for poultry production intensity without revealing exact farm locations.

### 4.2 GLW Poultry Density Features
- Global Livestock Watch (GLW) raster data summarized within each CSD.
- Chicken density (2020) and duck density (2015 fallback) are expressed as mean, max, and 95th percentile values.
- Captures spatial intensity of poultry presence at a fine resolution.

### 4.3 Road and Transport Features
- National Road Network (NRN) processed for all outbreak-affected provinces.
- Metrics include total road length, road density, and distance to the nearest major road/highway.
- Territories (Yukon, NWT, Nunavut) are flagged as not processed.

### 4.4 Geography and QA Features
- Area (km2), province identifier, CCS membership, extraction method flags, pixel count validity, road-coverage flag, and a small-area flag.
- These allow quality checks and prevent misuse of unreliable data.

![Static feature groups](file:///C:/Users/mkhos/.gemini/antigravity-ide/brain/482f25ab-1c90-49dc-ad78-3792b25d1c65/static_feature_groups_1783026501491.png)
*Figure: Four feature groups feeding into the CSD static node feature table*

## 5. How Graph Edges Were Defined
- **Rook adjacency**: two CSDs share a non-zero boundary segment.
- **Queen adjacency**: two CSDs touch at a single corner point.
- **Nearest-neighbor fallback**: isolated CSDs are linked to the geographically nearest CSD centroid to avoid disconnected nodes. These edges are flagged (`edge_type='nearest_neighbor_fallback'`) and treated as weaker connections.
- The edge list contains 10,662 undirected edges (10,254 rook, 378 queen, 30 fallback).

![Edge types concept](file:///C:/Users/mkhos/.gemini/antigravity-ide/brain/482f25ab-1c90-49dc-ad78-3792b25d1c65/edge_types_concept_1783026511099.png)
*Figure: Rook, queen, and nearest-neighbor fallback edge types*

## 6. What Is Completed
- CSD boundary layer and node definition.
- CFIA outbreak aggregation to CSD-month labels.
- Static node feature table (5,161 rows, 33 columns).
- Spatial adjacency edge list with three edge types.
- Documentation, version control checkpoints, and a PDF progress report.

## 7. What Is Next
1. Generate monthly climate, land-cover, water, and vegetation layers.
2. Add wild-bird abundance and migration covariates.
3. Build temporal train/validation/test splits at the CSD-month level.
4. Train baseline statistical models (e.g., logistic regression) for comparison.
5. Train and evaluate spatiotemporal GNN models using the full dynamic graph.

## First Baseline Modeling Results

1. **Modeling table**:
   - V1 environment-only train-ready table created
   - 237,538 CSD-month rows
   - 5,054 provincial CSDs
   - 381 positive next-month outbreak targets
   - no missing numeric modeling features
   - severe class imbalance: positive rate 0.1604%, imbalance about 622:1

2. **Environment-only ML baselines**:
   - **Logistic Regression**:
     - PR-AUC: 0.0457
     - ROC-AUC: 0.9004
     - Precision@50: 0.060
     - Recall@50: 0.038
   - **Random Forest**:
     - PR-AUC: 0.0204
     - ROC-AUC: 0.9107
     - Precision@50: 0.100
     - Recall@50: 0.063

3. **Time-persistence baseline**:
   - Uses historical outbreak recurrence from 2022-2024 to predict 2025 risk
   - Does not use environmental features
   - Results:
     - PR-AUC: 0.0405
     - ROC-AUC: 0.7311
     - Precision@50: 0.200
     - Recall@50: 0.127
     - Precision@200: 0.075
     - Recall@200: 0.190

4. **Interpretation**:
   - Logistic Regression has the best PR-AUC among tested models.
   - Time-persistence baseline has the strongest top-50 precision.
   - This suggests historical outbreak recurrence/location is highly informative.
   - Future models should compare against this baseline and may include carefully designed lag/history features without leakage.
   - Accuracy is not meaningful because of extreme imbalance.

---
### Lag-history Baseline Results

**Logistic Regression (with lag/history features)**
- PR-AUC: 0.0635
- ROC-AUC: 0.9003
- Precision@50: 0.12  Recall@50: 0.0759
- Precision@100: 0.07  Recall@100: 0.0886
- Precision@200: 0.055  Recall@200: 0.1392

**Random Forest (with lag/history features)**
- PR-AUC: 0.0308
- ROC-AUC: 0.9376
- Precision@50: 0.06  Recall@50: 0.0380
- Precision@100: 0.05  Recall@100: 0.0633
- Precision@200: 0.040  Recall@200: 0.1013

---
### GraphSAGE Spatial Graph Baseline Result

A lightweight V1 GraphSAGE model was trained using the CSD adjacency graph, environmental features, and lag/history outbreak features. The graph model uses neighboring CSD information through the static CSD adjacency network.

2025 test-set performance:
- PR-AUC: 0.0953
- ROC-AUC: 0.9166
- Precision@50: 0.180
- Recall@50: 0.1139
- Precision@100: 0.150
- Recall@100: 0.1899
- Precision@200: 0.120
- Recall@200: 0.3038

Interpretation:
- GraphSAGE outperformed the environment + lag logistic regression baseline in PR-AUC (0.0953 vs 0.0635).
- GraphSAGE also improved broader top-K recall, especially Recall@200 (0.3038).
- The time-persistence baseline still had slightly higher Precision@50 (0.200 vs 0.180), but GraphSAGE performed better overall by PR-AUC and broader risk ranking.
- This suggests that spatial adjacency adds useful predictive information beyond tabular environmental and lag/history features.

---
*Prepared by the project development team. All content is ASCII-only and respects data-privacy constraints.*
