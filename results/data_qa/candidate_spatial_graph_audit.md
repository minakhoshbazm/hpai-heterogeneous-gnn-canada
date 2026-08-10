# Spatial Graph-Topology Audit Report

**Scope**: 5,054 Census Subdivisions (CSDs) across Canada  
**CRS**: Statistics Canada Lambert (`EPSG:3347`, metres)  
**Date**: 2026-07-30  

---

## 1. Executive Summary & Graph Comparison

This audit evaluates candidate spatial graph topologies for Canada-wide HPAI disease risk modeling using GraphSAGE. Candidate graph topologies are evaluated across connectivity, degree distribution, isolation, non-border spatial reach, and suitability for GraphSAGE 2-layer message passing.

### Candidate Graph Metrics

| Graph Candidate | Total Edges | Unique Active Nodes | Isolated Nodes | Connected Components | Mean Degree | Median Degree | Min Degree | Max Degree | Non-Border Edges | Non-Border Dist Mean (km) | Non-Border Dist Range (km) | Too Dense for GraphSAGE? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Shared-border Adjacency** | 10,508 | 5,054 | 0 | 13 | 4.16 | 3.0 | 1 | 75 | 0 | N/A | N/A | No (Practical Density) |
| **Proximity Radius (25 km)** | 20,777 | 4,683 | 371 | 676 | 8.22 | 5.0 | 0 | 47 | 14,534 | 17.43 | 0.85 - 25.0 | No (Practical Density) |
| **Proximity Radius (50 km)** | 70,427 | 4,944 | 110 | 221 | 27.87 | 20.0 | 0 | 131 | 61,551 | 33.65 | 0.85 - 50.0 | No (Practical Density) |
| **Proximity Radius (100 km)** | 229,242 | 5,038 | 16 | 45 | 90.72 | 69.0 | 0 | 374 | 219,451 | 64.67 | 0.85 - 100.0 | Yes (Too Dense - Over-smoothing Risk) |
| **Proximity Radius (200 km)** | 663,376 | 5,054 | 0 | 2 | 262.52 | 218.0 | 1 | 750 | 653,157 | 122.48 | 0.85 - 200.0 | Yes (Too Dense - Over-smoothing Risk) |
| **K-Nearest-Neighbours (k=3)** | 9,655 | 5,054 | 0 | 35 | 3.82 | 4.0 | 3 | 8 | 4,359 | 24.34 | 0.85 - 268.73 | No (Practical Density) |
| **K-Nearest-Neighbours (k=5)** | 15,416 | 5,054 | 0 | 3 | 6.1 | 6.0 | 5 | 11 | 8,500 | 29.21 | 0.85 - 293.62 | No (Practical Density) |
| **K-Nearest-Neighbours (k=8)** | 24,260 | 5,054 | 0 | 1 | 9.6 | 9.0 | 8 | 17 | 15,965 | 35.12 | 0.85 - 335.22 | No (Practical Density) |


---

## 2. Structural & Topological Findings

1. **Shared-Border Adjacency Graph**:
   - Contains **10,508 undirected edges** with a mean node degree of **4.16** (median 3.0, min 1, max 75).
   - All 5,054 model-scope CSDs are represented with degree >= 1 (0 isolated nodes in model scope), organized into **13 connected components** corresponding to geographic landmasses, islands, and river-separated regions.
   - All edges in this graph are physical shared-border adjacencies (0 non-border edges).

2. **Geographic Distance Radius Graphs**:
   - **25 km Radius**: Contains **20,777 edges** (mean degree 8.22), but leaves **371 CSDs isolated** and fragments into **676 components** because large rural/northern CSD centroids exceed 25 km distance.
   - **50 km Radius**: Expands connectivity to **70,427 edges** (mean degree 27.87), reducing isolated CSDs to **110**. Non-border edges average **33.65 km**.
   - **100 km Radius**: Reaches **229,242 edges** (mean degree 90.72), but experiences extreme degree inflation in dense southern clusters (max degree 374).
   - **200 km Radius**: Explodes to **663,376 edges** (mean degree 262.52, max degree 750). **Extremely dense and unsuitable for GraphSAGE training** due to severe over-smoothing (node representations blur across multi-province regions) and massive memory overhead.

3. **K-Nearest-Neighbour (KNN) Graphs**:
   - **k = 3**: Produces **9,655 edges** (mean degree 3.82, min degree 3, max degree 8). **0 isolated CSDs** and **35 connected components**. Ensures all remote CSDs maintain spatial neighbors.
   - **k = 5**: Produces **15,416 edges** (mean degree 6.1, min degree 5, max degree 11). **0 isolated CSDs** and only **3 connected components**. Non-border edges average **29.21 km**.
   - **k = 8**: Produces **24,260 edges** (mean degree 9.6, min degree 8, max degree 17). **0 isolated CSDs** and **1 single fully connected component** across all of Canada. Non-border edges average **35.12 km**.

---

## 3. Recommended Graph Candidates for Next Validation Experiment

Based purely on graph topology, spatial representation, and GraphSAGE training feasibility (without referencing 2025 validation performance):

### Recommendation 1: Sparse Graph Candidate -> **K-Nearest-Neighbours (k = 3)**
- **Topological Rationale**: 
  - Guarantees **0 isolated CSDs** across all 5,054 model-scope CSDs (unlike distance radius graphs which isolate rural northern CSDs).
  - Highly sparse and efficient (**9,655 edges**, mean degree 3.82), serving as a topology candidate for testing localized spatial information sharing without signal dilution.
  - Non-border edges average **24.34 km**, connecting isolated island/coastal CSDs to their nearest regional neighbors.

### Recommendation 2: Moderately Connected Graph Candidate -> **K-Nearest-Neighbours (k = 5)**
- **Topological Rationale**: 
  - Provides **15,416 edges** (mean degree 6.1, max degree 11), serving as a topology candidate for testing broader spatial information sharing across neighboring regions.
  - Reduces total connected components down to **3** across all of Canada while keeping non-border edge distances around **29.21 km**.
  - Controlled maximum degree ensures balanced message passing while preventing over-smoothing.

---

*Report generated automatically by `scripts/audit_candidate_spatial_graphs.py`.*
