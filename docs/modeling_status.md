# Modeling Status & Prerequisites

This document outlines the current status of spatial modeling within the project and defines the strict prerequisites for scientific runs.

---

## 1. Local GNN Smoke Tests Status
The recent local executions of the GCN (`gcn_model.pt`) and GraphSAGE (`sage_model.pt`) models:
*   **Must be treated exclusively as toy smoke tests.**
*   **Do not represent valid scientific results.**
*   They were run on synthetic/mock features to verify model integration, training loop mechanics, and tensor shapes.

---

## 2. Modeling Prerequisites (Data Foundation Work)
No valid scientific modeling or forecasting can begin until the following data layers are fully cleaned, validated, and documented:

1.  **Official CFIA Labels & Cleaning:** Outbreak cleaning, validation, deduplication, and mapping must be finalized based on the [CFIA Outbreak Cleaning and Spatial Mapping Plan](file:///d:/gdrive/models/hpai-heterogeneous-gnn-canada/docs/cfia_cleaning_plan.md).
2.  **Boundary CRS Audit:** Ensuring geometric consistency and preventing area-based distortions by verifying Statistics Canada shapefiles and validating all operations under projected `EPSG:3347`.
3.  **Census Division & Census Subdivision Validation:** Resolving and documenting administrative and geographic mismatches across Stats Canada subdivisions.
4.  **CFIA-to-CSD/CD Mapping:** Verifying high-confidence spatial joins and resolving coordinate discrepancies or municipal fallback mappings.
5.  **CD-Month Label Matrix:** Generating the target label matrix `outbreak_next_month` using stable identifiers (`cd_uid`).
6.  **Road Networks & Density:** Cleaning and validating local highway buffer distances and trade density layers.
7.  **Ecological (eBird) Features:** Processing, filtering, and aligning seasonal and species-specific migratory waterfowl abundance features.

---

## 3. Modeling Policy
*   **The CSD label table is the current official label foundation, but modeling remains paused until feature layers are added and QA is reviewed.**
*   **No new model architectures** will be implemented.
*   **No local or Compute Canada training jobs** will be run.
*   **No Slurm jobs** will be submitted.
*   All future modeling attempts are strictly paused until all data foundation milestones are completed, validated, and formally reviewed.
