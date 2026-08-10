# Repository Audit Report
## Hybrid Local-Spatial Residual GraphSAGE Project for HPAI Outbreak Prediction in Canada

---

## Executive Summary

This repository audit provides a comprehensive code, pipeline, data schema, and evaluation audit of the HPAI outbreak prediction codebase. 

### Audit Highlights:
1. **Target Mapping & Time Safety Verified**: All models adhere strictly to $X_{i,t} \rightarrow Y_{i,t+1}$, where features aggregated up to the final day of calendar month $t$ predict binary outbreak occurrence $Y_{i,t+1} \in \{0, 1\}$ in month $t+1$.
2. **Reconciliation of Cohort Row Counts (24,233 vs 24,442 Rows)**:
   - **Commercial Farm Reporting Cohort** (`ccs_farms_reporting > 0`): Exactly **2,203 CSDs**, which across 11 test months in 2025 equals **24,233 region-month rows** (38 positive outbreaks, prevalence = 0.1568%).
   - **Commercial Farm Capacity Cohort** (`ccs_farms_reporting > 0 OR ccs_production_kg > 0`): Exactly **2,222 CSDs**, which across 11 test months in 2025 equals **24,442 region-month rows** (45 positive outbreaks, prevalence = 0.1841%).
3. **Data Leakage & Preprocessing Rules**: All standard scalers, imputation parameters, negative-sampling masks, and Platt calibration sigmoids are fitted **strictly on 2022–2024 training/validation data** with zero leakage into the 2025 evaluation set.

---

## 1. Repository Structure & Key Components

```text
hpai-heterogeneous-gnn-canada/
├── data/
│   ├── processed/modeling/
│   │   ├── model_table_v1_environment_with_lags_wildbird.csv (237,538 rows x 62 features)
│   │   └── model_table_v1_environment_with_lags.csv
│   └── processed/graph/
│       └── csd_adjacency_edges_v1_model_scope.csv (Spatial kNN-3 and boundary edges)
├── scripts/
│   ├── train_mlp_environment_lag_strict.py (Standalone PyTorch MLP baseline)
│   ├── train_graphsage_baseline_v1_early_stopping_strict.py (PyG GraphSAGE baseline)
│   ├── train_spatiotemporal_directed_relational_gnn_transfer.py (4-relation PyG R-GCN)
│   ├── run_pseudoabsence_ablation_rgcn_transfer.py (Imbalance & negative sampling ablation)
│   └── run_rolling_wildbird_supervised_transfer_graphsage.py (Rolling validation pipeline)
└── results/
    ├── baselines/ (Contains predictions, metrics, and summaries for standard baselines)
    ├── pseudoabsence_ablation/ (Contains negative ratio ablation predictions)
    ├── publication_readiness/ (Contains stage 0-20 reports, canonical benchmarks, and manuscript)
    └── hybrid_local_spatial_graphsage/ (NEW directory for hybrid model execution)
```

---

## 2. Dataset Schemas & Feature Availability

- **Unit of Analysis**: Census Subdivision (CSD $i$) $\times$ Calendar Month ($t$).
- **Total Spatiotemporal Scope**: 5,054 Canadian CSDs across 47 calendar months (Jan 2022 – Nov 2025), giving **237,538 region-month records**.
- **Features (62 numeric features)**:
  - ERA5-Land monthly climate (mean temp, precipitation, dewpoint, thermal anomalies).
  - Sentinel-2 Dynamic World remote sensing (surface water coverage, water bodies distance, NDVI).
  - Host Density (GLW2020 chicken and duck density, Census of Agriculture farm counts & production).
  - Temporal Lags (12 features): `same_csd_outbreak_current_month`, `same_csd_outbreak_prev_1_month`, `past_3_month_count`, `past_12_month_count`, `province_current_month_count`.
- **Target Variable**: `outbreak_next_month` $\in \{0, 1\}$ (1-month-ahead forecast $t+1$).

---

## 3. Temporal Split & Isolation Audit

| Temporal Split | Calendar Period | Total Region-Months | Outbreak Count ($Y=1$) | Natural Prevalence | Role in Hybrid Pipeline |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Development Set** | Jan 2022 – Dec 2023 | 121,296 | 262 | 0.002160 (0.216%) | Model training & representation learning |
| **Validation Set** | Jan 2024 – Dec 2024 | 60,648 | 40 | 0.000660 (0.066%) | Architecture selection & Platt calibration |
| **Retrospective Test** | Jan 2025 – Nov 2025 | 55,594 | 79 | 0.001421 (0.142%) | Post-hoc retrospective benchmark ONLY |

*Leakage Verification*: All scaler objects (`StandardScaler`) and imputation routines are fit on 2022–2024 training data and applied to 2025 test data without updating parameters.

---

## 4. Graph Construction & Message-Passing Setup

- **Spatial Graph**: Homogeneous spatial graph constructed using $k$-Nearest Neighbors ($k=3$) geographic proximity edges between CSD centroids (`csd_adjacency_edges_v1_model_scope.csv`).
- **Graph Safety**: Graph edges use static geographic centroid distance and static boundary adjacency. Edge structures contain zero target labels or dynamic outcome information.
- **Relational Graph**: Spatiotemporal Directed R-GCN uses 4 edge types: (1) Self-temporal transition, (2) $kNN-3$ spatial geographic, (3) Environmental surface water similarity, and (4) Commercial poultry density similarity.

---

## 5. Reconciling Primary Commercial Cohort Discrepancy

We audited the cohort construction functions and resolved the exact source of the 24,233 vs 24,442 row-count variance:

```text
Cohort Definition A: ccs_farms_reporting > 0
  - CSD Count: 2,203 CSDs
  - 2025 Test Rows (11 months): 2,203 x 11 = 24,233 region-months
  - 2025 Outbreaks Retained: 38 outbreaks (Prevalence = 0.1568%)

Cohort Definition B: ccs_farms_reporting > 0 OR ccs_production_kg > 0
  - CSD Count: 2,222 CSDs
  - 2025 Test Rows (11 months): 2,222 x 11 = 24,442 region-months
  - 2025 Outbreaks Retained: 45 outbreaks (Prevalence = 0.1841%)
```

**Resolution**: Definition B (`ccs_farms_reporting > 0 OR ccs_production_kg > 0`) is adopted as the **Primary Commercial Poultry Cohort** because it retains 45 (57.0%) of national outbreaks compared to 38 (48.1%) for Definition A. All scripts, tables, and manuscript text will be updated to explicitly report **24,442 rows and 45 outbreaks** for the Primary Commercial Cohort.

---

## 6. Existing Model Architectures & Pipeline Reuse

1. **MLP Model (`train_mlp_environment_lag_strict.py`)**:
   - PyTorch Module: `Linear(62 -> 32) -> ReLU -> Dropout(0.2) -> Linear(32 -> 32) -> ReLU -> Dropout(0.2) -> Linear(32 -> 1)`.
   - Optimized with Adam (`lr=0.005`, `weight_decay=1e-4`).
2. **GraphSAGE Model (`train_graphsage_baseline_v1_early_stopping_strict.py`)**:
   - PyTorch Geometric Module: `SAGEConv(62 -> 32) -> ReLU -> Dropout(0.2) -> SAGEConv(32 -> 32) -> ReLU -> Dropout(0.2) -> Linear(32 -> 1)`.
3. **Platt Calibration & Brier Skill Score (`scratch/execute_peer_review_fixes.py`)**:
   - Out-of-fold Platt scaling logistic regression fitted on 2024 validation logits to output true posterior probabilities $\hat{P}(Y=1|X)$.

---

## 7. Audit Conclusion & Next Steps

The repository pipeline is clean, time-safe, reproducible, and ready for the implementation of the **Local-Spatial Residual GraphSAGE** hybrid model.
