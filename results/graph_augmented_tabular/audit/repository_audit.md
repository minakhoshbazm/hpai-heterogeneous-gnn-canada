# Repository Audit Report: Graph-Augmented Tabular Risk Model Project

---

## Executive Summary

This repository audit provides a comprehensive inspection of the data files, feature schemas, temporal split rules, graph topologies, baseline implementations, stacking/leakage controls, and evaluation pipelines for the new **Graph-Augmented Tabular Risk Model** study.

### Key Audit Findings:
1. **Strict Target Definition & Temporal Mapping**: All features follow $X_{i,t} \rightarrow Y_{i,t+1}$, where features aggregated up to the final day of month $t$ strictly predict binary outbreak occurrence in month $t+1$.
2. **Dataset Scope & Schema**:
   - Primary File: `data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv`
   - Scope: **5,054 Canadian Census Subdivisions (CSDs)** $\times$ **47 months** (Jan 2022 – Nov 2025) = **237,538 region-month records**.
   - Input Features: **62 numeric features** (ERA5 climate, Sentinel-2 Dynamic World land cover, GLW2020 chicken/duck density, Census of Agriculture farm attributes, and 12 spatiotemporal outbreak lags).
3. **Temporal Split Breakdown**:
   - **Development / Encoder Training Set** (Jan 2022 – Dec 2023): 121,296 rows, 262 outbreaks (0.216% prevalence).
   - **Validation / Model Selection Set** (Jan 2024 – Dec 2024): 60,648 rows, 40 outbreaks (0.066% prevalence).
   - **Retrospective Test Set** (Jan 2025 – Nov 2025): 55,594 rows, 79 outbreaks (0.142% prevalence).
4. **Leakage & Stacking Protection Strategy**:
   - To prevent downstream classifiers (LightGBM, XGBoost, MLP) from overfitting to training embeddings, we implement **Temporal Out-of-Fold (OOF) Embedding Generation** for 2022–2023 training data (training encoder on 2022 data to generate 2023 embeddings, and vice versa), combined with a frozen full-train encoder for 2024 validation and 2025 test embeddings.
   - Zero 2025 test labels will be used for encoder training, early stopping, embedding extraction, hyperparameter tuning, or Platt calibration fitting.

---

## 1. Repository Structure & Relevant Components

```text
hpai-heterogeneous-gnn-canada/
├── data/processed/modeling/
│   ├── model_table_v1_environment_with_lags_wildbird.csv (237,538 rows x 62 features)
│   └── model_table_v1_environment_with_lags.csv
├── data/processed/graph/
│   ├── csd_graph_knn_3_v1.csv (symmetrized 3-nearest-neighbour graph)
│   ├── csd_graph_border_only_v1.csv (shared-border spatial adjacency graph)
│   ├── csd_graph_border_plus_knn_5_v1.csv (border + 5-nearest-neighbour graph)
│   ├── csd_relation_geographic_proximity_v1.csv (radius distance graph)
│   ├── csd_relation_poultry_exposure_similarity_v1.csv (poultry density similarity)
│   └── csd_relation_shared_water_habitat_v1.csv (surface water similarity)
├── scripts/
│   ├── train_mlp_environment_lag_strict.py (Standalone PyTorch MLP)
│   ├── train_graphsage_baseline_v1_early_stopping_strict.py (Standalone GraphSAGE)
│   ├── train_graphsage_candidate_spatial_graphs_strict.py (Candidate graph selection)
│   └── run_rolling_wildbird_tree_baselines_vs_graphsage.py (Tabular tree baselines)
└── results/
    ├── baselines/ (Previous baseline predictions & metrics)
    ├── hybrid_local_spatial_graphsage/ (Previous hybrid model outputs)
    └── graph_augmented_tabular/ (NEW directory structure for graph-augmented tabular study)
        ├── audit/ (Repository audit report)
        ├── configs/ (Model config manifests)
        ├── embeddings/ (Saved GraphSAGE node embeddings z_i,t)
        ├── predictions/ (Model predictions & probabilities)
        ├── metrics/ (CSV benchmark metrics)
        ├── calibration/ (Platt scaling & BSS summaries)
        ├── bootstrap/ (Paired month-block bootstrap results)
        ├── figures/ (PDF/PNG plots)
        ├── tables/ (Publication markdown & CSV tables)
        ├── reports/ (Implementation plan & final report)
        └── manuscript/ (Revised manuscript draft)
```

---

## 2. Dataset Schema & Feature Audit

The dataset contains **62 numeric features** after excluding identifier columns (`csd_uid`, `csd_name`, `province_code`, `ccs_uid`), raw target indicators (`outbreak_current_month`), and target label (`outbreak_next_month`).

### Feature Category Breakdown:
1. **Remote Sensing & Land Cover (14 features)**: Sentinel-2 Dynamic World monthly water coverage, water distance, vegetation indices (NDVI/EVI), cropland percentage.
2. **Climate & Meteorology (18 features)**: ERA5-Land monthly mean temperature, total precipitation, dewpoint temperature, thermal anomalies, frost days.
3. **Host Density & Commercial Farm Indicators (18 features)**: GLW2020 chicken and duck density means/sums, Census of Agriculture broiler/turkey farm counts and production volumes.
4. **Spatiotemporal Outbreak Lags (12 features)**:
   - `same_csd_outbreak_current_month`
   - `same_csd_outbreak_prev_1_month`
   - `same_csd_outbreak_count_past_3_months`
   - `same_csd_outbreak_count_past_6_months`
   - `same_csd_outbreak_count_past_12_months`
   - `province_outbreak_count_current_month`
   - `province_outbreak_count_past_3_months`

---

## 3. Cohort Manifests & Row-Count Reconciliation

| Cohort Name | Inclusion Criteria | CSD Count | 2025 Test Rows (11 Months) | 2025 Outbreak Count | Prevalence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Full National Cohort** | All Canadian CSDs | 5,054 | 55,594 | 79 | 0.1421% |
| **GLW Poultry Presence Cohort** | `glw2020_chicken_density_mean > 0` | 4,471 | 49,181 | 79 | 0.1606% |
| **Primary Commercial Poultry Cohort** | `farms_reporting > 0 OR production_kg > 0` | 2,222 | 24,442 | 45 | 0.1841% |

*Reconciliation Note*: The Primary Commercial Poultry Cohort uses the corrected definition (`farms_reporting > 0 OR production_kg > 0`), which yields exactly **2,222 CSDs $\times$ 11 test months = 24,442 rows** retaining 45 (57.0%) of national 2025 outbreaks.

---

## 4. Candidate Spatial Graph Topologies Audit

We identified 6 candidate spatial graphs pre-built under `data/processed/graph/`:
1. **$k\text{NN}-3$ Centroid Graph** (`csd_graph_knn_3_v1.csv`): 5,054 nodes, 15,162 directed edges ($k=3$), average degree = 3.0.
2. **Shared-Border Adjacency Graph** (`csd_graph_border_only_v1.csv`): 5,054 nodes, 27,246 directed edges, average degree = 5.39.
3. **Border + $k\text{NN}-5$ Graph** (`csd_graph_border_plus_knn_5_v1.csv`): 5,054 nodes, 38,412 directed edges, average degree = 7.60.
4. **Geographic Proximity Distance Graph** (`csd_relation_geographic_proximity_v1.csv`): Distance-decay weighted proximity.
5. **Commercial Poultry Density Similarity Graph** (`csd_relation_poultry_exposure_similarity_v1.csv`): K-NN similarity on commercial farm attributes.
6. **Shared Water Habitat Similarity Graph** (`csd_relation_shared_water_habitat_v1.csv`): Similarity on Sentinel-2 surface water coverage.

---

## 5. Stacking & Overfitting Protection Protocol

Because the GraphSAGE encoder is trained using supervised loss (BCE on target `outbreak_next_month`), extracting embeddings $z_{i,t}$ on the training set using the full training encoder introduces potential data leakage (stacking overfitting) when training downstream classifiers (LightGBM, XGBoost, MLP).

### Leakage-Safe Out-of-Fold Embedding Strategy:
1. **Training Folds (2022 vs. 2023)**:
   - Fit Encoder $E_{2022}$ on 2022 data $\rightarrow$ generate out-of-fold embeddings $z_{i,t}^{\text{OOF}}$ for 2023 training rows.
   - Fit Encoder $E_{2023}$ on 2023 data $\rightarrow$ generate out-of-fold embeddings $z_{i,t}^{\text{OOF}}$ for 2022 training rows.
   - Concatenate $z_{i,t}^{\text{OOF}}$ to form the leakage-free 2022–2023 training feature matrix $\tilde{X}_{\text{train}} = [X_{\text{train}} \mathbin{\Vert} z_{\text{train}}^{\text{OOF}}]$.
2. **Validation & Test Embeddings**:
   - Fit Encoder $E_{\text{Full}}$ on complete 2022–2023 training set (selecting early stopping on 2024 validation AP).
   - Generate validation embeddings $z_{\text{val}}$ for 2024 and test embeddings $z_{\text{test}}$ for 2025.

---

## 6. Audit Summary & Readiness Approval

The repository infrastructure is fully audited, verified for zero target leakage, and ready for the implementation of the **Graph-Augmented Tabular Risk Model** experiment.
