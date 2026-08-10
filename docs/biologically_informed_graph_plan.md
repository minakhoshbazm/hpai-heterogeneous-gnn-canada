# Biologically Informed Multi-Relation Spatial Graph Design & Audit Plan

## 1. Overview & Objectives

This document details the design, mathematical formulation, feature provenance, and zero-leakage audit for a multi-relation spatial graph engineered for Canada-wide Highly Pathogenic Avian Influenza (HPAI) risk forecasting.

Rather than assuming a single spatial topology (such as shared physical borders or isotropic k-nearest-neighbours), we construct **three distinct, biologically and epidemiologically motivated spatial relations** among Canada's 5,054 Census Subdivisions (CSDs):
1. **Geographic Proximity Relation**: Local spatial proximity based on CSD representative-point distances within 50 km.
2. **Shared Water Habitat Similarity Relation**: Micro-environmental similarity in static water body and flooded vegetation cover for CSD pairs within 100 km.
3. **Poultry Exposure Similarity Relation**: Agricultural exposure similarity in poultry inventory, farm counts, and density for CSD pairs within 100 km.

---

## 2. Zero-Leakage & Data Integrity Audit

> [!IMPORTANT]
> **STRICT ZERO-LEAKAGE AUDIT GUARANTEES**:
> - **NO Bird Observation Data**: Zero usage of eBird, GBIF, or any wild bird occurrence records.
> - **NO Target Label Leakage**: Edge relations are constructed strictly from static environmental and agricultural features derived prior to 2025 (`model_table_v1_environment_with_lags.csv`). No outbreak history labels (`outbreak_next_month`, `outbreak_current_month`, or lag features) are used to construct graph edges.
> - **NO Future Data**: Centroids are calculated in Statistics Canada Lambert projection (`EPSG:3347`, metres) from 2021 CSD boundary geometries (`lcsd000b21a_e.shp`).
> - **Strict 50-Feature Table**: The GNN node feature matrix uses exactly the 50 strictly audited features (43 environment + 7 approved lag features). `ccs_uid`, all `uid` fields, identifiers, names, province IDs, target, and raw `outbreak_current_month` are strictly excluded.

---

## 3. Mathematical Formulation of Edge Relations

All distance calculations use Euclidean distance $d(i, j) = \|\mathbf{c}_i - \mathbf{c}_j\|_2$ between CSD representative-point centroids $\mathbf{c}_i, \mathbf{c}_j$ in `EPSG:3347` (metres).

### Relation 1: `geographic_proximity`
- **Definition**: Connects CSD pairs $(i, j)$ where $d(i, j) \le 50\text{ km}$ ($50,000\text{ m}$).
- **Edge Weighting**: Distance-decay weight:
  $$w_{\text{geo}}(i, j) = \exp\left(-\frac{d(i, j)}{d_0}\right), \quad d_0 = 25,000\text{ m}$$
- **Interpretation**: Captures localized spatial connectivity and continuous spatial decay.

### Relation 2: `shared_water_habitat`
- **Definition**: Connects CSD pairs $(i, j)$ where $d(i, j) \le 100\text{ km}$ AND standardized water habitat feature similarity $S_{\text{water}}(i, j) \ge \tau_{\text{water}}$.
- **Source Feature Columns**:
  - `dw_water_mean`: Mean Dynamic World water surface proportion
  - `dw_water_p95`: 95th percentile Dynamic World water surface proportion
  - `dw_flooded_vegetation_mean`: Mean Dynamic World flooded vegetation proportion
  - `dw_seasonal_pond_proxy`: Seasonal water body variability proxy
- **Similarity Calculation**:
  Each feature vector $\mathbf{v}_i^{\text{water}} \in \mathbb{R}^4$ is standardized ($z$-score transformed across all 5,054 CSDs). Pairwise Cosine Similarity is computed:
  $$S_{\text{water}}(i, j) = \frac{\langle \mathbf{v}_i^{\text{water}}, \mathbf{v}_j^{\text{water}} \rangle}{\|\mathbf{v}_i^{\text{water}}\|_2 \|\mathbf{v}_j^{\text{water}}\|_2}$$
- **Interpretation**: **Habitat Similarity Candidate**, NOT proof of wild bird movement vectors.

### Relation 3: `poultry_exposure_similarity`
- **Definition**: Connects CSD pairs $(i, j)$ where $d(i, j) \le 100\text{ km}$ AND standardized poultry production similarity $S_{\text{poultry}}(i, j) \ge \tau_{\text{poultry}}$.
- **Source Feature Columns**:
  - `ccs_broilers_roasters_cornish_farms_reporting`: Census agriculture broiler farm count
  - `ccs_broilers_roasters_cornish_production_kg`: Census agriculture broiler production volume
  - `ccs_turkey_farms_reporting`: Census agriculture turkey farm count
  - `ccs_turkey_production_kg`: Census agriculture turkey production volume
  - `glw2020_chicken_density_mean`: Gridded Livestock of the World chicken density mean
  - `glw2020_chicken_density_max`: Gridded Livestock of the World chicken density max
  - `glw2015_duck_density_mean`: Gridded Livestock of the World duck density mean
  - `glw2015_duck_density_max`: Gridded Livestock of the World duck density max
- **Similarity Calculation**:
  Each feature vector $\mathbf{v}_i^{\text{poultry}} \in \mathbb{R}^8$ is log-transformed ($\log(1 + x)$) and $z$-score standardized. Pairwise Cosine Similarity $S_{\text{poultry}}(i, j)$ is computed.
- **Interpretation**: **Shared Production-Risk Context Candidate**, NOT proof of poultry transportation routes.

---

## 4. Relational GNN Architecture (`RGCNConv`)

We employ a 2-layer Relational Graph Convolutional Network (`RGCNConv`) in PyTorch Geometric.

$$h_i^{(l+1)} = \sigma \left( W_0^{(l)} h_i^{(l)} + \sum_{r \in \{0, 1, 2\}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}} W_r^{(l)} h_j^{(l)} \right)$$

- Relation $r=0$: `geographic_proximity`
- Relation $r=1$: `shared_water_habitat`
- Relation $r=2$: `poultry_exposure_similarity`

### Hyperparameters:
- `hidden_channels`: 32
- `dropout`: 0.2
- `learning_rate`: 0.005
- `weight_decay`: 1e-4
- `pos_weight`: `train_neg / train_pos` (BCEWithLogitsLoss)
- `early_stopping_patience`: 10 epochs (monitoring validation PR-AUC on 2024-10..2024-12)

---

## 5. Protocol & Scientific Interpretation Guidance

1. **Selection & Evaluation Protocol**:
   - Thresholds $\tau_{\text{water}}, \tau_{\text{poultry}}$ selected strictly on 2022–2024-09 train and 2024-10..2024-12 validation splits across seeds `[11, 22, 33]`.
   - Final model retrained on all 2022–2024 data and evaluated once on 2025 test set.
2. **Cautious Interpretation**:
   - An increase in predictive ranking performance (PR-AUC / ROC-AUC) demonstrates that these multi-relational topologies provide informative message-passing channels for neural network feature aggregation.
   - It does **NOT** constitute direct empirical proof of biological transmission pathways or transport vectors.
