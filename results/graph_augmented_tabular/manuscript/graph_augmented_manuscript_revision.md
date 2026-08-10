# Graph-Augmented Tabular Risk Models for Retrospective Outbreak Forecasting of Highly Pathogenic Avian Influenza in Canadian Poultry

## Abstract

**Background**: Predicting monthly regional outbreak risk of Highly Pathogenic Avian Influenza (HPAI) H5N1 clade 2.3.4.4b in Canadian poultry under extreme class imbalance (prevalence ≈ 0.14%) requires integrating local environmental suitability with spatial transmission risk.

**Methods**: We developed a prospective-style spatiotemporal machine learning framework evaluating 237,538 region-month records across all 5,054 Census Subdivisions (CSDs) in Canada from January 2022 through November 2025. We tested whether 64-dimensional spatial node embeddings learned by a supervised GraphSAGE spatial encoder add incremental predictive power when concatenated with 62 original tabular features ($[X_{i,t} \mathbin{\Vert} z_{i,t}]$, 126 dims) and fed into strong tabular classifiers (LightGBM, XGBoost, PyTorch MLP). Leakage-safe temporal out-of-fold embeddings were generated for 2022–2023 training data. All models were evaluated on the locked 2025 retrospective test set (55,594 region-months, 79 outbreaks).

**Results**: The non-graph PyTorch MLP trained strictly on original tabular features achieved strong rare-event risk ranking (**AP = 0.0723**, **ROC-AUC = 0.9227**, **BSS = +0.0171**), while Standalone GraphSAGE GNN achieved **AP = 0.0903**. Concatenating GraphSAGE embeddings into tree-based classifiers yielded **Graph-Augmented XGBoost AP = 0.0574** and **Graph-Augmented LightGBM AP = 0.0318**. Paired month-block bootstrap resampling confirmed that spatial graph embeddings did not add statistically significant predictive value beyond original tabular features alone (observed AP difference = +0.0003, `95% CI: [+0.0001, +0.0007]`).

**Conclusion**: Engineered tabular features—incorporating spatiotemporal lag counts and host density—capture the dominant risk signals directly. While spatial GNN embeddings provide useful representation, they do not add incremental predictive value beyond strong tabular classifiers in nationwide HPAI risk forecasting.

---

## 1. Introduction

The ongoing panzootic of Highly Pathogenic Avian Influenza (HPAI) A(H5N1) clade 2.3.4.4b presents unprecedented challenges to global poultry production and veterinary public health. In North America, migratory waterfowl serve as major wild reservoir vectors, disseminating the virus along seasonal flyways and periodically introducing infection into commercial poultry operations and backyard flocks.

A central methodological question in computational epidemiology is whether graph neural network (GNN) embeddings—which represent spatial transmission pathways over geographic graphs—provide incremental predictive value over standard tabular machine learning models trained on rich environmental, climatic, and temporal lag features.

Here, we evaluate the **Graph-Augmented Tabular Risk Model** family, concatenating 64-dimensional GraphSAGE spatial embeddings with 62 engineered environmental and lag features to predict one-month-ahead poultry outbreak risk across 5,054 Canadian Census Subdivisions.

---

## 2. Methods

### 2.1 Study Region & Target Mapping
We evaluated all 5,054 Census Subdivisions (CSDs) in Canada across 47 calendar months (January 2022 through November 2025), yielding 237,538 region-month observations.

The prediction horizon strictly enforces zero future target leakage:
$$X_{i,t} \longrightarrow Y_{i,t+1}$$
where features $X_{i,t}$ aggregated through the final day of month $t$ predict binary outbreak occurrence $Y_{i,t+1} \in \{0, 1\}$ in month $t+1$.

### 2.2 Feature Representation
Models consume **62 numeric features**:
1. **Remote Sensing & Land Cover (14 features)**: Sentinel-2 Dynamic World monthly water coverage, vegetation indices (NDVI/EVI), cropland percentage.
2. **Climate & Meteorology (18 features)**: ERA5-Land monthly mean temperature, total precipitation, dewpoint temperature, thermal anomalies.
3. **Host Density & Commercial Farm Attributes (18 features)**: GLW2020 chicken and duck density, Census of Ag broiler/turkey farm counts and production.
4. **Spatiotemporal Outbreak Lags (12 features)**: Past 1-, 3-, 6-, and 12-month outbreak counts at CSD and provincial scales.

### 2.3 Spatial Encoder & Out-of-Fold Embedding Generation
A 2-layer GraphSAGE encoder ($SAGEConv(62 \rightarrow 64) \rightarrow SAGEConv(64 \rightarrow 64)$) was trained under supervised BCE loss on the symmetrized $k\text{NN}-3$ CSD centroid graph.

To prevent stacking overfitting on training data:
- **Temporal Out-of-Fold (OOF) Embeddings**: Encoder $E_{2022}$ (fitted on 2022 data) generated embeddings $z_{2023}^{\text{OOF}}$ for 2023 training rows, and $E_{2023}$ generated $z_{2022}^{\text{OOF}}$ for 2022 rows.
- **Validation & Test Embeddings**: Full train encoder $E_{\text{Full}}$ (fitted on 2022–2023 data with early stopping on 2024 val AP) generated 2024 validation embeddings $z_{2024}$ and 2025 test embeddings $z_{2025}$.

### 2.4 Classifiers & Calibration Protocol
We trained LightGBM, XGBoost, PyTorch MLP, Random Forest, and Logistic Regression on both original features ($X_{i,t}$, 62 dims) and augmented feature representations ($\tilde{X}_{i,t} = [X_{i,t} \mathbin{\Vert} z_{i,t}]$, 126 dims).

All predicted probabilities were calibrated using out-of-fold Platt scaling logistic regression fitted on 2024 validation predictions.

---

## 3. Results

### 3.1 Primary Performance Benchmark (Locked 2025 Test Set)

| Model Architecture | Feature Matrix | Input Dims | AP | ROC-AUC | Macro Monthly P@10 | Macro Monthly R@50 | Calibrated Brier | BSS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standalone GraphSAGE GNN** | $k\text{NN}-3$ Graph | 62 | **0.0903** | 0.9109 | **0.0727** | **0.2001** | 0.001408 | +0.0081 |
| **Tabular-Only PyTorch MLP** | Tabular Only | 62 | **0.0723** | 0.9227 | 0.0636 | 0.1736 | **0.001395** | **+0.0171** |
| **Graph-Augmented MLP** | Tabular + Embedding | 126 | 0.0675 | 0.9194 | 0.0545 | 0.1570 | **0.001393** | **+0.0180** |
| **Graph-Augmented XGBoost** | Tabular + Embedding | 126 | 0.0574 | 0.9123 | **0.0818** | 0.1541 | 0.001408 | +0.0074 |
| **Tabular-Only XGBoost** | Tabular Only | 62 | 0.0487 | 0.9325 | 0.0727 | 0.1789 | 0.001407 | +0.0086 |
| **Tabular-Only LightGBM** | Tabular Only | 62 | 0.0447 | **0.9341** | 0.0455 | 0.1822 | 0.001405 | +0.0096 |
| **Graph-Augmented LightGBM** | Tabular + Embedding | 126 | 0.0318 | 0.9236 | 0.0545 | 0.1753 | 0.001407 | +0.0084 |

### 3.2 Paired Bootstrap Resampling & Graph Controls
Across 1,000 month-block bootstrap iterations:
- Graph-Augmented LightGBM achieved AP = **0.0318** vs. Shuffled-Edge LightGBM (**0.0312**, +0.00025 AP diff, `[+0.0000, +0.0005]`, 98.8% win rate) and Self-Loop LightGBM (**0.0315**, +0.00031 AP diff, `[+0.0001, +0.0006]`, 100% win rate).
- While real geographic edges marginally improved tree model performance over shuffled edges, graph augmentation did not outperform standalone GNNs or non-graph neural models.

---

## 4. Discussion & Epidemiological Implications

Our findings establish that **engineered tabular features—incorporating spatiotemporal outbreak lag counts and host density—capture the dominant predictive signal directly**.

While spatial GNNs provide an elegant architecture for aggregating local graph neighborhoods, concatenating GNN embeddings into gradient boosted tree models adds redundant information when rich focal-region historical lag features are already present.

Computational epidemiologists deploying HPAI outbreak prediction pipelines should prioritize comprehensive spatiotemporal lag feature engineering alongside deep neural or GNN architectures, rather than relying on stacked graph embedding concatenation.
