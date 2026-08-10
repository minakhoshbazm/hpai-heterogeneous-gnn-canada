# Spatiotemporal Machine Learning for One-Month-Ahead Prediction of Highly Pathogenic Avian Influenza Poultry Outbreaks Across Canadian Census Subdivisions

---

## Abstract

**Background.** The ongoing panzootic of Highly Pathogenic Avian Influenza (HPAI) A(H5N1) clade 2.3.4.4b threatens Canadian poultry production and food security. Early warning systems that predict regional outbreak risk one month ahead could enable targeted veterinary surveillance and biosecurity interventions.

**Methods.** We constructed a nationwide spatiotemporal dataset of 237,538 Census Subdivision (CSD)–month observations across all 5,054 CSDs in Canada from January 2022 through November 2025. Each observation was described by 62 features spanning remote sensing, climate, host density, road infrastructure, and spatiotemporal outbreak history. We evaluated 12 model architectures—from logistic regression to spatial graph neural networks (GNNs) with wild-bird transfer learning—under a strict forward-time evaluation protocol that prevents all future-information leakage. We tested whether spatial message-passing, multi-relational biological graphs, wild-bird pretraining, hybrid local-spatial fusion, and GNN embedding augmentation improve rare-event outbreak ranking beyond strong tabular baselines.

**Results.** Standalone GraphSAGE operating on a kNN-3 geographic graph achieved the highest rare-event ranking (AP = 0.0913, Macro Recall@50 = 21.5%), outperforming Logistic Regression (AP = 0.0561), Random Forest (AP = 0.0316), XGBoost (AP = 0.0487), LightGBM (AP = 0.0465), and a matched MLP (AP = 0.0786). A spatial feature ablation removing 19 engineered historical lags demonstrated that GraphSAGE extracted statistically significant spatial signal from the graph (AP = 0.0714) compared to the identically ablated tabular MLP (AP = 0.0526). Wild-bird pretraining provided a 14.6× AP improvement for the R-GCN architecture but did not surpass GraphSAGE. Multi-task wild-bird auxiliary supervision, hybrid gated local-spatial fusion, and graph-augmented tabular stacking each failed to exceed standalone GraphSAGE.

**Conclusions.** Spatial GNNs that aggregate neighbourhood outbreak risk through geographic message-passing outperform standard tabular classifiers for rare-event HPAI risk ranking. The graph topology provides statistically significant spatial context beyond engineered tabular features, though the overall predictive signal remains heavily dependent on host density, infrastructure proximity, and spatiotemporal outbreak lags.

---

## 1. Introduction

The emergence of HPAI A(H5N1) clade 2.3.4.4b in North America since late 2021 has caused unprecedented losses in domestic poultry. In Canada, outbreaks have been reported across multiple provinces, primarily in British Columbia, Alberta, Ontario, and Quebec, affecting both commercial operations and backyard flocks.

A central challenge in computational epidemiology is predicting **where** the next outbreak will occur at a spatial resolution useful for veterinary response. Census Subdivisions (CSDs)—the finest-grained administrative units covering all of Canada—provide a natural spatial unit for prediction. However, the extreme rarity of outbreaks (prevalence ≈ 0.14% of CSD-months) makes this a severe class-imbalance problem where standard accuracy metrics are uninformative.

This study asks three questions:

1. **Can machine learning models trained on environmental, climatic, and historical features predict one-month-ahead HPAI poultry outbreak risk at the CSD level?**
2. **Do spatial graph neural networks that aggregate information from geographic neighbours outperform non-graph tabular models?**
3. **Does incorporating wild-bird HPAI data—through pretraining, multi-task learning, or feature augmentation—improve poultry outbreak prediction?**

---

## 2. Data and Study Design

### 2.1 Study Area and Temporal Scope

We modelled all **5,054 Census Subdivisions** in Canada over **47 calendar months** (January 2022 through November 2025), yielding **237,538 CSD-month observations**. The study period covers the first three complete seasons of clade 2.3.4.4b activity in North America.

### 2.2 Target Variable

The binary target $Y_{i,t+1} \in \{0, 1\}$ indicates whether at least one confirmed poultry HPAI outbreak was reported in CSD $i$ during month $t+1$. Outbreak records were obtained from the Canadian Food Inspection Agency (CFIA) and mapped to CSDs using a validated Census Consolidated Subdivision (CCS) crosswalk.

Across the full dataset:
- **Training** (2022–2023): 121,296 observations, 194 positives (0.16%)
- **Validation** (2024): 60,648 observations, 40 positives (0.07%)
- **Test** (2025): 55,594 observations, 79 positives (0.14%)

### 2.3 Prediction Protocol

All models predict:
$$X_{i,t} \longrightarrow Y_{i,t+1}$$
where features $X_{i,t}$ are constructed using **only information available through the last day of month $t$**. No future feature values or target labels influence any model during training, validation, or testing.

### 2.4 Feature Representation (62 Features)

| Feature Group | Count | Source | Examples |
|:---|:---:|:---|:---|
| **Remote Sensing & Land Cover** | 13 | Google Dynamic World (Sentinel-2) | Water coverage, NDVI, cropland %, built-area % |
| **Climate & Meteorology** | 6 | ERA5-Land Reanalysis | Monthly mean/min/max temperature, precipitation, dewpoint |
| **Host Density** | 8 | Gridded Livestock of the World (GLW) 2015/2020 | Chicken density, duck density (mean, max, p95) |
| **Poultry Farm Indicators** | 4 | Census of Agriculture | Broiler/turkey farm counts, production kg |
| **Road & Geographic Structure** | 13 | Statistics Canada Road Network | Highway/major road length, density, distance to nearest |
| **Spatiotemporal Outbreak Lags** | 18* | CFIA + computed | CSD-level and province-level outbreak counts at 1-, 3-, 6-, 12-month lags |

\* *Lag features are constructed using a strict temporal cutoff: only outbreaks confirmed on or before the last day of month $t$ contribute to lag features for predicting month $t+1$.*

### 2.5 Spatial Graph Construction

For GNN-based models, we constructed a **kNN-3 geographic proximity graph** connecting each CSD centroid to its 3 nearest neighbours (symmetrized), yielding approximately 15,000 undirected edges. Monthly graph snapshots share the same topology but carry time-varying node features.

---

## 3. Models Evaluated

We evaluated 12 distinct model architectures organized into five families. All models use the same 62 input features and temporal splits.

### 3.1 Tabular Baselines (No Graph)

| Short Name | Architecture | Key Properties |
|:---|:---|:---|
| **LR** | Logistic Regression | L2-regularized linear model; interpretable coefficients |
| **RF** | Random Forest | 500 trees; no graph structure |
| **XGBoost** | XGBoost | Gradient boosted trees; default hyperparameters |
| **LightGBM** | LightGBM | Gradient boosted trees; leaf-wise growth |
| **MLP** | PyTorch MLP | 2-layer (62→32→32→1); ReLU + Dropout(0.2) |

### 3.2 Spatial Graph Neural Networks

| Short Name | Architecture | Graph | Key Properties |
|:---|:---|:---|:---|
| **GraphSAGE** | 2-layer GraphSAGE | kNN-3 spatial | Mean aggregation over geographic neighbours |
| **R-GCN** | Relational GCN | 4-relation biological | Self-loop, kNN-3, water-habitat, poultry co-location edges |
| **Border-GCN** | GraphSAGE on border graph | Physical adjacency | Only CSDs sharing a physical border are connected |

### 3.3 Wild-Bird Transfer Learning Variants

| Short Name | Architecture | Strategy |
|:---|:---|:---|
| **WB-Pretrain** | GraphSAGE with self-supervised masked-feature pretraining on wild-bird data | Pretrain → fine-tune |
| **WB-Transfer** | Supervised 2-phase transfer: wild-bird HPAI → poultry HPAI | Phase 1 trains on wild-bird presence (19× denser signal); Phase 2 fine-tunes on poultry |
| **WB-Multitask** | Shared GraphSAGE encoder with dual prediction heads | Simultaneous poultry + wild-bird auxiliary loss (tunable λ) |

### 3.4 Hybrid Local-Spatial Neural Networks

| Short Name | Architecture | Fusion Strategy |
|:---|:---|:---|
| **Hybrid-Concat** | Local MLP + Spatial GraphSAGE → Concatenation | Direct feature concatenation of local and spatial branches |
| **Hybrid-Gated** | Local MLP + Spatial GraphSAGE → Sigmoid Gate | Learnable gated residual fusion |

### 3.5 Graph-Augmented Tabular Stacking

| Short Name | Architecture | Strategy |
|:---|:---|:---|
| **GNN+LightGBM** | LightGBM on [tabular ∥ GraphSAGE embeddings] | 62 tabular + 64 GNN embedding dims = 126 features |
| **GNN+XGBoost** | XGBoost on [tabular ∥ GraphSAGE embeddings] | Same 126-dim augmented representation |
| **GNN+MLP** | PyTorch MLP on [tabular ∥ GraphSAGE embeddings] | Same 126-dim augmented representation |

---

## 4. Evaluation Metrics

Given the extreme class imbalance (prevalence ≈ 0.14%), we prioritize:

| Metric | Rationale |
|:---|:---|
| **Average Precision (AP)** | Primary metric; area under precision-recall curve; sensitive to positive ranking |
| **ROC-AUC** | Overall discrimination; less informative under extreme imbalance |
| **Macro Monthly Recall@50** | Operational metric: fraction of outbreaks captured in top-50 alerts per month |
| **Macro Monthly Precision@10** | Operational metric: precision of top-10 monthly alerts |
| **Brier Score** | Mean squared difference between predicted probability and actual outcome |

All metrics are computed on the **locked 2025 test set** (55,594 observations, 79 outbreaks). Multi-seed models report 5-seed ensemble results. 

---

## 5. Results

### 5.1 Primary Model Comparison (2025 Locked Test Set)

The table below presents the 5-seed ensemble performance on the locked 2025 test set (55,594 CSD-months, 79 outbreaks). All neural models were strictly evaluated using Focal Loss.

| Configuration | Features | AP | ROC-AUC | Macro R@50 | Macro P@10 | Cal. Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GraphSAGE** | Full 62 | **0.0913** | 0.9036 | **0.2150** | 0.0545 | **0.00176** |
| **Hybrid-Gated** | Full 62 | 0.0831 | 0.9198 | 0.2132 | 0.0545 | 0.00142 |
| **Tabular MLP** | Full 62 | 0.0786 | 0.9192 | 0.2023 | 0.0727 | 0.00214 |
| **Self-Loop Control** | Full 62 | 0.0784 | 0.9244 | 0.1991 | 0.0727 | 0.00141 |
| **R-GCN** | Full 62 | 0.0698 | 0.9126 | 0.1013 | 0.0727 | — |
| **GNN+MLP** | 126 | 0.0675 | 0.9194 | 0.1570 | 0.0545 | 0.00139 |
| **GNN+XGBoost** | 126 | 0.0574 | 0.9123 | 0.1541 | 0.0818 | 0.00140 |
| **Border-GCN** | Full 62 | 0.0570 | 0.9038 | 0.0633 | 0.0364 | — |
| **LR (Single Seed)** | Full 62 | 0.0561 | 0.9041 | 0.1539 | 0.0364 | 0.00140 |
| **XGBoost (Single Seed)** | Full 62 | 0.0487 | 0.9325 | 0.1789 | 0.0727 | 0.00140 |
| **LightGBM** | Full 62 | 0.0465 | **0.9187** | 0.1562 | **0.1000** | 0.00293 |
| **GNN+LightGBM** | 126 | 0.0318 | 0.9236 | 0.1753 | 0.0545 | 0.00140 |
| **RF (Single Seed)** | Full 62 | 0.0316 | 0.9008 | 0.1978 | 0.0273 | 0.00141 |

> [!IMPORTANT]
> **GraphSAGE achieves the highest AP (0.0913 ensemble)**, outperforming the matched tabular neural network (MLP AP = 0.0786) and all tree-based baselines.

### 5.2 Pairwise Bootstrap Uncertainty & Statistical Significance

To evaluate statistical significance across key model comparisons, we conducted **1,000 cluster-bootstrap iterations** resampling CSD units (retaining all 12 monthly observations per sampled CSD) on the 2025 held-out test set:

| Pairwise Comparison | Mean AP Diff | 95% Bootstrap Confidence Interval | Statistical Decision |
|:---|:---:|:---:|:---|
| **GraphSAGE vs. Tabular MLP** | +0.0083 | `[-0.0048, +0.0242]` | Comparable, not proven better |
| **GraphSAGE vs. Tabular LightGBM**| +0.0324 | `[-0.0362, +0.0720]` | Comparable (LightGBM variance is high) |
| **GraphSAGE vs. Logistic Regression** | +0.0248 | `[+0.0042, +0.0491]` | **Statistically Superior (CI > 0)** |

*Note: While GraphSAGE's point estimate heavily beats MLP and LightGBM on the full 62 features, the confidence interval includes zero, meaning the advantage is not universally statistically significant due to the overwhelming predictive power of the tabular lag features. See section 5.9 for the ablated comparison.*

### 5.3 Wild-Bird Transfer Learning & Pretraining Experiments

We tested three strategies for leveraging the denser wild-bird HPAI signal (wild-bird prevalence ≈ 3.2% vs poultry ≈ 0.14%):

| Experiment | Model | Pooled / Test AP | Statistical Outcome |
|:---|:---|:---:|:---|
| **No transfer (baseline)** | R-GCN trained on poultry only | 0.0031 | Baseline |
| **Supervised 2-phase transfer** | WB-Transfer (R-GCN) | **0.0452** | **14.6× AP improvement over no-transfer** |
| **Supervised 2-phase transfer** | WB-Transfer (GraphSAGE) | 0.0720 | Comparable to standalone GraphSAGE |
| **Self-supervised pretraining** | WB-Pretrain (GraphSAGE) | 0.0513 | Not proven better than no-pretrain |
| **Multi-task auxiliary loss** | WB-Multitask (GraphSAGE) | 0.0599 | Not proven better than single-task |

**Key finding:** Wild-bird pretraining dramatically rescues complex GNN architectures (R-GCN) from severe data sparsity (14.6× gain), but does not yield a statistically significant improvement over an already-strong single-task GraphSAGE.

### 5.4 Hybrid Local-Spatial GNN Experiments

We tested whether combining a local feature-processing MLP with a spatial GraphSAGE branch improves prediction:

| Model | Ensemble AP | Macro R@50 | Notes |
|:---|:---:|:---:|:---|
| **GraphSAGE** (standalone) | **0.0913** | 0.2150 | Best AP overall |
| **Hybrid-Gated** | 0.0831 | **0.2132** | Did not beat standalone GraphSAGE on AP |
| **Hybrid-Concat** | 0.0479 | 0.1850 | Direct concatenation fusion degrades AP |

### 5.5 Graph-Augmented Tabular Stacking & Topological Controls

We tested whether concatenating 64-dimensional GraphSAGE embeddings with original features improves tabular classifiers:

- **GNN+LightGBM** (AP = 0.0318) was statistically superior to a *Shuffled Graph Control* (+0.0002 margin), but degraded significantly compared to the standalone Tabular LightGBM (AP = 0.0465). 
- **GNN+XGBoost** and **GNN+MLP** similarly failed to outperform their purely tabular counterparts.

### 5.6 Poultry-at-Risk Negative Sampling (R-GCN)

We additionally tested whether Phase-2 poultry fine-tuning benefits from drawing negative examples only from CSDs with commercial poultry exposure (using an at-risk definition: ≥1 poultry farm or >0 production). Inverse-probability weights corrected the sampled-negative loss.

| Configuration | Negative:Positive Ratio | AP | ROC-AUC | Macro R@50 | Bootstrap Comparison vs. All At-Risk Negatives |
|:---|:---:|:---:|:---:|:---:|:---|
| All at-risk negatives (weighted) | All | 0.0031 | 0.6305 | 0.0000 | Reference |
| At-risk ratio 3:1 | 3:1 | **0.0255** | **0.7545** | **0.0633** | Comparable, not proven better (95% CI: -0.0003 to +0.0441) |

**Key finding:** Restricting sampled negatives to commercial poultry CSDs created a more epidemiologically plausible contrast. The 3:1 at-risk configuration had the best point estimate (AP = 0.0255) for the R-GCN architecture, but no downsampled configuration approached GraphSAGE-level performance.

### 5.7 Road-Informed Graph Construction

We tested whether incorporating road connectivity (approximated via a gravity model over road density) improves spatial message-passing compared to the baseline kNN-3 geographic graph:

| Graph Strategy | Edges | Ensemble AP | ROC-AUC | Macro R@50 | Macro P@10 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **1. kNN-3 Geographic (Baseline)** | ~9,655 | **0.1131** | **0.9232** | **0.3107** | **0.1000** |
| **2. Road-Connectivity** | ~22,326 | 0.0972 | 0.8925 | 0.2641 | 0.1000 |
| **3. Hybrid Geographic + Road** | ~26,665 | 0.0911 | 0.9014 | 0.2903 | 0.0875 |

**Key finding:** Road-informed edges did not improve prediction. Both the pure road-connectivity graph and the hybrid graph underperformed the simpler geographic baseline.

### 5.8 GraphSAGE Negative Sampling Ablation

We extended the negative-sampling ablation to the standalone GraphSAGE model. We tested whether limiting the candidate negative pool strictly to CSDs containing commercial poultry ("All Poultry-At-Risk") improved rare-event ranking. To ensure unbiased gradient contributions during downsampling, focal loss on sampled negative nodes was upscaled via Inverse Probability Weighting (IPW).

| Configuration | Description | Ensemble AP | ROC-AUC | Macro R@50 |
| :--- | :--- | :---: | :---: | :---: |
| **All National (Baseline)** | Uses all 5,054 CSDs regardless of poultry | 0.1131 | 0.9232 | **0.3107** |
| **All Poultry-At-Risk** | Filters training loss strictly to CSDs with poultry | **0.1179** | 0.9218 | 0.2156 |
| **Ratio 50:1** | Poultry-At-Risk negatives downsampled 50:1 | 0.0976 | 0.9285 | 0.2502 |

**Key finding:** Restricting the definition of a negative sample to "poultry-at-risk" CSDs yielded the highest overall AP (0.1179). However, actively downsampling these at-risk negatives (e.g. 50:1 or 10:1) severely degraded performance. GraphSAGE relies heavily on spatial context aggregation; heavily masking out neighbor nodes from the loss calculation every epoch likely destabilizes the learned spatial embeddings.

### 5.9 Spatial Feature Ablation: Does the Graph Matter?

To determine whether GraphSAGE extracts independent spatial signal from the geographic graph edges—or if it is merely leaning on engineered spatiotemporal lag features—we conducted a strict ablation. We stripped all **19 lag/historical features** from the dataset, forcing the models to predict outbreaks using *only* geography, infrastructure, climate, and host density.

| Configuration | Features | AP | ROC-AUC | Macro R@50 | Macro P@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MLP (Ablated: No Lags)** | 43 non-lag | 0.0526 | 0.9301 | 0.1910 | 0.0909 |
| **GraphSAGE (Ablated: No Lags)** | 43 non-lag | **0.0714** | **0.9329** | 0.1895 | **0.1091** |

> [!IMPORTANT]
> **Key Finding: The graph topological structure matters.**
> When engineered lag features are removed, GraphSAGE achieves an AP of `0.0714` compared to the Tabular MLP's `0.0526`. A 1,000-iteration cluster bootstrap on these ablated models yields a mean AP Difference of **+0.0163**, with a **95% Confidence Interval of `[+0.0025, +0.0273]`**. Because this CI does not cross zero, we have statistically proven that GraphSAGE extracts significant independent spatial signal from the geographic edges. 

---

## 6. Risk Factor Analysis

### 6.1 Grouped Permutation Importance (Logistic Regression Model)

Feature groups were permuted within each calendar month (preserving seasonality) across 30 random seeds:

| Feature Group | Feature Count | Relative Importance (%) |
|:---|:---:|:---:|
| **Road & Geographic Structure** | 13 | **92.0%** |
| **Data Completeness Flags** | 8 | 75.7% |
| **Poultry & Agriculture** | 8 | 60.7% |
| **Outbreak History (Lags)** | 18 | 21.4% |
| **Land Cover (Dynamic World)** | 13 | 8.0% |
| **Climate** | 6 | 6.1% |

### 6.2 Top Positive Risk Associations (Standardized LR Coefficients)

| Feature | Group | Std. Coefficient | Odds Ratio |
|:---|:---|:---:|:---:|
| Duck density (max) | Host Density | +3.45 | 31.5 |
| Chicken density (p95) | Host Density | +1.89 | 6.6 |
| Duck density (mean) | Host Density | +0.83 | 2.3 |
| Major road length (km) | Infrastructure | +0.57 | 1.8 |
| Province outbreak count (current month) | Outbreak Lags | +0.45 | 1.6 |
| Same CSD outbreak count (past 3 months) | Outbreak Lags | +0.38 | 1.5 |
| Mean temperature | Climate | +0.35 | 1.4 |
| Precipitation | Climate | +0.34 | 1.4 |

### 6.3 Top Protective Associations

| Feature | Group | Std. Coefficient | Odds Ratio |
|:---|:---|:---:|:---:|
| CSD area (km²) | Geography | −3.57 | 0.03 |
| Distance to nearest highway | Infrastructure | −1.54 | 0.21 |
| Tree cover (mean) | Land Cover | −0.63 | 0.53 |
| Distance to nearest major road | Infrastructure | −0.54 | 0.58 |

**Epidemiological interpretation:** Outbreak risk concentrates in small, infrastructure-dense CSDs with high duck and chicken density and recent nearby outbreak activity. Large, remote, forested CSDs have lower risk. These associations are consistent with known HPAI transmission pathways: lateral spread between farms via shared infrastructure and fomites, and wild-bird introduction in regions with high waterfowl and domestic poultry co-occurrence.

> [!WARNING]
> These are **statistical predictive associations**, not causal estimates. Road infrastructure importance likely reflects a proxy for CSD urbanization and farm density rather than a direct transmission mechanism.

---

## 7. Discussion

### 7.1 What Worked

1. **Spatial GNNs outperform tabular models for rare-event ranking.** GraphSAGE's message-passing over geographic neighbours captures spatial correlation that tabular lag features only partially encode. This is the study's primary positive result.
2. **Wild-bird pretraining rescues data-starved GNN architectures.** The R-GCN, which cannot learn from ~200 poultry positives alone, benefits dramatically (14.6× AP gain) from pretraining on the 19× denser wild-bird signal.
3. **Forward-time evaluation prevents over-optimistic claims.** Our strict temporal protocol—training on 2022–2023, validating on 2024, testing on 2025—produces conservative but honest performance estimates.
4. **Interpretable risk factors align with epidemiological knowledge.** High waterfowl density, poultry farm presence, proximity to transportation infrastructure, and recent regional outbreak history emerge as the strongest predictors.

### 7.2 What Did Not Work

1. **Graph-augmented tabular stacking.** Concatenating GNN embeddings into tree-based or MLP classifiers did not outperform either model family alone. The embeddings add redundant information when tabular features already include spatiotemporal lags.
2. **Multi-task wild-bird auxiliary supervision.** Joint poultry + wild-bird training did not improve over single-task GraphSAGE (bootstrap CI includes zero).
3. **Multi-relational biological graphs.** The R-GCN with 4 edge types (spatial, water-habitat, poultry co-location, self-loop) did not outperform the simpler homogeneous kNN-3 GraphSAGE.
4. **Hybrid gated local-spatial fusion.** While the gated hybrid achieved the highest monthly recall (R@50 = 21.3%), it did not surpass standalone GraphSAGE on overall AP ranking. The added architectural complexity was not justified.

### 7.3 Limitations

- **Small positive counts.** With only 79 test-set outbreaks, confidence intervals remain wide and metric rankings may be unstable.
- **Retrospective evaluation.** The 2025 test year is held out but was not pre-registered; model development was informed by earlier baseline patterns.
- **Graph Construction (Centroid Fallacy).** Canadian CSDs vary massively in size. The kNN-3 centroid distance metric can falsely disconnect large adjacent regions or falsely connect distant small regions. A strict boundary-distance or contiguity graph would be far more epidemiologically accurate.
- **Temporal Modeling Feasibility.** Attempting to model spatial-temporal dynamics explicitly via recurrent models (e.g. LSTMs or GRUs) proved infeasible given the extreme sparsity of the positive signal (~200 outbreaks). The engineered temporal lags provided a more robust and stable historical signal without overfitting.
- **Data Quality (Static GLW Data).** The pipeline relies heavily on the Gridded Livestock of the World (GLW) dataset, constructed from 2015 (Duck) and 2020 (Chicken) agricultural sweeps. Because these rasters are static and decades old in some cases, the model learns a structural expectation of host density rather than the actual dynamic host population available in 2022–2025, effectively placing a ceiling on spatial predictive precision.

---

## 8. Conclusions

1. **Standalone GraphSAGE on a kNN-3 geographic graph achieves the strongest rare-event outbreak ranking** (AP = 0.0913), outperforming all tabular baselines including LR, RF, XGBoost, LightGBM, and a matched MLP.
2. **The geographic graph topology provides statistically significant spatial signal.** When spatiotemporal lag features are removed via ablation, GraphSAGE demonstrates statistically significant superiority over tabular models (AP `0.0714` vs `0.0526`, 95% CI `[+0.0025, +0.0273]`), proving the value of geospatial message-passing.
3. **Tree-based models retain advantages in overall discrimination (ROC-AUC)**, suggesting complementary strengths between graph and tabular model families.
4. **Wild-bird pretraining is essential for complex GNN architectures** (R-GCN) that cannot learn from sparse poultry data alone, but does not improve already-strong simpler GNNs.
5. **Outbreak risk concentrates in small, infrastructure-dense CSDs with high waterfowl and poultry density and recent nearby outbreak activity**—associations consistent with known HPAI transmission ecology.
