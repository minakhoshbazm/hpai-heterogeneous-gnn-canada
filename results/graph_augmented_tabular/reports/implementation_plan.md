# Implementation Plan: Graph-Augmented Tabular Risk Model

## Research Goal
Design, implement, evaluate, and document a new candidate model family: **Graph-Augmented Tabular Risk Model**. The model concatenates original 62 CSD-month tabular features $X_{i,t}$ with 64-dimensional GraphSAGE spatial embeddings $z_{i,t}$ to form an augmented 126-dimensional representation $\tilde{X}_{i,t} = [X_{i,t} \mathbin{\Vert} z_{i,t}]$, which is fed into strong tabular classifiers (LightGBM, XGBoost, MLP, Random Forest, Logistic Regression).

The central scientific question tested is:
> *Do learned GraphSAGE spatial embeddings add predictive information beyond original tabular features when used by strong tabular classifiers?*

---

## 1. Experimental Architecture & Workflow

```text
Original CSD-month features X(i,t) [62 dims]
        │
        ├───> GraphSAGE Spatial Encoder (csd_graph_knn_3_v1) ───> Embedding z(i,t) [64 dims]
        │                                                              │
        └───────────────────────────────┬──────────────────────────────┘
                                        │
                                 Concatenation
                                        │
                       Augmented Feature Matrix X_tilde(i,t) [126 dims]
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
      LightGBM Classifier       XGBoost Classifier          PyTorch MLP
             │                          │                          │
             └──────────────────────────┼──────────────────────────┘
                                        ▼
                           Out-of-Fold Platt Calibration
                                        ▼
                  Evaluation on Locked 2025 Test Set Across 3 Cohorts
```

---

## 2. Core Model Family & Baseline Matrix

| Model Identifier | Feature Input Set | Classifier Architecture | Embedding Origin | Primary Scientific Role |
| :--- | :--- | :--- | :--- | :--- |
| **Tabular LightGBM** | Original 62 Features | LightGBM (100 trees, lr=0.05) | None | **Primary Non-Graph Boosted Baseline** |
| **Tabular XGBoost** | Original 62 Features | XGBoost (100 trees, lr=0.05) | None | Non-Graph Tree Baseline |
| **Tabular PyTorch MLP** | Original 62 Features | 2-Layer PyTorch MLP (32-dim) | None | Non-Graph Neural Baseline |
| **Tabular Random Forest** | Original 62 Features | Scikit-Learn Random Forest | None | Non-Graph Ensemble Baseline |
| **Tabular Logistic Regression**| Original 62 Features | L2-Penalized Logistic Regression | None | Linear Baseline |
| **Graph-Augmented LightGBM** | 126 Features $[X_{i,t} \mathbin{\Vert} z_{i,t}]$ | LightGBM (100 trees, lr=0.05) | $k\text{NN}-3$ GraphSAGE (OOF) | **Primary Graph-Augmented Model** |
| **Graph-Augmented XGBoost** | 126 Features $[X_{i,t} \mathbin{\Vert} z_{i,t}]$ | XGBoost (100 trees, lr=0.05) | $k\text{NN}-3$ GraphSAGE (OOF) | Graph-Augmented Tree Model |
| **Graph-Augmented MLP** | 126 Features $[X_{i,t} \mathbin{\Vert} z_{i,t}]$ | 2-Layer PyTorch MLP (64-dim) | $k\text{NN}-3$ GraphSAGE (OOF) | Graph-Augmented Neural Model |
| **Shuffled-Edge LightGBM** | 126 Features $[X_{i,t} \mathbin{\Vert} z_{\text{shuff}}]$ | LightGBM | Shuffled Graph Realizations | **Topological Edge Control** |
| **Self-Loop LightGBM** | 126 Features $[X_{i,t} \mathbin{\Vert} z_{\text{self}}]$ | LightGBM | Self-Loop Only GraphSAGE | Disconnected Graph Control |

---

## 3. Stacking & Overfitting Protection Protocol

To prevent downstream classifiers from overfitting to training embeddings $z_{i,t}$, we enforce **Temporal Out-of-Fold (OOF) Embedding Generation**:
1. **Fold 1**: Train GraphSAGE Encoder $E_{2022}$ on 2022 data $\rightarrow$ generate embeddings $z_{2023}^{\text{OOF}}$ for 2023 training rows.
2. **Fold 2**: Train GraphSAGE Encoder $E_{2023}$ on 2023 data $\rightarrow$ generate embeddings $z_{2022}^{\text{OOF}}$ for 2022 training rows.
3. **Full Encoder**: Train GraphSAGE Encoder $E_{\text{Full}}$ on complete 2022–2023 training set (early stopping on 2024 val AP) $\rightarrow$ generate 2024 validation embeddings $z_{2024}$ and 2025 test embeddings $z_{2025}$.
4. Downstream classifiers (LightGBM, XGBoost, MLP) are trained on $[X_{\text{train}} \mathbin{\Vert} z_{\text{train}}^{\text{OOF}}]$ and validated on $[X_{\text{val}} \mathbin{\Vert} z_{\text{val}}]$.

---

## 4. Multi-Graph Topology Comparison

We will compare 4 graph topologies for GraphSAGE embedding extraction (selected using 2024 validation AP):
1. **$k\text{NN}-3$ Geographic Centroid Graph** (`csd_graph_knn_3_v1.csv`) [Primary]
2. **Shared-Border Spatial Adjacency Graph** (`csd_graph_border_only_v1.csv`)
3. **Border + $k\text{NN}-5$ Graph** (`csd_graph_border_plus_knn_5_v1.csv`)
4. **Commercial Poultry Density Similarity Graph** (`csd_relation_poultry_exposure_similarity_v1.csv`)

---

## 5. Controlled Edge Ablation & Multi-Realization Shuffled Control

Generate **10 independent shuffled-edge graph realizations**. For each shuffled realization:
- Extract GraphSAGE embeddings $z_{\text{shuff}}^{(r)}$
- Train downstream LightGBM / XGBoost on $[X_{i,t} \mathbin{\Vert} z_{\text{shuff}}^{(r)}]$
- Compute mean, standard deviation, and 95% CIs across all 10 realizations.

---

## 6. Verification Plan & Statistical Protocol

### Automated Verification
- Assert that augmented matrices $\tilde{X}_{i,t}$ have exactly 126 columns and match raw tabular row order 1-to-1.
- Assert zero label leakage across temporal splits.
- Assert exact row-count alignment across 3 cohorts (Full National: 55,594 rows; GLW Presence: 49,181 rows; Primary Commercial: 24,442 rows).

### Statistical Testing
- 1,000-iteration paired month-block bootstrap resampling computing 95% CIs and two-sided p-values for:
  - Graph-Augmented LightGBM vs. Tabular-Only LightGBM
  - Graph-Augmented LightGBM vs. Standalone GraphSAGE
  - Graph-Augmented LightGBM vs. Gated Hybrid GraphSAGE
  - Real-Edge Graph-Augmented LightGBM vs. Shuffled-Edge LightGBM
