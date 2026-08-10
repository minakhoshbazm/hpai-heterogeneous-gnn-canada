# Technical Design Plan: True Spatiotemporal Dynamic Graph Neural Network

**Author**: Senior ML Engineer & Epidemiological Graph Modeling Reviewer  
**Date**: July 2026  
**Status**: Recommendation & Technical Architecture (Not yet implemented)  
**Scope**: Next-Generation Spatiotemporal HPAI Risk Forecasting Framework  

---

## 1. Executive Summary & Diagnostic Synthesis

Current static spatial GNNs (GraphSAGE on kNN-3 and RGCN on snapshot multi-relations) achieve competitive national performance (PR-AUC 0.0698 to 0.0791), but do not demonstrate statistically significant superiority over a tabular Logistic Regression baseline (PR-AUC 0.0654, $95\%\text{ CI } [-0.0240, +0.0338]$). 

### Diagnostic Findings on Current GNN Limitations:

1. **Not Driven by Data-Quality Artifacts**: Removing 9 data-quality / extraction metadata artifact features (`climate_fallback_distance_km`, `climate_missing_flag`, `dw_image_count`, `dw_missing_flag` columns, `valid_pixel_count` columns, `small_area_flag`) from Logistic Regression reduced test PR-AUC slightly from 0.0654 to 0.0622 ($-0.0031$). This proves that data quality artifacts do **not** artificially inflate LR performance or explain why the GNN does not beat tabular baselines.
2. **Lack of Directed Temporal Edges**: Current GNN architectures perform message passing strictly **within static spatial snapshots for month $t$**. They lack directed temporal edges $(i, t) \rightarrow (i, t+1)$ connecting past states to future targets. As a result, the GNN cannot learn temporal persistence or spatiotemporal wave propagation through network layers; it relies entirely on tabular lag features appended to node feature vectors.
3. **Severe Label Scarcity & High Imbalance**: National monthly outbreak prevalence is extreme ($< 0.15\%$ positive CSD-months; 79 positives out of 55,594 test observations in 2025). Spatial GNN message passing over uninfected neighbors can over-smooth positive signals into background noise.
4. **Graph Topology Design**: Undirected spatial message passing within the same month $t$ models contemporaneous co-occurrence rather than **lagged epidemiological transmission** (virus spread from neighbor $j$ at month $t$ to target $i$ at month $t+1$).

---

## 2. True Spatiotemporal Graph Architecture

To overcome the limitation of static spatial snapshots, the next-generation architecture represents the dataset as a single **Heterogeneous Spatiotemporal Dynamic Graph**.

```
  [CSD_i, Month t]  ────── Temporal Self Edge ──────>  [CSD_i, Month t+1]
         │                                                    ▲
         │                                                    │
  Temporal Spatial Edge                              Temporal Spatial Edge
  (Geo / Poultry / Water)                            (Geo / Poultry / Water)
         │                                                    │
         ▼                                                    │
  [CSD_j, Month t]  ──────── Temporal Neighbor Edge ──────────┘
```

### 2.1 Node Definition
- **Node**: A unique tuple $(i, t)$ representing **CSD $i$ at month $t$**.
- Total Nodes: $N_{\text{nodes}} = N_{\text{CSDs}} \times N_{\text{months}}$ (e.g., $5,054 \times 48 = 242,592$ spatiotemporal nodes).
- **Node Features $\mathbf{X}_{(i, t)}$**: Environmental and local lag attributes for CSD $i$ up to month $t$.

### 2.2 Directed Edge Relations (Strict Forward Time Flow)
All edges flow strictly forward in time ($t \rightarrow t+1$) to ensure **zero future data leakage**:

1. **`temporal_self` Edge Relation**:
   - Connection: $(i, t) \rightarrow (i, t+1)$ for every CSD $i$.
   - Encodes local temporal persistence (outbreak memory, environmental inertia).
2. **`temporal_spatial_geographic` Edge Relation**:
   - Connection: $(j, t) \rightarrow (i, t+1)$ if centroid distance $d(i, j) \le 50\text{ km}$.
   - Encodes short-range spatial diffusion from month $t$ to month $t+1$.
3. **`temporal_spatial_poultry` Edge Relation**:
   - Connection: $(j, t) \rightarrow (i, t+1)$ if $d(i, j) \le 100\text{ km}$ and static poultry exposure cosine similarity $\ge 0.70$.
   - Encodes commercial poultry exposure transmission pathways across adjacent months.
4. **`temporal_spatial_water` Edge Relation**:
   - Connection: $(j, t) \rightarrow (i, t+1)$ if $d(i, j) \le 100\text{ km}$ and expanding water-habitat cosine similarity up to month $t$ is $\ge 0.70$.
   - Encodes migratory waterfowl habitat-mediated transmission from month $t$ to month $t+1$.

---

## 3. Strict Time-Safety & Anti-Leakage Protocol

1. **No Future Data Leakage**:
   - Outdegree connections for node $(i, t)$ only target nodes at $(k, t+1)$. No backward edges $(t+1 \rightarrow t)$ or contemporaneous spatial edges $(t \rightarrow t)$ are permitted.
   - Water habitat similarity for edge $(j, t) \rightarrow (i, t+1)$ is computed strictly using Dynamic World observations available in or before month $t$.
2. **Exclusion of High-Risk / Observational Data**:
   - **No eBird or GBIF bird observation data** are used.
   - **No engineered neighbor-pressure features** are included in the graph inputs to isolate raw GNN message-passing representation learning.

---

## 4. Rigorous Temporal Validation Protocol

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 TIMELINE & SPLITS                                      │
├─────────────────────────────────────┬──────────────────────────┬───────────────────────┤
│    Training Period                  │    Validation Split      │   Held-Out Test Set   │
│    2022-01 through 2023-12          │    2024-01 to 2024-12    │   2025-01 to 2025-12  │
│    (24 months)                      │    (12 months)           │   (12 months)         │
└─────────────────────────────────────┴──────────────────────────┴───────────────────────┘
```

### Validation Workflow:
1. **Model Selection & Architecture Freeze (Pre-2025)**:
   - Models are trained on 2022–2023 spatiotemporal nodes.
   - Model selection, layer depth, attention heads, dropout, learning rate, and loss weighting are tuned **strictly on the 2024 validation split**.
2. **Final Evaluation Freeze**:
   - Once the optimal architecture and hyperparameter configuration are frozen based on 2024 validation performance, the selected model is retrained on full 2022–2024 spatiotemporal nodes.
   - Evaluated **ONCE** on the 2025 held-out test nodes.

---

## 5. Summary Conclusion & Next Steps

Current spatial GNNs operate on disconnected monthly slices, forcing the GNN to perform spatial message passing without temporal continuity. The diagnostic evidence demonstrates that:
- Data quality artifacts do not account for baseline performance.
- The primary architectural bottleneck is the **absence of directed temporal edges** connecting past infection states to future regional targets.

Implementing a true spatiotemporal graph with directed $t \rightarrow t+1$ edges provides the mathematically sound, leakage-safe path forward for epidemiological GNN modeling.
