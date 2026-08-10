# Implementation Plan: Local-Spatial Residual GraphSAGE

## Research Goal
Design, implement, evaluate, and document a new hybrid neural graph architecture provisionally named **Local-Spatial Residual GraphSAGE**. The model combines a local non-linear MLP branch with a spatial GraphSAGE branch to test the core hypothesis:
> *Preserving strong local non-linear feature interactions while separately aggregating neighboring-region spatial information will improve rare-outbreak risk ranking compared with either a standalone MLP or standalone GraphSAGE.*

---

## 1. Architectural Design & Hybrid Variants

### Variant A: Concatenation Fusion (`ConcatenationLocalSpatialGraphSAGE`)
- **Local Branch**: $\mathbf{h}^{\text{local}}_{i,t} = \text{MLP}_{\text{local}}(X_{i,t})$
  - Input: 62 numeric features $\rightarrow$ Dense(32) $\rightarrow$ ReLU $\rightarrow$ Dropout(0.2) $\rightarrow$ Dense(32)
- **Spatial Branch**: $\mathbf{h}^{\text{spatial}}_{i,t} = \text{GraphSAGE}(X_{i,t}, \mathcal{N}(i))$
  - Input: 62 numeric features $\rightarrow$ SAGEConv(62, 32) $\rightarrow$ ReLU $\rightarrow$ Dropout(0.2) $\rightarrow$ SAGEConv(32, 32)
- **Fusion Head**: $\mathbf{h}^{\text{fusion}}_{i,t} = [\mathbf{h}^{\text{local}}_{i,t} \mathbin{\Vert} \mathbf{h}^{\text{spatial}}_{i,t}]$ (64 dims) $\rightarrow$ Dense(32) $\rightarrow$ ReLU $\rightarrow$ Dropout(0.2) $\rightarrow$ Linear(1)

### Variant B: Gated Residual Fusion (`GatedLocalSpatialGraphSAGE`)
- **Gating Mechanism**: $g_{i,t} = \sigma\left(W_g [\mathbf{h}^{\text{local}}_{i,t} \mathbin{\Vert} \mathbf{h}^{\text{spatial}}_{i,t}] + b_g\right) \in (0, 1)^{32}$
- **Gated Combination**: $\mathbf{h}^{\text{gated}}_{i,t} = g_{i,t} \odot \mathbf{h}^{\text{local}}_{i,t} + (1 - g_{i,t}) \odot \mathbf{h}^{\text{spatial}}_{i,t}$
- **Prediction Head**: $\mathbf{h}^{\text{gated}}_{i,t} \rightarrow \text{Dense}(32) \rightarrow \text{ReLU} \rightarrow \text{Linear}(1)$
- **Gate Value Analysis**: Record gate distributions across positive vs negative CSD-months, recurrent vs emerging outbreaks, and high vs low poultry density.

---

## 2. Experimental Protocol & Non-Negotiable Rules

1. **Zero Test Leakage**: All hyperparameters, layer dimensions, learning rates, negative ratios (1:25), and Platt calibration parameters are selected strictly on 2024 validation data before evaluating on the 2025 retrospective test set.
2. **Multi-Seed Ensembling**: Trained across 5 independent random seeds (`SEEDS = [11, 22, 33, 44, 55]`). Probability predictions are the 5-seed ensemble average.
3. **Canonical Metrics**: Evaluated using `sklearn.metrics.average_precision_score` (AP), `roc_auc_score`, monthly macro Precision@K / Recall@K, and Platt-calibrated Brier Skill Score (BSS).
4. **Resampling Significance**: 1,000-iteration month-block bootstrap resamples used to compute 95% Confidence Intervals for AP differences and win rates.

---

## 3. Essential Ablation Matrix

| Model Variant | Local MLP Branch | Spatial Graph Branch | Fusion Type | Edge Index |
| :--- | :---: | :---: | :---: | :---: |
| **Tabular LR Baseline** | No | No | Linear | None |
| **Standalone MLP** | Yes | No | None | None |
| **Standalone GraphSAGE** | No | Yes | None | $k\text{NN}-3$ Spatial |
| **Concatenation Hybrid** | Yes | Yes | Concatenation | $k\text{NN}-3$ Spatial |
| **Gated Residual Hybrid** | Yes | Yes | Gated Sigmoid | $k\text{NN}-3$ Spatial |
| **Shuffled-Edge Hybrid Control** | Yes | Yes | Gated Sigmoid | Randomly Shuffled Edges |

---

## 4. Work Breakdown & File Delivery

```text
results/hybrid_local_spatial_graphsage/
├── configs/ (Model hyperparameters & setup configs)
├── checkpoints/ (Saved PyTorch model weights)
├── predictions/ (Raw & calibrated prediction CSV files)
├── metrics/ (CSV tables of all evaluation metrics)
├── figures/ (Publication-quality PDF/PNG plots)
├── tables/ (Markdown & CSV benchmark tables)
├── manifests/ (Dataset fingerprints & prediction manifests)
├── repository_audit.md
├── implementation_plan.md
└── final_experiment_report.md
```

---

## 5. Verification Plan

### Automated Tests
- Verification script asserting row-count consistency across all cohorts (55,594 Full Test, 49,181 GLW Presence, 24,442 Commercial Presence).
- Automated check ensuring zero target leakage in lag feature matrices.
- Metric assertions confirming monotonic Platt calibration.

### Manual Verification
- Review final benchmark decision against outcome criteria (Outcome A: Clear Success, Outcome B: Complementary Success, Outcome C: No Improvement).
