# Official Research Decision Record: Time-Safe Relational GNN Evaluation

**Date**: July 2026  
**Status**: Finalized Research Decision Record  
**Scope**: HPAI Risk Forecasting under Strict 2025 Held-Out Test Lock  

---

## 1. Executive Summary & Policy Decisions

This document records the official research decision regarding the biologically informed multi-relation Relational Graph Neural Network (RGCN) baseline and establishes the valid benchmarking metrics for HPAI risk prediction across Canada.

### Key Decisions:
1. **Superseding of Earlier Static Graph Result**: The earlier RGCN 3-seed ensemble PR-AUC result of **0.0838** is explicitly **superseded** for strict temporal forecasting. That result relied on a `shared_water_habitat` graph constructed by averaging Dynamic World land cover features across the 2022–2024 historical reference period, which inadvertently allowed early 2022 training snapshots to access 2024 environmental covariates. The earlier artifact files remain preserved for sensitivity comparison, but **must not be used or cited as the primary forecasting result**.
2. **Designation of Primary Valid Relational Result**: The **snapshot-specific time-safe Relational GNN**—which constructs water-habitat edge relations dynamically for each month $M$ using expanding historical Dynamic World observations available in or before month $M$—is established as the official valid relational GNN benchmark.
3. **Strict Evaluation Freeze**: The 2025 held-out test split has been evaluated across multiple iterations during exploratory QA. **No further model selection, graph tuning, threshold adjustments, or epoch optimization using 2025 data will be performed.**

---

## 2. Official Valid National Model Benchmarks (2025 Test Set)

All models were evaluated on the unchanged 2025 held-out test set ($N = 55,594$ CSD-month observations, $N_{\text{pos}} = 79$ positive outbreak months).

| Model Variant | Graph Architecture | PR-AUC | PR-AUC Std | ROC-AUC | Precision@50 | Recall@50 | Status |
|---|---|---|---|---|---|---|---|
| **kNN-3 GraphSAGE (3-Seed Ensemble)** | kNN-3 Spatial Graph ($k=3$, $d\le 100$km) | **0.0791** | — | 0.9168 | **0.2000** | **0.1266** | Valid National Baseline |
| **Relational GNN (3-Seed Ensemble)** | Snapshot Time-Safe Multi-Relation Graph | **0.0698** | — | **0.9126** | 0.1600 | 0.1013 | **Primary Valid Relational Result** |
| **Strict Environment + Lag Logistic Regression** | No Graph (Tabular) | **0.0654** | — | 0.9084 | 0.1200 | 0.0759 | Valid Linear Baseline |
| **Relational GNN (Multiseed Mean)** | Snapshot Time-Safe Multi-Relation Graph | **0.0648** | $\pm$ 0.0084 | 0.9076 | 0.1400 | 0.0886 | Valid Relational Mean |
| **Strict MLP (No Graph)** | No Graph (Tabular Neural Net) | **0.0601** | $\pm$ 0.0209 | 0.9048 | 0.1200 | 0.0759 | Valid Non-Graph Neural Net |
| **Border-Only GraphSAGE** | Spatial Contiguity (Bordering CSDs) | **0.0570** | $\pm$ 0.0084 | 0.9038 | 0.1000 | 0.0633 | Valid Spatial Baseline |
| *Superseded RGCN (Static Aggregate)* | *Static 2022–2024 Water Habitat Graph* | *0.0838* | *—* | *0.9174* | *0.2000* | *0.1266* | **SUPERSEDED** *(Future Covariate Leakage)* |

---

## 3. Cluster-Bootstrap Uncertainty Analysis

To quantify statistical uncertainty while preserving spatial-temporal correlation structure, we conducted **1,000 cluster-bootstrap iterations** resampling CSD units with replacement (retaining all 12 monthly observations for each sampled CSD) on the 2025 test set.

| Comparison | Observed PR-AUC Diff | 95% Percentile Confidence Interval | Resample Fraction (RGCN Better) | Formal Decision Conclusion |
|---|---|---|---|---|
| **Time-Safe Relational GNN vs. Strict Env + Lag LR** | **+0.0045** | **[-0.0240, +0.0338]** | 0.6760 (67.6%) | **comparable, not proven better** |
| **Time-Safe Relational GNN vs. kNN-3 GraphSAGE Ensemble** | **-0.0092** | **[-0.0303, +0.0076]** | 0.1490 (14.9%) | **comparable, not proven better** |

> [!DECISION]
> **Statistical Decision Rule Applied**:
> Because the 95% confidence intervals for both pairwise comparisons include zero ($0 \in [-0.0240, +0.0338]$ and $0 \in [-0.0303, +0.0076]$), the performance of the Relational GNN is formally designated as **"comparable, not proven better"** than either baseline. Resample fractions reflect bootstrap variability across CSD resamples and are **not** interpreted as probabilities.

---

## 4. Strict Regional Spatial-Temporal Validation

The snapshot-specific time-safe Relational GNN was evaluated across four Canadian regional blocks on seeds `[11, 22, 33]`, enforcing **zero cross-region edges** and constructing graph relations independently within training/validation nodes and test nodes.

| Regional Block | Provinces Included | Test CSDs | Test Outbreaks ($N_{\text{pos}}$) | RGCN PR-AUC (Mean $\pm$ Std) | RGCN ROC-AUC | Strict LR PR-AUC | kNN-3 GraphSAGE PR-AUC | Reliability Assessment |
|---|---|---|---|---|---|---|---|---|
| **British Columbia** | BC (59) | 751 | 17 | **0.1750 $\pm$ 0.0125** | 0.8937 | **0.1863** | 0.1748 | Valid regional evaluation |
| **Prairies** | MB (46), SK (47), AB (48) | 1,613 | 41 | **0.0275 $\pm$ 0.0065** | 0.8438 | 0.0266 | **0.0298** | Valid regional evaluation |
| **Quebec & Ontario** | QC (24), ON (35) | 1,859 | 16 | **0.0096 $\pm$ 0.0029** | 0.8786 | 0.0059 | **0.0125** | Valid regional evaluation |
| **Atlantic** | NL (10), PEI (11), NS (12), NB (13) | 831 | 5 | **0.0065 $\pm$ 0.0009** | 0.7868 | 0.0097 | **0.0152** | **UNRELIABLE** ($N_{\text{pos}} = 5 < 10$) |

---

## 5. Cautious Scientific Interpretation & Guidelines

1. **No Claims of Proven Superiority**: The time-safe Relational GNN ensemble achieves competitive performance (PR-AUC 0.0698), but does not demonstrate statistically significant superiority over simpler linear models (Logistic Regression PR-AUC 0.0654, $95\%\text{ CI } [-0.0240, +0.0338]$) or distance-based graph architectures (kNN-3 GraphSAGE PR-AUC 0.0791).
2. **No Claims of Causal Biological Transmission Mechanisms**: The multi-relation graph incorporating geographic proximity, poultry exposure similarity, and water-habitat similarity provides a domain-informed relational structure. However, empirical performance does not confirm specific biological transmission pathways. Biologically informed relations are feasible and competitive, but their added predictive value remains uncertain with the available 2025 test data.
3. **Strict Evaluation Freeze**: Because the 2025 test split has been accessed during exploratory validation and QA, no further hyperparameter tuning, graph edge design changes, or epoch optimization using 2025 data will be permitted.

---

*Record created automatically by `scripts/summarize_final_valid_model_results.py` and approved under strict time-safety guidelines.*
