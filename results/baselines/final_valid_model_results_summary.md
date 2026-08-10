# Final Valid Model Results Summary & Research Decision Record

## National Model Benchmarks (2025 Test Split)

| Model Name | Model Type | PR-AUC | PR-AUC Std | ROC-AUC | Precision@50 | Recall@50 | Status |
|---|---|---|---|---|---|---|---|
| **kNN-3 GraphSAGE Ensemble** | Spatial Graph GNN | **0.0791** | — | 0.9168 | **0.2000** | **0.1266** | Valid National Baseline |
| **Relational GNN Ensemble** | Multi-Relation RGCN | **0.0698** | — | **0.9126** | 0.1600 | 0.1013 | **Primary Valid Relational Result** |
| **Strict Env + Lag LR** | Tabular Linear Baseline | **0.0654** | — | 0.9084 | 0.1200 | 0.0759 | Valid Linear Baseline |
| **Relational GNN (Multiseed Mean)** | Multi-Relation RGCN | **0.0648** | $\pm$ 0.0084 | 0.9076 | 0.1400 | 0.0886 | Valid Relational Mean |
| **Strict MLP (No Graph)** | Tabular Neural Net | **0.0601** | $\pm$ 0.0209 | 0.9048 | 0.1200 | 0.0759 | Valid Non-Graph Neural Net |
| **Border-Only GraphSAGE** | Spatial Contiguity GNN | **0.0570** | $\pm$ 0.0084 | 0.9038 | 0.1000 | 0.0633 | Valid Spatial Baseline |
| *Superseded RGCN (Static Aggregate)* | *Multi-Relation RGCN* | *0.0838* | *—* | *0.9174* | *0.2000* | *0.1266* | **SUPERSEDED** *(Future Covariate Leakage)* |


## Pairwise Cluster-Bootstrap Uncertainty Results (1,000 Resamples)

| Pairwise Comparison | Observed PR-AUC Diff | 95% Percentile CI | Resample Fraction (RGCN Better) | Formal Conclusion |
|---|---|---|---|---|
| **Time-Safe Relational GNN vs. Strict Env + Lag LR** | +0.0045 | [-0.0240, +0.0338] | 0.6760 (67.6%) | **comparable, not proven better** |
| **Time-Safe Relational GNN vs. kNN-3 GraphSAGE Ensemble** | -0.0092 | [-0.0303, +0.0076] | 0.1490 (14.9%) | **comparable, not proven better** |

*Note: In both comparisons, the 95% confidence interval includes zero; therefore, performance is formally designated as comparable, not proven better. Resample fractions describe bootstrap sample variation and are not probabilities.*



## Strict Regional Spatial-Temporal Validation Results

| Regional Block | Provinces | Test CSDs | Test Outbreaks ($N_{\text{pos}}$) | RGCN PR-AUC (Mean $\pm$ Std) | ROC-AUC | Strict LR PR-AUC | kNN-3 GraphSAGE PR-AUC | Evaluation Note |
|---|---|---|---|---|---|---|---|---|
| **British Columbia** | BC (59) | 751 | 17 | **0.1750 $\pm$ 0.0125** | 0.8937 | **0.1863** | 0.1748 | Valid regional evaluation |
| **Prairies** | MB (46), SK (47), AB (48) | 1,613 | 41 | **0.0275 $\pm$ 0.0065** | 0.8438 | 0.0266 | **0.0298** | Valid regional evaluation |
| **Quebec & Ontario** | QC (24), ON (35) | 1,859 | 16 | **0.0096 $\pm$ 0.0029** | 0.8786 | 0.0059 | **0.0125** | Valid regional evaluation |
| **Atlantic** | NL (10), PEI (11), NS (12), NB (13) | 831 | 5 | **0.0065 $\pm$ 0.0009** | 0.7868 | 0.0097 | **0.0152** | **UNRELIABLE** ($N_{\text{pos}} = 5 < 10$) |


## Key Scientific Guidelines
1. **Superseded Benchmark**: The earlier 0.0838 RGCN result relied on static aggregation over 2022-2024 which exposed future Dynamic World covariates to early 2022 training snapshots. It is preserved for historical comparison but **superseded** for forecasting.
2. **Valid Benchmark**: The snapshot-specific time-safe Relational GNN (PR-AUC 0.0698) is the official valid relational result.
3. **No Proven Superiority**: Bootstrap 95% confidence intervals contain zero relative to both Logistic Regression (+0.0045, 95% CI [-0.0240, +0.0338]) and kNN-3 GraphSAGE (-0.0092, 95% CI [-0.0303, +0.0076]). Performance is formally **comparable, not proven better**.
4. **2025 Test Lock**: 2025 test data has been accessed during exploratory validation; no further hyperparameter, epoch, or graph design tuning will use 2025.

---
*Report generated automatically by `scripts/summarize_final_valid_model_results.py`.*
