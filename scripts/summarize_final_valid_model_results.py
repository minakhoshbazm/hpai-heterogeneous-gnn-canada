#!/usr/bin/env python3
"""
scripts/summarize_final_valid_model_results.py

Assembles and outputs the final research decision summary table of all valid national
and regional model results for HPAI risk prediction across Canada.

Outputs:
- results/baselines/final_valid_model_results_summary.csv
- results/baselines/final_valid_model_results_summary.md
"""

import os
import pandas as pd
import numpy as np

BASELINES_DIR = 'results/baselines'
CSV_OUT = os.path.join(BASELINES_DIR, 'final_valid_model_results_summary.csv')
MD_OUT = os.path.join(BASELINES_DIR, 'final_valid_model_results_summary.md')

def main():
    print("==========================================================================================")
    print("        FINAL VALID MODEL RESULTS SUMMARY & RESEARCH DECISION RECORD")
    print("==========================================================================================")

    # 1. National Model Benchmarks Summary
    national_records = [
        {
            'model_name': 'kNN-3 GraphSAGE (3-Seed Ensemble)',
            'model_type': 'Spatial Graph GNN',
            'pr_auc': 0.0791,
            'pr_auc_std': 0.0000,
            'roc_auc': 0.9168,
            'precision_50': 0.2000,
            'recall_50': 0.1266,
            'validity_status': 'VALID PRIMARY GRAPH BASELINE',
            'notes': 'Selected spatial graph baseline on 2025 test set.'
        },
        {
            'model_name': 'Relational GNN (3-Seed Ensemble)',
            'model_type': 'Multi-Relation Biologically Informed RGCN',
            'pr_auc': 0.0698,
            'pr_auc_std': 0.0000,
            'roc_auc': 0.9126,
            'precision_50': 0.1600,
            'recall_50': 0.1013,
            'validity_status': 'VALID PRIMARY RELATIONAL RESULT',
            'notes': 'Snapshot-specific time-safe water habitat edges (expanding window <= M).'
        },
        {
            'model_name': 'Strict Environment + Lag Logistic Regression',
            'model_type': 'Tabular Linear Baseline',
            'pr_auc': 0.0654,
            'pr_auc_std': 0.0000,
            'roc_auc': 0.9084,
            'precision_50': 0.1200,
            'recall_50': 0.0759,
            'validity_status': 'VALID LINEAR BASELINE',
            'notes': 'Strict 50-feature set (43 env + 7 approved lag features).'
        },
        {
            'model_name': 'Relational GNN (Multiseed Mean)',
            'model_type': 'Multi-Relation Biologically Informed RGCN',
            'pr_auc': 0.0648,
            'pr_auc_std': 0.0084,
            'roc_auc': 0.9076,
            'precision_50': 0.1400,
            'recall_50': 0.0886,
            'validity_status': 'VALID RELATIONAL MEAN',
            'notes': 'Mean performance across seeds [11, 22, 33].'
        },
        {
            'model_name': 'Strict MLP (No Graph)',
            'model_type': 'Tabular Neural Network',
            'pr_auc': 0.0601,
            'pr_auc_std': 0.0209,
            'roc_auc': 0.9048,
            'precision_50': 0.1200,
            'recall_50': 0.0759,
            'validity_status': 'VALID NON-GRAPH NEURAL NET',
            'notes': 'Multilayer perceptron on strict 50-feature set.'
        },
        {
            'model_name': 'Border-Only GraphSAGE',
            'model_type': 'Spatial Contiguity Graph GNN',
            'pr_auc': 0.0570,
            'pr_auc_std': 0.0084,
            'roc_auc': 0.9038,
            'precision_50': 0.1000,
            'recall_50': 0.0633,
            'validity_status': 'VALID SPATIAL BASELINE',
            'notes': 'Adjacency graph of physically bordering CSDs.'
        },
        {
            'model_name': 'Superseded Relational GNN (Static Aggregate Graph)',
            'model_type': 'Multi-Relation RGCN',
            'pr_auc': 0.0838,
            'pr_auc_std': 0.0000,
            'roc_auc': 0.9174,
            'precision_50': 0.2000,
            'recall_50': 0.1266,
            'validity_status': 'SUPERSEDED FOR FORECASTING',
            'notes': 'Static water-habitat graph used future 2022-2024 Dynamic World features; preserved for sensitivity only.'
        }
    ]

    df_nat = pd.DataFrame(national_records)
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    df_nat.to_csv(CSV_OUT, index=False)

    # 2. Cluster Bootstrap Summary
    bootstrap_summary = """
## Pairwise Cluster-Bootstrap Uncertainty Results (1,000 Resamples)

| Pairwise Comparison | Observed PR-AUC Diff | 95% Percentile CI | Resample Fraction (RGCN Better) | Formal Conclusion |
|---|---|---|---|---|
| **Time-Safe Relational GNN vs. Strict Env + Lag LR** | +0.0045 | [-0.0240, +0.0338] | 0.6760 (67.6%) | **comparable, not proven better** |
| **Time-Safe Relational GNN vs. kNN-3 GraphSAGE Ensemble** | -0.0092 | [-0.0303, +0.0076] | 0.1490 (14.9%) | **comparable, not proven better** |

*Note: In both comparisons, the 95% confidence interval includes zero; therefore, performance is formally designated as comparable, not proven better. Resample fractions describe bootstrap sample variation and are not probabilities.*
"""

    # 3. Regional Validation Summary
    regional_summary = r"""
## Strict Regional Spatial-Temporal Validation Results

| Regional Block | Provinces | Test CSDs | Test Outbreaks ($N_{\text{pos}}$) | RGCN PR-AUC (Mean $\pm$ Std) | ROC-AUC | Strict LR PR-AUC | kNN-3 GraphSAGE PR-AUC | Evaluation Note |
|---|---|---|---|---|---|---|---|---|
| **British Columbia** | BC (59) | 751 | 17 | **0.1750 $\pm$ 0.0125** | 0.8937 | **0.1863** | 0.1748 | Valid regional evaluation |
| **Prairies** | MB (46), SK (47), AB (48) | 1,613 | 41 | **0.0275 $\pm$ 0.0065** | 0.8438 | 0.0266 | **0.0298** | Valid regional evaluation |
| **Quebec & Ontario** | QC (24), ON (35) | 1,859 | 16 | **0.0096 $\pm$ 0.0029** | 0.8786 | 0.0059 | **0.0125** | Valid regional evaluation |
| **Atlantic** | NL (10), PEI (11), NS (12), NB (13) | 831 | 5 | **0.0065 $\pm$ 0.0009** | 0.7868 | 0.0097 | **0.0152** | **UNRELIABLE** ($N_{\text{pos}} = 5 < 10$) |
"""

    # 4. Generate Full Markdown Summary
    md_content = rf"""# Final Valid Model Results Summary & Research Decision Record

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

{bootstrap_summary}

{regional_summary}

## Key Scientific Guidelines
1. **Superseded Benchmark**: The earlier 0.0838 RGCN result relied on static aggregation over 2022-2024 which exposed future Dynamic World covariates to early 2022 training snapshots. It is preserved for historical comparison but **superseded** for forecasting.
2. **Valid Benchmark**: The snapshot-specific time-safe Relational GNN (PR-AUC 0.0698) is the official valid relational result.
3. **No Proven Superiority**: Bootstrap 95% confidence intervals contain zero relative to both Logistic Regression (+0.0045, 95% CI [-0.0240, +0.0338]) and kNN-3 GraphSAGE (-0.0092, 95% CI [-0.0303, +0.0076]). Performance is formally **comparable, not proven better**.
4. **2025 Test Lock**: 2025 test data has been accessed during exploratory validation; no further hyperparameter, epoch, or graph design tuning will use 2025.

---
*Report generated automatically by `scripts/summarize_final_valid_model_results.py`.*
"""

    with open(MD_OUT, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(df_nat[['model_name', 'pr_auc', 'pr_auc_std', 'roc_auc', 'validity_status']].to_string(index=False))
    print(f"\nSaved CSV summary to: {CSV_OUT}")
    print(f"Saved MD summary to:  {MD_OUT}")

if __name__ == '__main__':
    main()
