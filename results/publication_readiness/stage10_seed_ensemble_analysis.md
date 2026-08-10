# STAGE 10: Fair Seed & Ensemble Analysis Report

## Multi-Seed & Ensemble Breakdown (2025 Test Set)

| Model | AP | ROC-AUC | Type |
| --- | --- | --- | --- |
| GraphSAGE Seed 11 | 0.0542 | 0.9258 | Single Seed |
| GraphSAGE Seed 22 | 0.0518 | 0.9284 | Single Seed |
| GraphSAGE Seed 33 | 0.0563 | 0.9184 | Single Seed |
| GraphSAGE 3-Seed Mean | 0.0541 | 0.9242 | Single Seed Mean |
| GraphSAGE 3-Seed Ensemble | 0.0791 | 0.9263 | Ensemble Average |
| R-GCN ratio_25 Seed 11 | 0.0055 | 0.6363 | Single Seed |
| R-GCN ratio_25 Seed 22 | 0.015 | 0.4389 | Single Seed |
| R-GCN ratio_25 Seed 33 | 0.0298 | 0.7439 | Single Seed |
| R-GCN ratio_25 3-Seed Mean | 0.0168 | 0.6064 | Single Seed Mean |
| R-GCN ratio_25 3-Seed Ensemble | 0.0452 | 0.6636 | Ensemble Average |


## Key Findings
- **Ensemble Advantage**: For both GNN architectures, ensembling 3 independent random seeds provides a substantial AP gain (**+0.0250 for GraphSAGE**, **+0.0284 for R-GCN**), confirming that seed averaging is essential for stabilizing GNN risk scores.
