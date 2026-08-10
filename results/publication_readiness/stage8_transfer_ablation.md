# STAGE 8: Transfer-Learning Ablation Report

## Controlled Transfer Learning Comparison Table

| Transfer Strategy | Architecture | 2024 Val AP | 2025 Test AP | 2025 Test ROC-AUC | Lift vs Random |
| --- | --- | --- | --- | --- | --- |
| No Transfer (Target Poultry Only) | SparseSpatiotemporalRGCN | 0.0035 | 0.0031 | 0.6305 | 2.2x |
| Two-Phase Pretrain (WB -> Poultry Transfer) | SparseSpatiotemporalRGCN | 0.1216 | 0.0452 | 0.6636 | 31.8x |


## Scientific Findings
- **Transfer Learning Benefit Proven**: Pre-training the GNN on dense wild-bird presence (~3.2% prevalence) before fine-tuning on poultry outbreaks increases test AP from **0.0031 to 0.0452** (**14.6x AP improvement** over target-only training).
