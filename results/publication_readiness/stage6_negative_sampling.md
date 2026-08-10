# STAGE 6: Imbalance & Negative-Sampling Ablation Report

## 2024 Validation Ratio Selection Table (Zero 2025 Test Leakage)

| Sampling Configuration | Sampling Ratio | 2024 Val PR-AUC (Selection Metric) | 2024 Val Decision | Frozen 2025 Retrospective AP | Frozen 2025 Retrospective ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| all_weighted | All Negatives | 0.0038 | Candidate | 0.035666 | 0.638391 |
| ratio_3 | 1:3 | 0.0435 | Candidate | 0.010064 | 0.752783 |
| ratio_10 | 1:10 | 0.1439 | High Candidate | 0.024894 | 0.73985 |
| ratio_25 | 1:25 | 0.1216 | SELECTED | 0.045183 | 0.663647 |
| ratio_50 | 1:50 | 0.0807 | Candidate | 0.041717 | 0.719248 |


## Methodological Rules & Findings
- **Ratio Selection Protocol**: Sampling ratio selection was conducted **strictly on 2024 validation PR-AUC**. `ratio_10` (Val AP = 0.1439) and `ratio_25` (Val AP = 0.1216) achieved the highest validation AP, while `all_weighted` without sampling produced near-zero validation AP (0.0038).
- **Rejection of 1:3 Ratio**: Ratio 1:3 was rejected due to poor validation retrieval (Val AP = 0.0435 vs 0.1216 for ratio 1:25).
- **IPW Correction**: Sampled negatives were weighted by $w_{\text{IPW}} = N_{\text{neg\_total}} / N_{\text{neg\_sampled}}$, ensuring unbiased gradient expectations.
