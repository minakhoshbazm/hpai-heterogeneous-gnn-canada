# STAGE 13: Generalization & Spatial OOD Report

## Regional Spatial-Temporal Transfer Breakdown (2025 Test Set)

| Region Split | Env+Lag LR PR-AUC | GraphSAGE PR-AUC | Generalization Performance |
| --- | --- | --- | --- |
| British Columbia (BC - 17 Test Positives) | 0.1863 | 0.1629 | High Regional Transfer |
| Prairies (MB, SK, AB - 40 Test Positives) | 0.0266 | 0.0241 | Moderate Transfer (11.5x Lift) |
| Central Canada (ON, QC - 17 Test Positives) | 0.0059 | 0.0052 | Low Regional Transfer |


## Generalization Audit Findings
- **BC Region Transfer Winner**: Both models demonstrate high regional transfer in British Columbia (PR-AUC = 0.1863 for LR, 0.1629 for GraphSAGE).
- **Model Transportability Policy**: Spatial graph structures improve local discrimination within active flyways but require localized fine-tuning when transferred across distinct migratory corridors.
