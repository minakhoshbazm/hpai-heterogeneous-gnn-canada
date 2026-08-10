# Wild-Bird Monthly-Snapshot kNN-3 GraphSAGE Summary

**Architecture**: Monthly Spatio-Temporal Snapshot GraphSAGE (Dynamic Node Features per Month, Shared Spatial Edge Topology, No Temporal Collapse)  
**Training Period**: 2022-01 through 2024-12  
**Test Period**: 2025 (Held-out single evaluation)  
**Feature Count**: 62  

## Multi-Seed Metrics

| Seed | Selected Epoch | Test PR-AUC | Test ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
|---|---|---|---|---|---|---|---|---|---|
| 11 | 2 | 0.0542 | 0.9258 | 0.1600 | 0.1013 | 0.1100 | 0.1392 | 0.0650 | 0.1646 |
| 22 | 2 | 0.0460 | 0.9284 | 0.1000 | 0.0633 | 0.0900 | 0.1139 | 0.0550 | 0.1392 |
| 33 | 10 | 0.0558 | 0.9184 | 0.1200 | 0.0759 | 0.0800 | 0.1013 | 0.0750 | 0.1899 |
| **Ensemble** | 4 | **0.0738** | **0.9332** | **0.1600** | **0.1013** | **0.1100** | **0.1392** | **0.0700** | **0.1772** |

## Prediction Stats (Ensemble)
- Min Prob: 0.000000
- Max Prob: 1.000000
- Mean Prob: 0.176361
- Std Dev: 0.226714
- Unique Predictions: 55567
