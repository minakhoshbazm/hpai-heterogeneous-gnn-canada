# True Spatiotemporal Directed Relational GNN Bootstrap Comparison

## Pairwise Bootstrap Uncertainty Results (1,000 Resamples on 2025 Test Set)

| Comparison Baseline | Temporal GNN PR-AUC | Baseline PR-AUC | Observed Diff | 95% Percentile Confidence Interval | Resample Fraction (TGNN Better) | Decision Conclusion |
|---|---|---|---|---|---|---|
| **Strict Env + Lag LR** | **0.0510** | 0.0654 | **-0.0144** | **[-0.0693, +0.0232]** | 0.3850 (38.5%) | **comparable, not proven better** |
| **kNN-3 GraphSAGE Ensemble** | **0.0510** | 0.0791 | **-0.0281** | **[-0.0674, +0.0021]** | 0.0370 (3.7%) | **comparable, not proven better** |

---

## Technical Notes
- **Graph Design**: 237,538 CSD-month nodes ($5,054 \times 47$). Directed forward temporal edges ($t \rightarrow t+1$) across 4 relation types (`temporal_self`, `temporal_spatial_geographic`, `temporal_spatial_water`, `temporal_spatial_poultry`).
- **Anti-Leakage**: Zero backward or contemporaneous spatial edges. Water features for month $t$ constructed using expanding window $\le t$.
- **Strict Test Lock**: Model depth, early stopping, and hyperparameter selection tuned strictly on 2024 validation split prior to single 2025 test evaluation.
