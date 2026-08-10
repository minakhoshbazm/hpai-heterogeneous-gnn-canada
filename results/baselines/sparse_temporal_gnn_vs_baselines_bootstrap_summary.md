# Sparse Temporal GNN (Focal Loss + Hard Negative Sampling) Bootstrap Comparison

## Pairwise Bootstrap Uncertainty Results (1,000 Resamples on 2025 Test Set)

| Comparison Baseline | Sparse Temporal GNN PR-AUC | Baseline PR-AUC | Observed Diff | 95% Percentile Confidence Interval | Resample Fraction (SGNN Better) | Decision Conclusion |
|---|---|---|---|---|---|---|
| **Strict Env + Lag LR** | **0.0481** | 0.0654 | **-0.0173** | **[-0.0638, +0.0273]** | 0.2810 (28.1%) | **comparable, not proven better** |
| **kNN-3 GraphSAGE Ensemble** | **0.0481** | 0.0791 | **-0.0310** | **[-0.0669, +0.0055]** | 0.0460 (4.6%) | **comparable, not proven better** |

---

## Technical Notes
- **Winning Graph Architecture**: `Candidate A (Self + kNN-3)` with 2 relation types.
- **Focal Loss**: Binary Focal Loss ($\gamma = 2.0, \alpha = 0.85$).
- **Hard Negative Sampling**: 20:1 ratio during training prioritizing poultry-producing CSDs, matching outbreak months, and geographic proximity to positive outbreak nodes.
- **Un-Sampled Evaluation**: Validation and test splits evaluated on 100% full imbalanced datasets.
