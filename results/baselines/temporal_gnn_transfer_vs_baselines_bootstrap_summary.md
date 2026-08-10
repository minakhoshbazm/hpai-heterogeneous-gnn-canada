# True Spatiotemporal Directed Relational GNN Transfer Bootstrap Comparison

## Pairwise Bootstrap Uncertainty Results (1,000 Resamples on 2025 Test Set)

| Comparison Baseline | Temporal GNN Transfer PR-AUC | Baseline PR-AUC | Observed Diff | 95% Percentile Confidence Interval | Resample Fraction (TGNN Better) | Decision Conclusion |
|---|---|---|---|---|---|---|
| **Strict Env + Lag LR** | **0.0655** | 0.0654 | **+0.0001** | **[-0.0176, +0.0228]** | 0.5990 (59.9%) | **comparable, not proven better** |
| **kNN-3 GraphSAGE Ensemble** | **0.0655** | 0.0791 | **-0.0136** | **[-0.0380, +0.0019]** | 0.0490 (4.9%) | **comparable, not proven better** |

---

## Technical Notes
- **Graph Design**: 237,538 CSD-month nodes ($5,054 \times 47$). Directed forward temporal edges ($t \rightarrow t+1$) across 4 relation types.
- **Transfer Learning**: Pretrained on wild-bird HPAI next-month targets prior to fine-tuning on poultry.
