# Loss Ablation GraphSAGE: Prediction Alignment & QA Audit

**Date**: 2026-08-04

## 1. Alignment Checks

| Check | Result |
|---|---|
| Valid GNN/LR prediction source | `results/baselines/rolling_wildbird_monthly_snapshot_graphsage_vs_lr_predictions.csv` |
| wildbird_snapshot_gs_ensemble_prob present | PASS |
| wildbird_lr_prob present | PASS |
| Ablation probs missing after join | 0  PASS |
| RF probs missing after join | 0  PASS |
| XGB probs missing after join | 0  PASS |
| Duplicate (csd_uid, year, month) in merged | 0  PASS |
| Aligned total rows | 176,890 |

## 2. Feature Integrity
- 62 features asserted every fold: PASS

## 3. Fold Selection Decisions (No Test-Year Data Used)

| Fold | Selected Loss | wBCE Still Best? |
|---|---|---|
| 1 | `cb_focal_g1.0_b0.999` | NO |
| 2 | `focal_g1.0_a0.75` | NO |
| 3 | `focal_g2.0_a0.5` | NO |

## 4. Prediction Integrity Per Fold

All folds passed: constant-CSD < 10%, unique probs > 100.
