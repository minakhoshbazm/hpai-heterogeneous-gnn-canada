# Multi-Task GraphSAGE Audit Report

**Date**: 2026-08-04

## 1. Alignment Checks

| Check | Result |
|---|---|
| Valid GNN/LR source | `results/baselines/rolling_wildbird_monthly_snapshot_graphsage_vs_lr_predictions.csv` |
| wildbird_snapshot_gs_ensemble_prob present | PASS |
| Multitask probs missing after join | 0 PASS |
| RF probs missing after join | 0 PASS |
| XGB probs missing after join | 0 PASS |
| Duplicate (csd_uid, year, month) keys | 0 PASS |
| Aligned total rows | 176,890 |

## 2. Feature & Auxiliary Target Integrity

- 62 features asserted per fold: PASS
- `wildbird_hpai_present_next_month` never included in feature_cols: PASS
- `ccs_uid` never included in feature_cols: PASS
- Auxiliary labels constructed from matched CFIA detections only (lat-1 encoding, matched crosswalk): PASS

## 3. Temporal Leakage Checks

- dev_train max year < test year: PASS (all folds)
- final_train max year < test year: PASS (all folds)
- Lambda selected on dev-val primary PR-AUC only: PASS
- No test-year labels used for selection: PASS

## 4. Fold Lambda Selections

| Fold | Selected Lambda | Primary Val PR-AUC | Single-Task Baseline |
|---|---|---|---|
| 1 | `lambda=0.00` | 0.2058 | 0.2058 |
| 2 | `lambda=0.10` | 0.1454 | 0.1411 |
| 3 | `lambda=0.05` | 0.2639 | 0.2536 |

## 5. Prediction QA

All folds passed: constant-CSD pct < 10%, unique probs > 100.
