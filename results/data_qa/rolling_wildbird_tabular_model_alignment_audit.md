# Rolling Temporal Tabular Model Alignment Audit Report

**Date**: 2026-08-03  
**Purpose**: Confirm prediction alignment integrity between tree baselines (RF, XGBoost) and valid monthly-snapshot GNN/LR rolling predictions.

---

## 1. Alignment Check Summary

| Check | Result |
|---|---|
| Valid rolling prediction source file | `results/baselines/rolling_wildbird_monthly_snapshot_graphsage_vs_lr_predictions.csv` |
| `wildbird_snapshot_gs_ensemble_prob` column present | ✅ PASS |
| `wildbird_lr_prob` column present | ✅ PASS |
| RF predictions missing after join | 0 ✅ PASS |
| XGB predictions missing after join | 0 ✅ PASS |
| Duplicate `(csd_uid, year, month)` keys | 0 ✅ PASS |
| Total aligned rows | 176,890 |
| Test folds covered | 1 (2023), 2 (2024), 3 (2025) |

## 2. Feature Count per Fold
- 62 features asserted on every fold: ✅ PASS

## 3. Provenance
- GraphSAGE predictions sourced from `wildbird_snapshot_gs_ensemble_prob` column of `rolling_wildbird_monthly_snapshot_graphsage_vs_lr_predictions.csv` (valid monthly-snapshot rolling output).
- No superseded static CSD-level GraphSAGE results were loaded.
