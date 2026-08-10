# Wild-Bird Monthly-Snapshot GraphSAGE & Retrained LR Audit Report

**Date**: 2026-08-03  
**Audit Decision**: **PASSED**  

---

## 1. Audit Check Summary

| Check | Requirement | Result | Status |
|---|---|---|---|
| **Schema Integrity** | 55,594 rows, 5,054 CSDs, Months 1–11 (2025), 79 Positives | 55,594 rows, 5,054 CSDs, 79 Positives | **PASS** |
| **Key Uniqueness** | Zero duplicate `(csd_uid, year, month)` keys | 0 duplicates | **PASS** |
| **Cross-Model Alignment** | Exact row match & 0 target mismatches | 55,594 rows, 0 mismatches | **PASS** |
| **Monthly Prediction Variation** | Predictions vary dynamically across months for each CSD | GraphSAGE constant CSDs: **0 / 5,054 (0.00%)** | **PASS** |

---

## 2. Dynamic Monthly Variation Verification

Unlike previous static GraphSAGE evaluations where 100% of CSD predictions were static across all months, the **Monthly-Snapshot GraphSAGE** models each month using its actual dynamic feature vector:

- **GraphSAGE Total Unique Predictions**: 55,567
- **GraphSAGE Within-CSD Score Std Dev**: Median = 0.044724, Mean = 0.065338, Max = 0.356475
- **Retrained LR Total Unique Predictions**: 55,594

---

## 3. Final Model Comparison

| Model | Test PR-AUC | Test ROC-AUC | Precision @ 50 | Recall @ 50 | Precision @ 100 | Recall @ 100 | Precision @ 200 | Recall @ 200 |
|---|---|---|---|---|---|---|---|---|
| **Final Retrained Wild-Bird LR** | 0.0679 | 0.9125 | 0.1000 | 0.0633 | 0.0800 | 0.1013 | 0.0650 | 0.1646 |
| **Wild-Bird Monthly-Snapshot GraphSAGE (Ensemble)** | **0.0738** | **0.9332** | **0.1600** | **0.1013** | **0.1100** | **0.1392** | **0.0700** | **0.1772** |

---

## 4. 1,000 CSD-Cluster Bootstrap Comparison

- **Comparator**: Final Retrained Wild-Bird Logistic Regression
- **Observed Mean PR-AUC Difference**: +0.0095
- **95% Percentile Confidence Interval**: [-0.0142, +0.0434]
- **GraphSAGE Win Rate**: 75.3%
- **Statistical Decision**: **COMPARABLE, NOT PROVEN BETTER**

---

## 5. Formal Scientific Statement

> [!IMPORTANT]
> **SUPERSEDED METHODOLOGY NOTICE**:  
> All previous static CSD-level GraphSAGE results (including prior reported PR-AUC values of `0.0999` and `0.0901`) were derived from a static graph construction that overwrote monthly node feature vectors, resulting in identical predictions across months for each CSD. **Those prior GraphSAGE results are officially SUPERSEDED and MUST NOT be used for manuscript claims or baseline comparisons.**  
> 
> The **Monthly-Snapshot GraphSAGE** (PR-AUC = `0.0738`) and **Retrained Wild-Bird LR** (PR-AUC = `0.0679`) represent the valid, rigorous spatio-temporal benchmarks.

