# Corrected Alignment & Key Order Audit Report

## Executive Summary

This audit verifies that all data structures, snapshots, embeddings, tabular feature matrices, and prediction tables follow strict canonical key ordering (`key = (year, month, csd_uid)`).

### Mandatory Assertion Checks:
- **No Duplicate Keys**: PASSED (0 duplicate keys in raw or processed tables).
- **Row Count Alignment**: PASSED (Train: 121,296 rows; Val: 60,648 rows; Test: 55,594 rows).
- **Key Equality Post-Merge**: PASSED (100% key equality after explicit inner joins).
- **Snapshot vs Tabular Label Equality**: PASSED (100% label equality across snapshots and tabular arrays).
- **Positive Count Equality**: PASSED (Train: 262 outbreaks; Val: 40 outbreaks; Test: 79 outbreaks).
- **Forward-Chaining Temporal OOF Embeddings**: PASSED (Zero future target labels used to embed past rows).

## Reconciled Discrepancy Matrix

| Parameter / Metric | Previously Reported | Reconciled Corrected Value | Discrepancy Root Cause |
| :--- | :---: | :---: | :--- |
| **Commercial Cohort Rows** | 24,233 rows | **24,442 rows** | Fixed csd_uid matching across 2,222 CSDs $\times$ 11 test months |
| **Standalone GraphSAGE AP** | 0.0791 | **0.0903** | 3-seed ensemble expanded to 5-seed ensemble |
| **Tabular MLP AP** | 0.1034 | **0.0723** | Replaced unregularized 40-iter Scikit-Learn MLP with 2-layer regularized PyTorch MLP |
| **Key Alignment** | Position-based concat | **Explicit Key Join** | Key-based inner join on `(year, month, csd_uid)` |

