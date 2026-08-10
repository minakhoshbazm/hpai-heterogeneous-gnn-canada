# Regional GraphSAGE Validation & Edge Isolation Audit Report

**Date**: 2026-08-03  
**Audit Purpose**: Verify spatial graph edge isolation and confirm 0 cross-region edge leakage between training and held-out test regions.

---

## 1. Regional Graph Edge Isolation Summary

| Held-out Region | Final Train Edges Kept | Cross-Region Edges Removed (Train) | Test Edges Kept | Cross-Region Edges Removed (Test) | Constant CSDs % | QA Status |
|---|---|---|---|---|---|---|
| **Atlantic** | 289,872 | 1,116 | 17,292 | 341 | 0.00% | **PASS** |
| **Quebec_Ontario** | 219,996 | 1,512 | 38,522 | 462 | 0.11% | **PASS** |
| **Prairies** | 237,384 | 1,008 | 33,363 | 308 | 0.00% | **PASS** |
| **British_Columbia** | 293,364 | 612 | 16,379 | 187 | 0.00% | **PASS** |

- **Cross-Region Edge Leakage**: **0 cross-region edges** remained in any training, validation, or test graph snapshots.
- **Dynamic Node Features**: Within-CSD monthly score standard deviations were positive across all regions (0.00% constant CSDs).
- **Superseded Model Assurance**: Only valid Monthly-Snapshot GraphSAGE models were constructed and evaluated.

