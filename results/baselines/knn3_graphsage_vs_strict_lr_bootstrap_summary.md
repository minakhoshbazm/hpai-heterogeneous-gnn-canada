# Statistical Uncertainty Comparison: kNN-3 GraphSAGE vs. Strict Logistic Regression

**Evaluation Split**: 2025 Held-Out Test Set (55,594 rows, 79 positives)  
**Bootstrap Method**: 1,000 Cluster Bootstraps by CSD (`csd_uid` sampled with replacement)  
**Date**: 2026-07-30  

---

## 1. Primary Statistical Decision Statement

> [!IMPORTANT]
> **CONCLUSION**: **GraphSAGE is comparable to, not proven better than, Logistic Regression.**

---

## 2. Cluster Bootstrap Uncertainty Analysis (1,000 Resamples)

| Metric | Value |
|---|---|
| **Strict Logistic Regression PR-AUC** | **0.0654** |
| **kNN-3 GraphSAGE Ensemble PR-AUC** | **0.0791** |
| **Observed PR-AUC Difference (Delta)** | **+0.0137** |
| **95% Bootstrap Confidence Interval** | **[-0.0100, +0.0472]** |
| **Proportion P(GraphSAGE > Logistic Regression)** | **88.6%** |

---

## 3. Detailed Model Comparison Table

| Model Variant | PR-AUC | ROC-AUC | PR-AUC Diff vs. LR | Precision@50 | Recall@50 | Precision@100 | Recall@100 | Precision@200 | Recall@200 |
|---|---|---|---|---|---|---|---|---|---|
| **Strict Env + Lag Logistic Regression** | 0.0654 | 0.9158 | +0.0000 | 0.1000 | 0.0633 | 0.0700 | 0.0886 | 0.0650 | 0.1646 |
| **kNN-3 GraphSAGE (Seed 11)** | 0.0538 | 0.9231 | -0.0116 | 0.0800 | 0.0506 | 0.1100 | 0.1392 | 0.0850 | 0.2152 |
| **kNN-3 GraphSAGE (Seed 22)** | 0.0783 | 0.9214 | +0.0130 | 0.1400 | 0.0886 | 0.0900 | 0.1139 | 0.0750 | 0.1899 |
| **kNN-3 GraphSAGE (Seed 33)** | 0.0687 | 0.9261 | +0.0033 | 0.1600 | 0.1013 | 0.1400 | 0.1772 | 0.0950 | 0.2405 |
| **kNN-3 GraphSAGE Ensemble (3-Seed)** | **0.0791** | **0.9263** | **+0.0137** | **0.1600** | **0.1013** | **0.1300** | **0.1646** | **0.0900** | **0.2278** |


---

## 4. Key Findings & Protocol Compliance

1. **Ensemble Performance**: Averaging prediction probabilities across the three seeds yields a 2025 test PR-AUC of **0.0791**, exceeding individual seeds (which ranged from 0.0538 to 0.0783).
2. **Top-K Ranking Comparison**:
   - At Top-50: kNN-3 GraphSAGE Ensemble achieves Precision@50 of **0.1600** (detecting 8 positives) vs. LR Precision@50 of **0.1000** (detecting 5 positives).
   - At Top-200: GraphSAGE Ensemble achieves Recall@200 of **0.2278** vs. LR Recall@200 of **0.1646**.
3. **Statistical Significance Assessment**: Because the 95% cluster bootstrap confidence interval **[-0.0100, +0.0472]** contains zero (with GraphSAGE outperforming LR in **88.6%** of bootstrap iterations), the strict decision rule specifies:
   - **"GraphSAGE is comparable to, not proven better than, Logistic Regression."**

---

*Report generated automatically by `scripts/compare_knn3_graphsage_vs_strict_lr.py`.*
