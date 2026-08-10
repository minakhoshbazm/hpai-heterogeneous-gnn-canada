# Final Benchmark Evaluation & Model Comparison Summary

## Executive Summary
This document synthesizes all **strictly audited and validated model results** for the HPAI spatio-temporal risk forecasting pipeline across Canada.

All models included in this official benchmark strictly enforce:
- **Strict Feature Audit**: Exactly **43 features** for Environment-Only models and **50 features** (43 environment + 7 approved lag features) for Environment + Lag models.
- **Strict Identifier Exclusion**: `ccs_uid`, `csd_uid`, all `uid` fields, and raw `outbreak_current_month` are completely excluded.
- **Zero Leakage**: Standard scaling fit strictly on training splits only. No test data used for hyperparameter tuning.

---

## 1. Primary National Model Benchmark (2025 Test Split: 55,594 Rows, 79 Positives, Prevalence = 0.001421)

| model_name | feature_count | pr_auc | roc_auc | precision_50 | recall_50 | precision_100 | recall_100 | precision_200 | recall_200 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Strict Env + Lag Logistic Regression | 50 | 0.0654 | 0.9158 | 0.1000 | 0.0633 | 0.0700 | 0.0886 | 0.0650 | 0.1646 | Primary National Benchmark (Best overall valid PR-AUC) |
| Strict Early-Stopping GraphSAGE (Mean) | 50 | 0.0541 | 0.9092 | 0.1333 | 0.0844 | 0.0933 | 0.1181 | 0.0783 | 0.1983 | Secondary model (Multiseed Mean +/- Std: PR-AUC 0.0541 +/- 0.0020) |
| Strict Environment-Only Logistic Regression | 43 | 0.0457 | 0.9004 | 0.0600 | 0.0380 | 0.0400 | 0.0506 | 0.0350 | 0.0886 | Baseline environment model without outbreak history (43 verified features) |
| Time-Persistence Baseline | 0 | 0.0405 | 0.7311 | 0.2000 | 0.1266 | 0.1000 | 0.1266 | 0.0750 | 0.1899 | Strong top-50 precision (0.20), lower overall PR-AUC ranking |

### National Performance Findings:
- **Primary Benchmark Winner**: **Strict Environment + Lag Logistic Regression** achieves the highest overall PR-AUC (**0.0654**, a **46.0x lift** over background prevalence).
- **Secondary Comparison Model**: **Strict Early-Stopping GraphSAGE** is a secondary GNN benchmark (**0.0541 +/- 0.0020** PR-AUC), demonstrating strong spatial graph structure learning but slightly lower overall PR-AUC ranking than weighted LR.
- **Time-Persistence Baseline**: Achieves strong top-50 precision (**0.2000**, detecting 10 of 79 positives in top 50 rows) because past outbreaks strongly predict immediate local recurrence, but has lower overall PR-AUC (**0.0405**) and ROC-AUC (**0.7311**).

---

## 2. Temporal & Regional Robustness Validation

### A. Rolling Temporal Validation (Strict Environment + Lag LR)

| fold | train_years | test_year | train_positives | test_positives | test_prevalence | pr_auc | pr_auc_lift | roc_auc | precision_50 | recall_50 | precision_100 | recall_100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [2022] | 2,023 | 190 | 72 | 0.0012 | 0.0649 | 54.6740 | 0.8730 | 0.1400 | 0.0972 | 0.0900 | 0.1250 |
| 2 | [2022, 2023] | 2,024 | 262 | 40 | 0.0007 | 0.0924 | 140.1074 | 0.9192 | 0.0800 | 0.1000 | 0.0500 | 0.1250 |
| 3 | [2022, 2023, 2024] | 2,025 | 302 | 79 | 0.0014 | 0.0654 | 46.0013 | 0.9158 | 0.1000 | 0.0633 | 0.0700 | 0.0886 |

- **Temporal Stability**: Mean PR-AUC across folds is **0.0742 +/- 0.0157** with an average PR-AUC lift of **80.26x** over annual prevalence.
- **Consistency**: 2025 Test PR-AUC (**0.0654**) is broadly consistent with earlier fold test years (2023: `0.0649`, 2024: `0.0924`). Year-to-year variation is driven by annual outbreak prevalence fluctuations rather than model instability.

### B. Regional Spatial-Temporal Validation (2025 Held-Out Test Blocks)

- **British Columbia (`BC`)**: Demonstrated the **highest observed regional transfer performance**, based on 17 positive test cases (PR-AUC = **0.1863** for Env+Lag LR, **0.1629** for GraphSAGE).
- **Prairies (`MB`, `SK`, `AB`)**: Exhibited **some regional transfer signal** (PR-AUC = **0.0266** for Env+Lag LR, 11.5x lift over prevalence).
- **Quebec & Ontario (`QC`, `ON`)**: Showed **limited regional transfer performance despite performance above background prevalence** (PR-AUC = **0.0059** for Env+Lag LR, 7.58x lift over prevalence).
- **Atlantic (`NL`, `PEI`, `NS`, `NB`)**: Results are **not reliably interpretable because there were only 5 positive test cases** (PR-AUC = **0.0097**).

---

## 3. Probability Interpretation & Guidance

| model_variant | pr_auc | roc_auc | brier_score | brier_skill_score | ece_quantile |
| --- | --- | --- | --- | --- | --- |
| Constant 2025 Prevalence Reference | 0.0014 | 0.5000 | 0.0014 | 0.0000 | 0.0000 |
| Uncalibrated LogisticRegression | 0.0659 | 0.9159 | 0.0746 | -51.5964 | 0.1790 |
| Platt OOF-Calibrated LogisticRegression | 0.0659 | 0.9159 | 0.0014 | 0.0228 | 0.0011 |

> [!WARNING]
> **PROBABILITY INTERPRETATION RULE**:
> - Platt probability calibration improves score scaling and achieves a positive Brier Skill Score (**+0.0228** vs constant prevalence), but high-risk predictions exhibit residual overconfidence/under-prediction in extreme tails.
> - Model outputs **MUST be used strictly as ranked risk-prioritization scores**, NOT as literal monthly outbreak probabilities for public health deployment.

---

## 4. Wild-Bird Data Audit Decisions

> [!CAUTION]
> **WILD-BIRD FEATURE EXCLUSION**:
> 1. **eBird Status & Trends**: Local eBird rasters exist ONLY for product year 2023. Static 2023 rasters are **retrospective only** and must never be repeated across 2022-2025 in primary forecast models.
> 2. **GBIF Waterfowl Occurrences**: Audit revealed a **96.7% data deficit in 2025** (41,442 records vs 1.25M/year in 2022-2024) caused by eBird batch ingest release cycles. Observations from month M are not available by month-end, creating severe lookahead bias and temporal leakage. **GBIF is strictly excluded from primary forecasting tables and models.**

---

## 5. Superseded Pre-Audit Results Note

> [!IMPORTANT]
> **SUPERSEDED RESULTS EXCLUDED**:
> Earlier runs of Environment + Lag Logistic Regression (`environment_with_lags_v1_metrics.csv`) and Early-Stopping GraphSAGE (`graphsage_v1_early_stopping_metrics.csv`) erroneously included `ccs_uid` (an integer identifier code) and raw `outbreak_current_month` (52 features total).
>
> All pre-audit results are **formally superseded and excluded** from this final official interpretation.
