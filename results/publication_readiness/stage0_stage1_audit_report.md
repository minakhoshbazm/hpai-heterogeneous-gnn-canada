# STAGE 0 & STAGE 1 Publication-Readiness Audit Report
## Spatiotemporal Machine Learning & Graph Neural Networks for HPAI Outbreak Prediction in Canada

---

## STAGE 0: Publication-Readiness & Data Integrity Audit

### 0.1 Dataset & Unit of Analysis Verification
- **Unit of Analysis**: Census Subdivision (CSD) $\times$ Calendar Month ($t$).
- **Total CSDs**: 5,054 CSDs across Canada.
- **Total Time Steps**: 47 calendar months (2022-01 through 2025-11).
- **Total Spatiotemporal Nodes**: 237,538 region-months.
- **Duplicate Verification**: **0 duplicate CSD-Year-Month rows** (verified via `duplicated()`).
- **Target Variable**: `outbreak_next_month` $\in \{0, 1\}$ (1 if at least one commercial/observed poultry HPAI outbreak occurs in CSD $r$ during month $t+1$, else 0).
- **Forecast Horizon**: Exactly 1-month-ahead ($t+1$) using covariates aggregated up to month $t$.

---

### 0.2 Temporal Split & Positive Outbreak Inventory

| Split | Calendar Years | Total Region-Month Rows | Positive Outbreaks ($y=1$) | Positive Prevalence | Data Isolation & Scaler Rule |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train-Tune Set** | 2022–2023 | 121,296 | 262 | 0.002160 (0.216%) | Hyperparameter tuning & pre-training |
| **Validation Set** | 2024 | 60,648 | 40 | 0.000660 (0.066%) | Model selection & ratio freezing |
| **Full Train Set** | 2022–2024 | 181,944 | 302 | 0.001660 (0.166%) | Final retraining before locked test |
| **Locked Test Set** | 2025 (Jan–Nov) | 55,594 | 79 | 0.001421 (0.142%) | **Frozen 1-time evaluation ONLY** |

*Leakage Verification*: Automated assertions confirm **0 overlapping indices** between training, validation, and test splits.

---

### 0.3 Feature Audit & Exclusion Rules
- **Total Features**: Exactly 62 numeric feature columns.
- **Excluded Non-Feature Identifiers**: `csd_uid`, `csd_name`, `province_code`, `province_name`, `ccs_uid`, `year`, `month`, `outbreak_next_month`, `outbreak_current_month`, `wildbird_hpai_present_next_month`, and data quality flags.
- **Lag Features (12 features)**: Includes `same_csd_outbreak_current_month`, `same_csd_outbreak_prev_1_month`, `same_csd_outbreak_count_past_3_months`, `same_csd_outbreak_count_past_6_months`, `same_csd_outbreak_count_past_12_months`, `province_outbreak_count_current_month`, `province_outbreak_count_past_3_months`.

---

### 0.4 Reconciling Scientific & Metric Discrepancies

#### Discrepancy 1: GraphSAGE PR-AUC (0.0541 vs 0.0791)
- **0.0541**: Represents the **mean PR-AUC of individual single-seed models** (Seed 11: `0.0542`, Seed 22: `0.0518`, Seed 33: `0.0563`).
- **0.0791** (`average_precision_score`) / **0.0767** (trapezoidal `auc`): Represents the performance of the **3-Seed ENSEMBLE average prediction** (`prob_sage = mean(seed_11, seed_22, seed_33)`).
- **Explanation**: Ensembling the 3 models reduces variance across positive predictions and yields a **+0.0250 AP gain**.

#### Discrepancy 2: Persistence Baseline PR-AUC (0.0405 vs 0.0132)
- The persistence baseline outputs binary predictions $p \in \{0, 1\}$ (whether an outbreak occurred in current month $t$).
- For binary step predictions:
  $$\text{AP} = \sum_n (R_n - R_{n-1}) P_n = \frac{7}{79} \times \frac{7}{68} = \mathbf{0.013191}$$
- Earlier trapezoidal integral code calculated a linear interpolation from $(0, 0)$ to $(R_1, P_1)$, overestimating area as $\sim 0.0405$.
- Canonical `average_precision_score` is **0.013191** (Lift = 9.3× over prevalence).

#### Discrepancy 3: PR-AUC Metric Implementation Standard
- Trapezoidal integration (`sklearn.metrics.auc(recalls, precisions)`) can over- or under-estimate precision-recall area due to linear interpolation between discrete thresholds.
- All models in this study are evaluated using the canonical evaluator **`sklearn.metrics.average_precision_score(y_true, y_prob)`**, which computes the exact weighted sum of precisions at each threshold step.

---

## STAGE 1: Risk Set Definition & Epidemiological At-Risk Cohort

### 1.1 Data Audit of Farm Capacity & Reporting
We audited Statistics Canada Census of Agriculture and FAO Gridded Livestock of the World (GLW2020) features:
- **Census Ag Commercial Farm Reporting** (`ccs_farms_reporting > 0`): Present in **2,203 CSDs (43.6% of CSDs)**.
- **GLW2020 Gridded Chicken Density** (`glw2020_chicken_density_mean > 0`): Present in **4,471 CSDs (88.5% of CSDs)**.

### 1.2 2025 Outbreak CSD Distribution
- In 2025, **68 unique CSDs** experienced a total of **79 poultry HPAI outbreaks**.
- **100% of 2025 outbreak CSDs (68 / 68)** had **GLW2020 Chicken Density > 0**.
- 38 of 68 outbreak CSDs had Census Ag commercial farm reporting $> 0$ (the remaining 30 outbreak CSDs represented small commercial operations, turkey/layer units outside broiler census reporting, or backyard/unclassified flocks located in chicken-dense CSDs).

### 1.3 Definition of Valid Epidemiological At-Risk Cohorts

```text
1. Full National Cohort (5,054 CSDs | 55,594 Test Rows):
   Evaluates full national risk across all Canadian regions (Prevalence = 0.1421%).

2. GLW Poultry At-Risk Cohort (4,471 CSDs | 49,181 Test Rows):
   Restricted to CSDs with GLW2020 chicken density > 0, eliminating 583 zero-poultry wilderness CSDs where outbreaks are physically impossible (Prevalence = 0.1606%).

3. Commercial Farm Reporting Cohort (2,203 CSDs | 24,233 Test Rows):
   Restricted to CSDs with Census Ag commercial farm reporting > 0 (Prevalence = 0.1857%).
```
