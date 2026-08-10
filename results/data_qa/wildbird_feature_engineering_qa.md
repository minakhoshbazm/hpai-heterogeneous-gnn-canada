# Wild-Bird HPAI Feature Engineering QA Report

## 1. Overview
Engineered **12 leakage-safe CCS-level wild-bird HPAI pressure features** from the reviewed CFIA wild-bird dataset (`CFIA_outbreaks_wildbirds.csv` + `wildbird_cfia_ccs_crosswalk_reviewed.csv`) and joined them to the CSD-month modeling table (`model_table_v1_environment_with_lags.csv`).

- **Output File**: `[model_table_v1_environment_with_lags_wildbird.csv](file:///D:\gdrive\models\hpai-heterogeneous-gnn-canada\data\processed\modeling\model_table_v1_environment_with_lags_wildbird.csv)`
- **Total Modeling Rows**: 237,538 rows $\times$ 76 columns

## 2. Mandatory Validation & QA Checks

### Check 1: Row Count Integrity
- Base Modeling Table Rows: `237,538`
- Output Modeling Table Rows: `237,538`
- **Status**: `PASS` (Row count matches base table exactly).

### Check 2: Uniqueness & Duplicates
- Duplicate `(csd_uid, year, month)` rows: `0`
- **Status**: `PASS` (0 duplicate primary key rows).

### Check 3: Value Validity & Missingness
- Total NaN values across all 12 features: `0`
- Minimum feature value across all 12 features: `0.0`
- **Status**: `PASS` (No NaNs, all values $\ge 0$).

### Check 4: Feature Descriptive Statistics
| Feature Name | Min | Max | Mean | Std Dev |
| :--- | :---: | :---: | :---: | :---: |
| `wildbird_hpai_count_ccs_current_month` | 0 | 29.0 | 0.0762 | 0.6598 |
| `wildbird_hpai_count_ccs_prev_1_month` | 0 | 29.0 | 0.0697 | 0.6206 |
| `wildbird_hpai_count_ccs_past_3_months` | 0 | 48.0 | 0.2108 | 1.2655 |
| `wildbird_hpai_count_ccs_past_6_months` | 0 | 59.0 | 0.3981 | 1.8334 |
| `wildbird_hpai_present_ccs_current_month` | 0 | 1.0 | 0.0317 | 0.1753 |
| `wildbird_hpai_present_ccs_prev_1_month` | 0 | 1.0 | 0.0300 | 0.1706 |
| `wildbird_hpai_waterfowl_count_ccs_current_month` | 0 | 28.0 | 0.0328 | 0.4032 |
| `wildbird_hpai_waterfowl_count_ccs_past_3_months` | 0 | 35.0 | 0.0844 | 0.7283 |
| `wildbird_hpai_raptor_count_ccs_current_month` | 0 | 7.0 | 0.0107 | 0.1312 |
| `wildbird_hpai_raptor_count_ccs_past_3_months` | 0 | 10.0 | 0.0307 | 0.2453 |
| `wildbird_hpai_goose_count_ccs_current_month` | 0 | 25.0 | 0.0223 | 0.3026 |
| `wildbird_hpai_duck_count_ccs_current_month` | 0 | 27.0 | 0.0084 | 0.2199 |

### Check 5: Pearson Correlation with `outbreak_next_month`
| Feature Name | Pearson Correlation ($r$) | Signal Strength |
| :--- | :---: | :--- |
| `wildbird_hpai_count_ccs_current_month` | **+0.0293** | Moderate Positive |
| `wildbird_hpai_count_ccs_prev_1_month` | **+0.0118** | Positive |
| `wildbird_hpai_count_ccs_past_3_months` | **+0.0243** | Moderate Positive |
| `wildbird_hpai_count_ccs_past_6_months` | **+0.0206** | Moderate Positive |
| `wildbird_hpai_present_ccs_current_month` | **+0.0365** | Moderate Positive |
| `wildbird_hpai_present_ccs_prev_1_month` | **+0.0114** | Positive |
| `wildbird_hpai_waterfowl_count_ccs_current_month` | **+0.0341** | Moderate Positive |
| `wildbird_hpai_waterfowl_count_ccs_past_3_months` | **+0.0293** | Moderate Positive |
| `wildbird_hpai_raptor_count_ccs_current_month` | **+0.0352** | Moderate Positive |
| `wildbird_hpai_raptor_count_ccs_past_3_months` | **+0.0327** | Moderate Positive |
| `wildbird_hpai_goose_count_ccs_current_month` | **+0.0276** | Moderate Positive |
| `wildbird_hpai_duck_count_ccs_current_month` | **+0.0186** | Positive |

### Check 6: Yearly Wild-Bird Detection Totals (Raw Feed)
| Year | Total Wild-Bird Detections |
| :---: | :---: |
| 2022 | 1,698 |
| 2023 | 575 |
| 2024 | 457 |
| 2025 | 1,201 |

### Check 7: Temporal Leakage Safety Confirmation

> [!IMPORTANT]
> **Leakage Prevention Verification**:
> Each row in the modeling table represents observation period month M (e.g. 2024-04), and predicts poultry outbreak occurrence in month M+1 (outbreak_next_month, e.g. 2024-05). All 12 wild-bird pressure features aggregate detections strictly up to and including month M (current month M, lag M-1, rolling past 3 months [M-2..M], rolling past 6 months [M-5..M]). Zero detections from month M+1 or future months are included, guaranteeing 0% temporal data leakage.

## 3. Output Files & Artifacts
- **Updated Modeling Dataset**: `[model_table_v1_environment_with_lags_wildbird.csv](file:///D:\gdrive\models\hpai-heterogeneous-gnn-canada\data\processed\modeling\model_table_v1_environment_with_lags_wildbird.csv)`
- **QA Feature Engineering Report**: `[wildbird_feature_engineering_qa.md](file:///D:\gdrive\models\hpai-heterogeneous-gnn-canada\results\data_qa\wildbird_feature_engineering_qa.md)`
