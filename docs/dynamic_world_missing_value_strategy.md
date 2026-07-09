# Missing-Value Strategy for Dynamic World Monthly Land Cover Features

This document outlines the strategy for handling missing values in the Census Subdivision (CSD)-level monthly Dynamic World land cover features for Canada (2022-2025).

## 1. Rationale and Context

Unlike ERA5 climate data, which provides continuous gap-free model reanalysis, the **Dynamic World** dataset is derived from Sentinel-2 optical imagery. Consequently, missing CSD-month land cover probability values are expected and arise primarily from:
* **Persistent Cloud Cover:** Long periods of cloudiness (especially during the spring/autumn transitions) obscuring the ground.
* **Snow and Ice:** Heavy winter snow and ice that mask the underlying land features or trigger snow-mask filters.
* **Observation Frequency Gaps:** A lack of cloud-free Sentinel-2 overpasses within a given calendar month.

Because missingness in optical data has a strong temporal component (e.g., winter snow cover) and local geographic characteristics, a simple spatial buffer fallback (which we used for ERA5) is not the first choice. Instead, we propose a hybrid temporal-spatial hierarchy.

---

## 2. Ontario Pilot Missingness Context

During the post-processing QA of the Ontario 2022 pilot dataset (comprising 577 unique CSDs over 12 months):
* **Total Rows:** 6,924
* **Missing Rows (`dw_missing_flag = 1`):** 826 rows (~11.9% of the dataset)
* **Status:** All other QA checks (column types, probability ranges `[0, 1]`, unique constraint, and schema alignment) passed successfully. The ~11.9% missingness rate confirms the necessity of a robust missing-value imputation strategy prior to training GNN models.

---

## 3. Recommended V1 Imputation Hierarchy

Imputation will proceed sequentially for each missing CSD-month feature:

1. **Observed (Tier 1):** Preserve direct observed values where they exist.
2. **Temporal 3-Month Median (Tier 2):** If a CSD-month is missing, fill using the temporal median of the same CSD across a 3-month window: the previous month, the current month, and the next month (where values are available).
3. **Annual CSD Median (Tier 3):** If the value is still unresolved (e.g., due to consecutive months of missingness), fill using the annual median of the same feature for the same CSD and the same year.
4. **Province-Month Median (Tier 4):** If the value remains unresolved (e.g., in newly incorporated northern CSDs with zero observations in that year), fill using the median value of all CSDs in the same province for the same month.
5. **Unresolved (Tier 5):** If the value is still missing (e.g., due to a lack of data across the entire province), leave it as missing/null.

---

## 4. Pipeline Transparency and Tracking

To ensure transparency and allow downstream models to account for filled values, the imputation script will add three tracking columns:

* **`dw_missing_flag_original`:** A binary flag (0 or 1) indicating whether the feature was originally missing in GEE.
* **`dw_missing_flag_final`:** A binary flag (0 or 1) indicating whether the feature remains missing after all imputation steps have run.
* **`dw_fill_method`:** A categorical string indicating the imputation outcome.

### Allowed Values for `dw_fill_method`:
* `observed`: The value was directly extracted from GEE.
* `temporal_3month_median`: Filled using the 3-month temporal window median.
* `annual_csd_median`: Filled using the annual median for that CSD.
* `province_month_median`: Filled using the province-wide monthly median.
* `missing`: The value remains unresolved (null).

---

## 5. Methodological Guidelines

* **Target Leakage Prevention:** Do not use CFIA outbreak labels, node connections, or other target variables to perform imputation. The filling process must rely strictly on covariates.
* **Imputation Reporting:** The post-processing pipeline must report the total missingness rate (both counts and percentages) before and after running the imputation hierarchy.
* **Probability Constraint:** Any filled value must be clipped/verified to reside within the valid probability range of `[0.0, 1.0]`.
