# Implementation Plan: Modeling Table V1 Assembly for CSD-Month HPAI Risk Prediction

This document outlines the plan for assembling the first modeling dataset at the CSD-month level to predict one-month-ahead Highly Pathogenic Avian Influenza (HPAI) risk in domestic poultry across Canada (2022-2025).

## 1. Scope and Modeling Unit

* **Modeling Unit:** CSD-month
* **Spatial Scope:** 5,054 provincial CSDs (excluding territories)
* **Temporal Scope:** 2022-2025 (48 months)
* **Expected National Row Count:** 242,592 rows (5,054 CSDs * 48 months)
* **Target Variable:** `outbreak_next_month` (1 if an outbreak is reported in the CSD in month $M+1$, 0 otherwise)

---

## 2. Input Datasets

We will merge the following compiled components:
1. **CFIA Outbreak Labels:** Spatiotemporal CSD-month outbreak logs, mapping to the one-month-ahead target.
2. **Static CSD Covariates:** Demographic, agricultural capacity, and geographic baseline features.
3. **ERA5-Land Climate Features (2022-2025):** Temperature, precipitation, and soil moisture features with multi-year fallback imputation.
4. **Dynamic World Land Cover (2022-2025):** Monthly land cover probability features with documented temporal/province median missing-value filling.
5. **Ontario 2022 GBIF Wild Bird Pilot:** Waterfowl activity indices and missing flags (Ontario 2022 only, as a proof-of-concept pilot).
6. **CSD Adjacency Graph:** Graph edges to define the spatial message-passing structure for GNN architectures.

---

## 3. Output Modeling Tables

We will assemble two versions of the modeling table:

### Table A: Baseline Environment-Only Dataset
* **Output Path:** `data/processed/modeling/model_table_v1_environment_only.csv`
* **Features Included:** Static CSD covariates + ERA5 climate + Dynamic World land cover.
* **Coverage:** National (5,054 CSDs, 2022-2025).
* **Purpose:** Serves as the primary baseline for training spatiotemporal models and evaluating environmental risk factors.

### Table B: Ontario Wild Bird Pilot Dataset
* **Output Path:** `data/processed/modeling/model_table_v1_ontario_gbif_pilot.csv`
* **Features Included:** Environment features + Ontario GBIF waterfowl occurrences and relative activity shares.
* **Coverage:** Ontario only (577 CSDs, 2022).
* **Expected Row Count:** 6,924 rows (577 CSDs * 12 months).
* **Purpose:** Test code integration, check GNN node feature loading, and validate target wild bird covariates before scaling wild bird extraction nationally.

---

## 4. Key Assembly Rules and Leakage Prevention

1. **Alignment:** Ensure the `csd_uid` is treated strictly as a string and mapped correctly across shapefiles, climate, land cover, and label datasets.
2. **One-Month-Ahead Target:** The target variable `outbreak_next_month` for CSD-month $(M)$ is defined as the outbreak status at month $(M+1)$.
3. **Future Leakage Prevention:**
   - For strict real-time forecasting, features at month M should use only information available up to month M. The current V1 Dynamic World table uses retrospective centered temporal filling for missing optical observations, so this must be flagged and revisited before claiming real-time forecasting performance.
   - Do not use future months or full-year aggregated totals for feature imputation or normalization.
   - Do not include target labels ($M+1$) or current month labels ($M$) as covariates.

---

## 5. Validation and Quality Assurance (QA)

The assembly scripts must execute and report the following verification metrics:
* **Duplicate Check:** Confirm zero duplicate `csd_uid`/`year`/`month` rows.
* **Temporal Integrity:** Check that label periods match feature periods (no target shifted outside the 2022-2025 boundary).
* **Class Imbalance Summary:** Report the number and percentage of positive targets (`outbreak_next_month = 1`) overall and grouped by year and province.
* **Row Counts Validation:** Confirm row counts are exactly 242,592 for Table A, and 6,924 for Table B.
* **Missingness Report:** Print the count and percentage of NaNs in features. Climate and land cover should show 0% missingness due to upstream imputation.
