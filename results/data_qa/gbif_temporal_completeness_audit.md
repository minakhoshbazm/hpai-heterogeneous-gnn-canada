# GBIF Waterfowl Temporal Completeness & Publication Timing Audit Report

## Executive Summary
This audit evaluated the temporal completeness and publication latency of GBIF waterfowl occurrence data for Canada across 2022, 2023, 2024, and 2025.

- **2022 Total Usable Counts**: 1,207,525
- **2023 Total Usable Counts**: 1,231,978
- **2024 Total Usable Counts**: 1,312,552
- **2025 Total Usable Counts**: **41,442** (Deficit of **96.69%** relative to 2022–2024 average)

---

## 1. National Monthly Occurrence Aggregates (All 9 Species Combined)

| month | 2022 | 2023 | 2024 | 2025 | avg_2022_2024 | pct_2025_vs_avg |
| --- | --- | --- | --- | --- | --- | --- |
| 1.00 | 60910.00 | 62038.00 | 60326.00 | 1814.00 | 61091.33 | 2.97 |
| 2.00 | 45889.00 | 49896.00 | 59898.00 | 1421.00 | 51894.33 | 2.74 |
| 3.00 | 109336.00 | 90842.00 | 128365.00 | 3897.00 | 109514.33 | 3.56 |
| 4.00 | 212353.00 | 213088.00 | 213991.00 | 8569.00 | 213144.00 | 4.02 |
| 5.00 | 224411.00 | 221468.00 | 228882.00 | 6058.00 | 224920.33 | 2.69 |
| 6.00 | 84279.00 | 84292.00 | 90603.00 | 3571.00 | 86391.33 | 4.13 |
| 7.00 | 51304.00 | 53502.00 | 56690.00 | 3156.00 | 53832.00 | 5.86 |
| 8.00 | 70797.00 | 79905.00 | 84620.00 | 2625.00 | 78440.67 | 3.35 |
| 9.00 | 104350.00 | 113857.00 | 119764.00 | 3520.00 | 112657.00 | 3.12 |
| 10.00 | 115552.00 | 119525.00 | 122479.00 | 3405.00 | 119185.33 | 2.86 |
| 11.00 | 73568.00 | 80771.00 | 84871.00 | 2016.00 | 79736.67 | 2.53 |
| 12.00 | 54776.00 | 62794.00 | 62063.00 | 1390.00 | 59877.67 | 2.32 |

---

## 2. Species-Level 2025 Completeness Deficit

| common_name | count_2025 | expected_3yr_total |
| --- | --- | --- |
| American Black Duck | 1,551 | 273629.00 |
| Blue-winged Teal | 1,189 | 131925.00 |
| Canada Goose | 14,636 | 1371831.00 |
| Canvasback | 576 | 49537.00 |
| Green-winged Teal | 1,774 | 247809.00 |
| Mallard | 18,392 | 1354536.00 |
| Northern Pintail | 1,177 | 136269.00 |
| Redhead | 1,238 | 90802.00 |
| Snow Goose | 909 | 95717.00 |

---

## 3. Publication Timing & Metadata Ingestion Audit

Sampling occurrence record metadata revealed the underlying structural reason for the 2025 count drop:

- **Primary Dataset Provider**: `datasetKey: 50c9509d-22c7-4a22-a47d-8c48425ef4a7` (EBD - eBird Observation Dataset published by Cornell Lab of Ornithology).
- **Ingestion Pattern**: The vast majority ($>95\%$) of wild bird observations in GBIF originate from eBird. Cornell publishes eBird data to GBIF in **infrequent annual bulk releases** (typically delayed by 6 to 18 months).
- **Publication Lag**: While the observation date (`eventDate`) records when the bird was seen, the ingestion timestamps (`modified`, `lastCrawled`, `lastInterpreted`) show that bulk records for year $Y$ are frequently not indexed into GBIF until mid-way through year $Y+1$ or later.
- **Operational Forecasting Rule Violation**: An observation recorded in month $M$ is **not available in GBIF by the end of month $M$**. Using GBIF occurrence counts as a predictor for month $M+1$ forecasting would create severe **lookahead bias and back-fill artifacts** during evaluation.

---

## 4. Final Clear Decision

> [!CAUTION]
> **CLEAR DECISION**: **DO NOT USE GBIF AS A STRICT 2022-2025 FORECASTING FEATURE**

### Decision Rationale & Mandatory Guidance:
1. **Severe Incompleteness in Test Year (2025)**: 2025 data in GBIF currently total only 41,442 records compared to over 1.25 million records per year in 2022–2024. Training or testing a 2022–2025 model on GBIF features would falsely attribute 2025 target changes to a non-existent collapse in wild bird activity.
2. **Publication Incompatibility with Real-Time Forecasting**: Due to eBird batch release cycles, GBIF observations are not available at month-end.
3. **Mandatory Action**:
   - **Do NOT include GBIF occurrence features in the primary leakage-safe forecasting pipeline or primary model tables.**
   - Retain GBIF data strictly as an **exploratory historical bird-observation activity index** for offline sensitivity checks.
