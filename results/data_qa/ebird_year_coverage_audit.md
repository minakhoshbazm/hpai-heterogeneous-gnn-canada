# eBird Status & Trends Product Year Coverage Audit

## Executive Summary
This report audits the locally available eBird Status and Trends raw raster dataset under `data/raw/hpai_ebird_birds_9km/`.

- **Required Modeling Years**: 2022, 2023, 2024, 2025
- **Discovered Product Years**: [2023]
- **Missing Required Product Years**: [2022, 2024, 2025]

---

## 1. Product Year Coverage Summary

| required_year | status | downloaded_species_count | downloaded_species_codes | missing_species_count | missing_species_codes |
| --- | --- | --- | --- | --- | --- |
| 2022 | MISSING | 0 | None | 9 | mallar3, ambduc, norpin, gnwtea, buwtea, snogoo, cangoo, canvas, redhea |
| 2023 | Present | 9 | ambduc, buwtea, cangoo, canvas, gnwtea, mallar3, norpin, redhea, snogoo | 0 | None |
| 2024 | MISSING | 0 | None | 9 | mallar3, ambduc, norpin, gnwtea, buwtea, snogoo, cangoo, canvas, redhea |
| 2025 | MISSING | 0 | None | 9 | mallar3, ambduc, norpin, gnwtea, buwtea, snogoo, cangoo, canvas, redhea |

---

## 2. GeoTIFF Raster Technical Specifications

No GeoTIFF rasters found in subdirectories.

---

## 3. Strict Methodological Rule Compliance Check
- **Rule**: Never copy or repeat 2023 eBird rasters into another year for primary forecasting.
- **Audit Findings**:
  - The local repository contains eBird Status and Trends data **only for product year 2023**.
  - Product years **2022, 2024, and 2025 are completely missing**.
  - eBird Status and Trends products are static multi-year model estimates (versioned by product year 2023) and do not provide separate, independent annual product releases for 2022, 2024, and 2025.

---

## 4. Final Clear Decision

> [!CAUTION]
> **DECISION**: **Year-specific eBird data are incomplete: do not use eBird in the primary forecast model; prepare a national GBIF fallback plan.**

### Decision Rationale:
1. Product years 2022, 2024, and 2025 are missing from the eBird raw dataset.
2. Repeating the 2023 static eBird seasonal relative abundance pattern across 2022, 2024, and 2025 violates temporal leakage safety and introduces false constant values across forecast years.
3. Therefore, eBird data **must not be used** in the primary leakage-safe forecasting pipeline.
4. Next Step: Prepare a national GBIF fallback plan for year-specific wild bird occurrence data.
