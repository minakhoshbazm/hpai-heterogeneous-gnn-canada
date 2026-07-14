# Implementation Plan: GBIF Wild Bird Activity Features (Ontario 2022 Pilot)

This document outlines the implementation plan for a lightweight proof-of-concept pilot to extract monthly wild bird occurrence and relative activity metrics in Ontario for the year 2022, using GBIF (Global Biodiversity Information Facility) data.

## 1. Rationale and Scope

Due to potential registration delays and complexity in accessing Cornell's effort-corrected eBird Status and Trends models, **GBIF occurrence data** serves as our temporary V1 fallback. Because citizen science observations in GBIF are unstandardized and heavily biased by observer effort, this pilot is designed as a **proof-of-concept for relative activity proxy metrics**, not true ecological abundance.

### Pilot Scope:
* **Spatial Unit:** Census Subdivisions (CSDs) in Ontario only (577 CSDs).
* **Temporal Unit:** Monthly bins for the year 2022 (January-December).
* **Expected Output Rows:** 6,924 rows (577 CSDs * 12 months).
* **Target Species:**
  * Canada Goose (*Branta canadensis*) - Geese group
  * Snow Goose (*Anser caerulescens*) - Geese group
  * Mallard (*Anas platyrhynchos*) - Ducks group
  * Northern Pintail (*Anas acuta*) - Ducks group
  * American Wigeon (*Mareca americana*) - Ducks group
  * Scaup Species (*Aythya affinis* and *Aythya marila*) - Diving ducks group

---

## 2. Planned Pilot Features (Per CSD-Month)

The following features will be extracted and computed:

* `gbif_waterfowl_count`: Sum of occurrences of all target waterfowl species combined.
* `gbif_duck_count`: Sum of occurrences of target duck species.
* `gbif_goose_count`: Sum of occurrences of target goose species.
* `gbif_total_bird_count`: Sum of occurrences of all class *Aves* records (serves as the denominator proxy for local observer effort).
* `gbif_waterfowl_share`: Ratio of target waterfowl occurrences to total bird occurrences:
  gbif_waterfowl_share = gbif_waterfowl_count / gbif_total_bird_count
* `gbif_reporting_intensity`: Total GBIF bird occurrence records in that CSD-month, used as a rough reporting-effort proxy.
* `bird_data_missing_flag`: A binary flag (1 if `gbif_total_bird_count` is 0 or NaN, 0 otherwise).

---

## 3. Leakage Prevention Guidelines

To prevent temporal leakage in downstream HPAI risk GNN modeling:
1. **Same-Month Bounds:** Features for CSD-month month M must be computed using occurrences recorded strictly in month M or earlier.
2. **No Future Normalization:** The denominator (`gbif_total_bird_count`) must be CSD-month specific. Do not use full-year CSD totals, future month counts, or national averages to normalize.
3. **Target Exclusion:** Under no circumstances should CFIA poultry outbreak labels or outbreak indicators be used during spatial aggregation, normalization, or feature imputation.

---

## 4. Proposed Pilot Workflow

1. Query GBIF for target species in Ontario, 2022.
2. Query GBIF for all Aves occurrences in Ontario, 2022.
3. Spatially join occurrences to CSD polygons.
4. Aggregate counts to CSD-month bins.
5. Compute waterfowl share and missing flags.
6. Run QA and save the pilot CSV.

1. **Query GBIF (Target Taxa):** Programmatically retrieve occurrence coordinates (lat/lon) and timestamps (year, month) for the target species in Ontario for 2022 using the `pygbif` API or direct downloads.
2. **Query GBIF (Observer Effort):** Retrieve coordinates and timestamps for all *Aves* occurrences in Ontario for 2022.
3. **Spatial Join:** Perform a spatial join (e.g., using `geopandas` or Earth Engine) to assign each occurrence to a CSD boundary based on coordinates.
4. **Temporal Grouping:** Group and sum occurrence counts by `csd_uid`, `year`, and `month`.
5. **Feature Calculation:** Compute waterfowl share and assign missing flags. Keep both the raw counts and the normalized shares.
6. **Seasonal Check:** Inspect the aggregated profiles to verify seasonal spikes match standard migratory patterns.

---

## 5. Validation and QA Plan

The generated pilot CSV must pass the following checks:
* **Completeness:** Verify row count is exactly 6,924 rows.
* **Integrity:** Ensure no duplicate `csd_uid`/`year`/`month` rows.
* **Non-Negativity:** Verify all count columns are >= 0.
* **Ratio Bounds:** Verify `gbif_waterfowl_share` is strictly within `[0.0, 1.0]` where total bird count > 0.
* **Missingness Reporting:** Document the percentage of CSD-months with zero bird observations (missing flag = 1).
* **Ecological Alignment:** Check that waterfowl counts peak during spring migration (April/May) and autumn migration (October/November).
