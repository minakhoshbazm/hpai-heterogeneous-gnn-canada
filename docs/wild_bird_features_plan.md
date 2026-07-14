# Implementation Plan: Wild Bird and Migration Features for CSD-Level HPAI Modeling

This document outlines the design and plan for extracting monthly wild bird abundance and migration features at the Census Subdivision (CSD) level across Canada for the years 2022-2025.

## 1. Rationale and Context

Wild birds, particularly migratory waterfowl, are the primary natural reservoirs and vectors for Highly Pathogenic Avian Influenza (HPAI) viruses. The risk of transmission to domestic poultry is strongly driven by:
* **Waterfowl Habitat Overlap:** Direct or indirect contact between wild waterfowl (e.g., ducks, geese) and domestic poultry in agricultural landscapes.
* **Migration Timings:** Seasonal migration corridors and stopover locations where large concentrations of birds gather during spring and autumn.
* **Species Abundance:** Local abundance and species richness of high-risk viral hosts.

Integrating monthly wild bird distribution features is essential for transitioning from static spatial models to dynamic, spatiotemporal risk prediction.

---

## 2. Modeling Specifications

* **Modeling Unit:** CSD-month (matching the ERA5 climate, Dynamic World land cover, and CFIA outbreak datasets)
* **Temporal Scope:** 2022-2025
* **Spatial Scope:** 5,054 provincial CSDs (excluding territories, in line with the Dynamic World spatial scope decision)
* **Target Output File:** `data/processed/features/csd_monthly_wild_birds_2022_2025.csv`

---

## 3. Candidate Data Sources

We evaluate three candidate data sources for extracting wild bird distribution metrics:

| Source | Strengths | Weaknesses |
| :--- | :--- | :--- |
| **eBird Status and Trends** *(Preferred)* | High-resolution weekly modeled abundance surfaces; corrected for observer effort. | Requires registration; data access API is restricted and may have rate limits. |
| **eBird Basic Dataset (EBD)** | Complete raw observation checklists; highly detailed. | Requires massive volume downloads, custom spatial filtering, and complex effort-correction modeling. |
| **GBIF Fallback** | Fully open-access; simple API; covers multiple citizen-science databases. | High spatial/temporal sampling bias; lacks observation effort standardization. |

---

## 4. Target Bird Taxa

Features will focus on key reservoir and carrier groups for avian influenza:
* **Ducks:** Dabbling ducks and diving ducks, such as Mallard, Northern Pintail, American Wigeon, and scaup species, which are important AIV reservoir groups.
* **Geese:** Canada Goose, Snow Goose (known carriers of HPAI).
* **Swans:** Tundra Swan, Trumpeter Swan.
* **Gulls & Shorebirds:** Secondary vectors that often bridge habitats between agricultural lands and waterways.

---

## 5. Planned CSD-Month Features

The following features will be extracted/computed for each CSD and calendar month:

* `waterfowl_abundance_mean`: Mean abundance of all target waterfowl species combined.
* `waterfowl_abundance_max`: Peak abundance in the CSD during the month.
* `waterfowl_species_richness`: Count of target species with non-zero abundance.
* `goose_abundance_mean`: Mean abundance index for geese.
* `duck_abundance_mean`: Mean abundance index for ducks.
* `migration_intensity_index`: An indicator combining the monthly rate of change in abundance to capture peak transit periods.
* `bird_data_missing_flag`: A binary flag (0 or 1) indicating if a CSD-month lacked sufficient raw observations or model coverage.

---

## 6. Recommended V1 Implementation Roadmap

1. **Proof-of-Concept Pilot:** Start with a pilot subset (e.g., Ontario in 2022) using a limited species list.
2. **eBird Status & Trends Access:** Attempt to access eBird Status & Trends data via the `ebirdst` R/Python package.
3. **GBIF Fallback Execution:** If eBird API credentials cannot be secured in a timely manner, use standard GBIF queries to retrieve raw occurrence counts for target species, normalized by total monthly observations per CSD, as a V1 fallback.
4. **National Scaling:** Once the pilot is validated, scale the extraction nationally to the 5,054 provincial CSDs.

---

## 7. Validation and QA Plan

Imputed/processed bird features must pass the following checks:
* **Scope Consistency:** Total row count must exactly match the 242,592 CSD-months (5,054 CSDs * 48 months).
* **Target Leakage:** Check that no future outbreak labels or downstream information is used during the normalization or spatial aggregation.
* **Ecological Plausibility:**
  - Waterfowl features must exhibit clear bimodal peaks corresponding to spring migration (April/May) and autumn migration (September/October/November).
  - Waterfowl abundance should drop to near-zero in inland northern/frozen CSDs during winter months (January/February).
  - Waterfowl abundance should be compared against wet/water land cover features (`dw_water_mean`) to check whether spatial patterns are ecologically plausible.
