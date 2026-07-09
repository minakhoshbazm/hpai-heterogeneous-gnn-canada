# Implementation Plan: Dynamic World Monthly Land Cover Features for CSD-Level HPAI Modeling

This document outlines the design and implementation plan for extracting spatiotemporal land cover features at the Census Subdivision (CSD) level across Canada for the years 2022-2025, using the Google Earth Engine (GEE) Dynamic World dataset.

## 1. Rationale for Dynamic World

Unlike static land cover products (e.g., annual or decadal maps), Google Earth Engine's **Dynamic World** (U+0060GOOGLE/DYNAMICWORLD/V1U+0060) provides high-resolution (10m), near-real-time land cover class probabilities. Using monthly composites derived from Dynamic World is critical for modeling Highly Pathogenic Avian Influenza (HPAI) risk because:
* **Waterfowl Habitat Dynamics:** Waterfowl (the primary reservoirs of HPAI) migrate and group based on seasonally shifting wetlands, flooded croplands, and open water.
* **Intra-Annual Changes:** Static datasets miss seasonal transitions (e.g., spring melting, summer drying, autumn pooling, and winter snow cover), whereas monthly Dynamic World features capture these temporal trends.
* **Probability Vectors:** Instead of hard categorical classifications, Dynamic World outputs continuous probability bands for multiple land cover classes, permitting sub-pixel analysis of land cover change over time.

---

## 2. Dataset and Modeling Specifications

* **Modeling Unit:** CSD-month
* **Temporal Scope:** 2022-2025 (matching ERA5 climate features and CFIA avian influenza labels)
* **Spatial Geometry Collection:** `projects/avianinfluenza/assets/canada_csd_2021_boundaries` (5,161 CSDs)
* **GEE Image Collection:** `GOOGLE/DYNAMICWORLD/V1`
* **Target Output File:** `data/processed/features/csd_monthly_dynamic_world_2022_2025.csv`

---

## 3. Planned Monthly Features (Per CSD)

The extraction script will aggregate the following continuous probability bands (values ranging from 0.0 to 1.0) and custom indicators:

| Feature Name | Source Band / Formula | Ecological Relevance to HPAI |
| :--- | :--- | :--- |
| `dw_water_mean` | `water` (mean probability) | Identifies permanent or major open water bodies. |
| `dw_water_p95` | `water` (95th percentile) | Detects peak water extent or maximum pooling in CSD-month. |
| `dw_flooded_vegetation_mean` | `flooded_vegetation` (mean) | Maps wetlands, marshes, and swamps (prime nesting/feeding habitats). |
| `dw_crops_mean` | `crops` (mean probability) | Croplands attract foraging birds, especially post-harvest. |
| `dw_grass_mean` | `grass` (mean probability) | Foraging context and open fields. |
| `dw_trees_mean` | `trees` (mean probability) | Forested context and roosting environments. |
| `dw_built_mean` | `built` (mean probability) | Human-wildlife interface and urban proximity. |
| `dw_bare_mean` | `bare` (mean probability) | Shorelines, mudflats, or tilled soil. |
| `dw_water_anomaly_vs_annual_mean` | `dw_water_mean` - Annual CSD mean | Identifies wetter/drier months relative to that CSD's baseline. |
| `dw_seasonal_pond_proxy` | Custom formula (see below) | Proxies ephemeral ponds, flooding, or temporary wetlands. |

### Seasonal Pond Proxy
The `dw_seasonal_pond_proxy` is a candidate proxy for patchy or localized temporary water (such as ephemeral ponds, agricultural flooding, or variable margins) that may attract wild birds but are not classified as permanent water:
$$\text{dw\_seasonal\_pond\_proxy} = \text{dw\_water\_p95} - \text{dw\_water\_mean}$$
* **Low values** indicate stable, permanent water bodies (e.g., deep lakes) or uniform dry land.
* **High values** suggest high transient water presence (e.g., temporary pooling or fluctuating margins) during the month, which may represent temporary wetland habitat.

---

## 4. Important Cautions and QA Masking

Dynamic World is generated from Sentinel-2 imagery. Because of this, it is susceptible to atmospheric and temporal artifacts:
1. **Cloud Cover:** Persistent cloud cover (especially in northern regions during winter) can corrupt monthly composites. Cloud masking using Sentinel-2 QA bands must be enforced prior to composite generation.
2. **Snow and Ice:** Winter snow cover can be misclassified or hide underlying land cover. Snow masking is required, or winter months must be validated for missingness.
3. **Data Availability:** Months with zero valid cloud-free observations will result in null features. A robust imputation strategy or fallback logic (such as spatial interpolation or multi-month buffers) must be planned.

---

## 5. Recommended Phased Implementation

### Phase 1: Smoke Test
Prior to national scale extraction, run a localized GEE smoke test:
* **Scope:** 10 CSDs (comprising a mix of inland, coastal, and island features, including controls).
* **Temporal Period:** 2-3 months in 2022 (e.g., April, July, and October to cover spring, summer, and autumn).
* **Execution:** Extract the mean probability bands, print diagnostics to the GEE console, and check execution performance.

### Phase 2: National Scaling
Upon validation of the smoke test results and confirmation that GEE resources can handle the computation, scale the script to extract all 5,161 CSDs for all 48 months (2022-2025) and save the output.

---

## 6. Validation and QA Plan

After extraction, the combined CSV file must pass the following QA validation steps:
* **Completeness:** Ensure exactly `247728` rows (5,161 CSDs * 48 months).
* **Feature Ranges:** Verify that all mean probability values (`dw_water_mean`, `dw_crops_mean`, etc.) fall strictly within the `[0.0, 1.0]` range.
* **Spatial Missingness:** Check missingness rates grouped by province and month to isolate cloud/snow artifacts.
* **Ecological Plausibility:** 
  - Compare seasonal water features: `dw_water_mean` and `dw_seasonal_pond_proxy` should show plausible regional seasonal patterns, such as spring wetness in snowmelt regions or localized summer/autumn water signals in wetlands and coastal areas.
  - Crop probabilities should align with agricultural provinces (Saskatchewan, Manitoba, Alberta, Ontario) and remain low in mountainous/forested zones.
