# Wild Bird Data Source Feasibility Decision

This document evaluates the feasibility of candidate wild bird datasets for monthly CSD-level Highly Pathogenic Avian Influenza (HPAI) risk prediction in Canada (2022-2025).

## 1. Data Source Comparison

We compare three candidate wild bird datasets: **eBird Status and Trends**, **eBird Basic Dataset (EBD)**, and **GBIF occurrence data**.

### 1. eBird Status and Trends
* **Access Difficulty:** Moderate to High. Requires requesting access credentials from the Cornell Lab of Ornithology and downloading packages via R (`ebirdst`) or Python wrappers.
* **Data Size:** Moderate. Modeled abundance is distributed as GeoTIFF rasters or spatial grids.
* **Monthly CSD-level support:** High. Provides modeled week-by-week relative abundance surfaces that can be aggregated to monthly CSD-level averages.
* **Bias/Limitations:** Modeled values are highly standardized and corrected for observer search effort. However, results are limited to years/regions where models are fully trained (may exclude some remote northern sections).
* **Suitability for V1:** **Preferred choice** if access is approved soon, due to effort-corrected abundance metrics.

### 2. eBird Basic Dataset (EBD)
* **Access Difficulty:** Moderate. Requires registering for access to the full dataset.
* **Data Size:** Extremely High. EBD is a massive tabular dataset containing billions of raw observations (multi-gigabyte downloads).
* **Monthly CSD-level support:** Moderate. Requires manual spatial joins on coordinate attributes and temporal binning.
* **Bias/Limitations:** Contains raw occurrences that suffer from high citizen science spatial reporting bias (over-reporting near urban centers, under-reporting in remote areas). Requires complex effort-correction modeling (checklists, time, distance, group size) to infer true abundance.
* **Suitability for V1:** **Not suitable** for now. The data size and complexity of spatial filtering and effort correction are too large for a lightweight implementation.

### 3. GBIF Occurrence Data (Citizen Science / Mixed)
* **Access Difficulty:** Low (Open access via simple python API queries or bulk download).
* **Data Size:** Low to Moderate. Filtered queries for specific target taxa can be downloaded rapidly.
* **Monthly CSD-level support:** High. Provides raw spatial occurrences with timestamps, enabling CSD-month grouping.
* **Bias/Limitations:** Raw GBIF occurrence counts are strongly affected by observer/reporting effort. For V1, keep both raw counts and effort/context proxies, such as total GBIF bird occurrence count and target-group share within the same CSD-month. These should be interpreted as relative activity indicators, not true abundance.
* **Leakage Caution:** To avoid temporal leakage, GBIF features for a CSD-month must be computed only from records in that same month or earlier. Do not normalize using future months, full-year totals, or outbreak labels.
* **Suitability for V1:** **Recommended fallback** for proof-of-concept modeling if eBird Status and Trends access is delayed.

---

## 2. Decision and Recommendations

1. **Primary Recommendation:** Apply for Cornell eBird Status and Trends access credentials. Use this data if it is available soon.
2. **Fallback Recommendation:** Use GBIF as a temporary V1 proof-of-concept source. Query coordinates and normalize counts to bypass access delays and test GNN integration.
3. **Avoid EBD:** Avoid full eBird Basic Dataset download and processing for V1 due to scale and effort-modeling overhead.

---

## 3. Short Pilot Species/Group List

For the initial pilot, the following high-risk reservoir and vector species/groups are selected:
* **Canada Goose** (*Branta canadensis*) - Geese group
* **Snow Goose** (*Anser caerulescens*) - Geese group
* **Mallard** (*Anas platyrhynchos*) - Ducks group
* **Northern Pintail** (*Anas acuta*) - Ducks group
* **American Wigeon** (*Mareca americana*) - Ducks group
* **Scaup Species** (*Aythya affinis/marila*) - Diving ducks group (if GEE/GBIF resolution allows)

### GBIF Pilot Features
For the GBIF-based pilot, the candidate CSD-month features to extract include:
* `gbif_waterfowl_count`
* `gbif_duck_count`
* `gbif_goose_count`
* `gbif_total_bird_count`
* `gbif_waterfowl_share`
* `gbif_reporting_intensity`
* `bird_data_missing_flag`
