# Dynamic World V1 Spatial Scope Decision Note

This document records the spatial scope decision for the Dynamic World monthly land cover features used in CSD-level Highly Pathogenic Avian Influenza (HPAI) risk modeling in Canada (2022-2025).

## 1. Scope Details

The combined Dynamic World land cover features dataset contains **5,054** unique Census Subdivisions (CSDs), which is less than the total count of 5,161 CSDs in the national CSD shapefile. This difference of **107** CSDs is intentional:
* **Included (5,054 CSDs):** All CSDs belonging to the 10 Canadian provinces: Newfoundland and Labrador (10), Prince Edward Island (11), Nova Scotia (12), New Brunswick (13), Quebec (24), Ontario (35), Manitoba (46), Saskatchewan (47), Alberta (48), and British Columbia (59).
* **Excluded (107 CSDs):** All CSDs belonging to the three northern territories: Yukon (60), Northwest Territories (61), and Nunavut (62).

---

## 2. Rationale for Territory Exclusion in V1

The poultry outbreak modeling and GNN architecture in V1 are focused on the southern provincial CSDs. Exclusions of northern territories are justified for the following reasons:
1. **Outbreak Profile:** Commercial poultry facilities and historical avian influenza outbreaks in domestic poultry are highly concentrated in the agricultural regions of the 10 provinces.
2. **Missingness Overhead:** High latitude territory CSDs suffer from severe, persistent winter darkness, snow, and clouds, which create significant compositing gaps in optical Sentinel-2/Dynamic World datasets. Including them in V1 would disproportionately bias missing-value imputations.
3. **Modeling Efficiency:** Restricting CSDs to the 5,054 provincial ones reduces spatial graph size and computation requirements without affecting model applicability to key poultry production corridors.

---

## 3. Modeling Recommendations

* **V1 Recommendation:** Restrict the final graph neural network modeling dataset to the 5,054 provincial CSDs. Any CSDs from territories (province codes 60, 61, 62) should be excluded or explicitly marked as not modeled.
* **Future Extension:** If future iterations require extending the model to northern regions (e.g. to model wild bird vectors in remote areas), Dynamic World and climate features must be explicitly handled for those 107 CSDs, incorporating appropriate high-latitude QA filters and long-term historical averages to impute missing optical values.
