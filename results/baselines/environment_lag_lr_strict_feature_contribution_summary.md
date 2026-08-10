# Strict Environment + Lag Logistic Regression Feature Contribution Analysis

## Executive Summary
This document reports the standardized feature coefficients and monthly-stratified grouped permutation importance for the strict 50-feature Environment + Lag Logistic Regression model evaluated on the untouched 2025 test set.

> [!IMPORTANT]
> **CAUSAL DISCLAIMER**: These results represent **statistical predictive associations** identified within observational data. They do **not** constitute proof of biological or epidemiological causation.

---

## 1. Grouped Permutation Importance (2025 Test Set)
Features were permuted within each calendar month across 30 random seeds to preserve monthly seasonality.

| feature_group | feature_count | baseline_pr_auc | mean_pr_auc_drop | std_pr_auc_drop | relative_importance_pct |
| --- | --- | --- | --- | --- | --- |
| Roads/geographic structure | 12 | 0.065369 | 0.060149 | 0.000511 | 92.014851 |
| Missingness/data-quality flags | 8 | 0.065369 | 0.049502 | 0.001282 | 75.727815 |
| Poultry/agriculture | 10 | 0.065369 | 0.039669 | 0.001111 | 60.685861 |
| Outbreak history | 7 | 0.065369 | 0.013967 | 0.002903 | 21.366620 |
| Dynamic World land cover | 9 | 0.065369 | 0.005244 | 0.002950 | 8.022726 |
| Climate | 4 | 0.065369 | 0.003991 | 0.001944 | 6.105550 |

---

## 2. Top 20 Positive Feature Coefficients (Higher Risk Association)

| feature_name | feature_group | standardized_coefficient | odds_ratio |
| --- | --- | --- | --- |
| glw2015_duck_density_max | Poultry/agriculture | 3.449954 | 31.498935 |
| glw2020_chicken_valid_pixel_count | Missingness/data-quality flags | 2.283211 | 9.808129 |
| glw2020_chicken_density_p95 | Poultry/agriculture | 1.894522 | 6.649369 |
| glw2015_duck_density_mean | Poultry/agriculture | 0.828133 | 2.289041 |
| major_road_length_km | Roads/geographic structure | 0.566154 | 1.761479 |
| province_outbreak_count_current_month | Outbreak history | 0.449338 | 1.567274 |
| same_csd_outbreak_count_past_3_months | Outbreak history | 0.379864 | 1.462086 |
| major_road_density_km_per_km2 | Roads/geographic structure | 0.368615 | 1.445730 |
| highway_length_km | Roads/geographic structure | 0.361985 | 1.436178 |
| climate_tmean_c | Climate | 0.353766 | 1.424421 |
| climate_precip_mm | Climate | 0.339159 | 1.403766 |
| dw_built_mean | Dynamic World land cover | 0.290187 | 1.336677 |
| climate_tmin_c | Climate | 0.281691 | 1.325369 |
| glw2015_duck_valid_pixel_count | Missingness/data-quality flags | 0.270570 | 1.310711 |
| all_road_length_km | Roads/geographic structure | 0.224858 | 1.252145 |
| dw_crops_mean | Dynamic World land cover | 0.192019 | 1.211693 |
| ccs_turkey_farms_reporting | Poultry/agriculture | 0.189576 | 1.208737 |
| same_csd_outbreak_current_month | Outbreak history | 0.180693 | 1.198048 |
| dw_water_mean | Dynamic World land cover | 0.134789 | 1.144295 |
| same_csd_outbreak_count_past_12_months | Outbreak history | 0.100972 | 1.106245 |

---

## 3. Top 20 Negative Feature Coefficients (Lower Risk Association)

| feature_name | feature_group | standardized_coefficient | odds_ratio |
| --- | --- | --- | --- |
| glw2015_duck_density_p95 | Poultry/agriculture | -4.113021 | 0.016358 |
| csd_area_km2 | Roads/geographic structure | -3.570931 | 0.028130 |
| small_area_flag | Roads/geographic structure | -2.826445 | 0.059223 |
| climate_fallback_distance_km | Missingness/data-quality flags | -2.227232 | 0.107826 |
| distance_to_nearest_highway_km | Roads/geographic structure | -1.543336 | 0.213667 |
| all_road_density_km_per_km2 | Roads/geographic structure | -1.455863 | 0.233199 |
| glw2020_chicken_density_max | Poultry/agriculture | -0.866627 | 0.420367 |
| glw2020_chicken_density_mean | Poultry/agriculture | -0.777270 | 0.459659 |
| dw_trees_mean | Dynamic World land cover | -0.630818 | 0.532156 |
| province_outbreak_count_past_3_months | Outbreak history | -0.551058 | 0.576340 |
| distance_to_nearest_major_road_km | Roads/geographic structure | -0.539978 | 0.582761 |
| highway_density_km_per_km2 | Roads/geographic structure | -0.301125 | 0.739985 |
| dw_bare_mean | Dynamic World land cover | -0.205014 | 0.814636 |
| dw_image_count | Missingness/data-quality flags | -0.185020 | 0.831088 |
| climate_tmax_c | Climate | -0.182039 | 0.833569 |
| dw_grass_mean | Dynamic World land cover | -0.177743 | 0.837158 |
| major_road_proportion | Roads/geographic structure | -0.168995 | 0.844513 |
| highway_proportion | Roads/geographic structure | -0.154733 | 0.856644 |
| ccs_broilers_roasters_cornish_production_kg | Poultry/agriculture | -0.148998 | 0.861571 |
| dw_water_p95 | Dynamic World land cover | -0.136649 | 0.872276 |
