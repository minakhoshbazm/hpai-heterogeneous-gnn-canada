# Decision Note: Deriving Monthly Precipitation from ERA5-Land in Google Earth Engine

## 1. Background
During the initial ERA5-Land climate feature smoke test for Census Subdivision (CSD) 1001101 (Division No. 1, Subd. V, NL, Canada) for January 2022, the temperature features were successfully extracted and validated. However, the precipitation feature precip_sum_mm yielded an anomalous total of 3166.5 mm. 

This document investigates the root cause, explores the aggregation mechanics of ERA5-Land in Google Earth Engine (GEE), evaluates alternative approaches, and recommends a verification path.

---

## 2. Root Cause: Accumulated Semantics in Hourly ERA5-Land
The hourly ERA5-Land dataset in GEE (ECMWF/ERA5_LAND/HOURLY) represents total_precipitation as an accumulated field rather than an instantaneous rate.

### Key Semantics:
1. Daily Forecast Accumulation: Unlike the standard hourly ERA5 dataset which accumulates over the forecast length, ERA5-Land accumulates values from the start of the forecast day and resets to zero at 00:00 UTC daily.
2. Hourly Value Meaning: At any given hour h of the day, the value in the total_precipitation band represents the total depth of water (in meters) that has accumulated since 00:00 UTC on that calendar day.
3. The Simple Sum Error: Performing a simple sum of all hourly values over a month results in double-counting:
   Daily Sum = sum over h from 1 to 24 of (Accumulated Precipitation at hour h)
   Daily Sum = 24 * P_1 + 23 * P_2 + ... + 1 * P_24
   where P_i is the actual hourly precipitation for hour i.
   
   - This mathematically overweights early-hour precipitation and inflates the daily total by roughly 12.5 to 24 times the true value.
   - The 3166.5 mm result is implausibly high for a monthly CSD-level precipitation total and should be treated as a failed aggregation method unless independently validated.
   - In contrast, the hourly maximum precip_max_mm was 41.2 mm, which represents the maximum daily accumulation reached just before a reset, a highly plausible winter storm daily total.

---

## 3. Google Earth Engine Official Collections and Bands
An inspection of the official Google Earth Engine and ECMWF documentation reveals key findings for monthly and hourly ERA5-Land collections.

### 3.1 Monthly Aggregated Collection
GEE provides a pre-aggregated monthly collection:
* Collection ID: ECMWF/ERA5_LAND/MONTHLY_AGGR
* Key Bands:
  - temperature_2m: Monthly mean air temperature (Kelvin)
  - temperature_2m_min: Monthly minimum air temperature (Kelvin)
  - temperature_2m_max: Monthly maximum air temperature (Kelvin)
  - total_precipitation_sum: Monthly sum of total precipitation (meters)

This collection pre-calculates sums for flow (accumulation) variables and averages for state variables.

### 3.2 Hourly Collection Finding: total_precipitation_hourly
In the hourly collection (ECMWF/ERA5_LAND/HOURLY), GEE provides two separate precipitation bands:
* total_precipitation: The raw accumulated field resetting daily at 00:00 UTC.
* total_precipitation_hourly: A band described by GEE as disaggregated from the original cumulative values into actual hourly values (representing the amount that fell during that specific hour).

This hourly band represents a strong alternative candidate for precipitation extraction, as a simple sum of total_precipitation_hourly over the month should theoretically yield the correct monthly total.

---

## 4. Evaluation of Options

### Option A: Use Monthly Aggregated Product (ECMWF/ERA5_LAND/MONTHLY_AGGR)
* Description: Transition the feature extraction pipeline to the pre-calculated monthly collection.
* Pros:
  - Simplest and safest: Aggregation logic is pre-computed and verified. No risk of daily accumulation reset errors or timezone boundary bugs.
  - Performance: Substantially reduces GEE computation overhead, avoiding potential out-of-memory or timeout errors during multi-year national extractions.
  - Contains pre-calculated monthly minimum and maximum temperature bands.
* Cons:
  - Latency: Consolidated monthly data typically has a latency of 2-3 months (acceptable for historical modeling).

### Option B: Use Hourly Collection with total_precipitation_hourly
* Description: Retrieve hourly data from ECMWF/ERA5_LAND/HOURLY, select total_precipitation_hourly, and sum it over the month.
* Pros:
  - Near-real-time updates (lower latency).
  - Keeps the codebase on a single (hourly) collection if preferred.
* Cons:
  - High computation cost: Reducing hourly data for all CSDs over a multi-year period in GEE can trigger memory limit exceptions compared to pre-aggregated monthly collections.

### Option C: Temporarily Exclude Precipitation from V1 Feature Set
* Description: Proceed with climate_tmean_c, climate_tmin_c, and climate_tmax_c only, setting precipitation to null or excluding it.
* Pros: Unblocks immediate downstream pipeline development.
* Cons: Precipitation is a key driver of environmental factors relevant to avian influenza vectors.

---

## 5. Recommendation and Next Steps

1. Prioritize Option A (Monthly Collection): Because our target timeline is historical and does not require real-time forecasts, the 2-3 month latency of ECMWF/ERA5_LAND/MONTHLY_AGGR is not a bottleneck. We should use it to obtain temperature_2m (mean), temperature_2m_min, temperature_2m_max, and total_precipitation_sum.
2. Execute a Single-CSD / Single-Month Comparison Test:
   Before updating the main pipeline scripts, run a focused test for CSD 1001101 for January 2022 comparing three methods:
   - Method A: ECMWF/ERA5_LAND/MONTHLY_AGGR using total_precipitation_sum (converted to mm by multiplying by 1000).
   - Method B: ECMWF/ERA5_LAND/HOURLY using total_precipitation_hourly summed over the month (converted to mm by multiplying by 1000).
   - Method C (Old/Incorrect): total_precipitation simple sum over hourly images (converted to mm).
3. Select the best method: Use the monthly collection (Option A) first if the one-CSD comparison is plausible, and fallback to Option B if necessary.

---

## 6. Official Sources
* GEE ERA5-Land Monthly Aggregated: https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR
* GEE ERA5-Land Hourly: https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY
* ECMWF ERA5-Land Documentation: https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation
