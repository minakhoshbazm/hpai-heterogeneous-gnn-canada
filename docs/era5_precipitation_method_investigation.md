# Decision Note: Deriving Monthly Precipitation from ERA5-Land in Google Earth Engine

## 1. Background
During the initial ERA5-Land climate feature smoke test for Census Subdivision (CSD) 1001101 (Division No. 1, Subd. V, NL, Canada) for January 2022, the temperature features were successfully extracted and validated. However, the precipitation feature precip_sum_mm yielded an anomalous total of 3166.5 mm. 

This document investigates the root cause, explores the aggregation mechanics of ERA5-Land in Google Earth Engine (GEE), presents the results of the comparison test, and defines the final extraction method.

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

---

## 4. GEE Comparison Test Results (January 2022, CSD 1001101)
To verify the correct method, a comparison test was conducted for CSD 1001101 in January 2022:

* **Method A (Monthly Collection):** `ECMWF/ERA5_LAND/MONTHLY_AGGR` using the `total_precipitation_sum` band:
  - Result: **254.55 mm** (calculated as 0.25455 meters * 1000)
* **Method B (Hourly Collection, Correct Disaggregation):** `ECMWF/ERA5_LAND/HOURLY` using the `total_precipitation_hourly` band summed over the month:
  - Result: **254.55 mm**
* **Method C (Old/Incorrect Method):** `ECMWF/ERA5_LAND/HOURLY` using the raw accumulated `total_precipitation` band summed over all hourly images:
  - Result: **3156.53 mm** (approx. 3166.5 mm in the initial unrounded smoke test)

### Key Conclusions:
1. **Method A and Method B are identical** in their final values (254.55 mm), confirming both correctly represent the true monthly accumulated precipitation.
2. **Method C is highly inflated** (~12.4x too high) due to summing the raw daily accumulations hour-by-hour.
3. **Choice of Collection:** Both Option A (Monthly Collection) and Option B (Hourly with `total_precipitation_hourly`) are valid. However, Option A is significantly more computationally efficient for long-term national-scale GEE extractions since it avoids summing 744 hourly scenes per month.

---

## 5. Recommendation and Next Steps

1. **Use Option A (Monthly Collection) for V1 Climate Extraction:**
   The extraction pipeline will use `ECMWF/ERA5_LAND/MONTHLY_AGGR` to extract all V1 climate features (both temperature and precipitation).
   * **Planned Bands & Variables:**
     - `temperature_2m` -> `climate_tmean_c` (mean temperature, Kelvin to Celsius)
     - `temperature_2m_min` -> `climate_tmin_c` (minimum temperature, Kelvin to Celsius)
     - `temperature_2m_max` -> `climate_tmax_c` (maximum temperature, Kelvin to Celsius)
     - `total_precipitation_sum` -> `climate_precip_mm` (sum of precipitation, meters to mm)
2. **Keep Option B as a Fallback/Validation Method:**
   Keep the hourly `total_precipitation_hourly` summation logic as a validation fallback in case sub-monthly granularity is required in a future iteration.
3. **Do Not Use Method C:**
   Never use raw hourly `total_precipitation` simple sum.

---

## 6. Official Sources
* GEE ERA5-Land Monthly Aggregated: https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR
* GEE ERA5-Land Hourly: https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY
* ECMWF ERA5-Land Documentation: https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation
