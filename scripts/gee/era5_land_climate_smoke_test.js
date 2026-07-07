// ERA5-Land GEE Smoke Test Script
// ------------------------------------------------------------
// Template for a tiny smoke test of ERA5-Land climate extraction.
// No private outbreak coordinates are included. The script must be
// edited before execution in the Earth Engine Code Editor.
//
// Comments:
// - This script uses monthly pre-aggregated ERA5-Land (ECMWF/ERA5_LAND/MONTHLY_AGGR).
// - Do not use raw hourly total_precipitation simple sum (causes double-counting).
// - Snow and freeze-thaw are future extensions.

// ----------------------- Configuration -----------------------
var CSD_ASSET_ID = 'users/YOUR_USERNAME/YOUR_CSD_ASSET'; // Replace with your CSD asset ID
var TEST_CSD_UIDS = ['PLACEHOLDER_1', 'PLACEHOLDER_2']; // 5-10 CSD uid strings
var TEST_YEAR = 2022; // Replace with desired year
var TEST_MONTH = 1;   // Replace with desired month (1-12)
var EXPORT_DESCRIPTION = 'era5_land_climate_smoke_test';
var EXPORT_FOLDER = 'hpai_gee_exports'; // Google Drive folder name
var ERA5_SCALE_METERS = 10000; // Approximate scale for ERA5-Land; verify if needed

// --------------------- Load CSD Feature Collection ---------------------
var csdFc = ee.FeatureCollection(CSD_ASSET_ID)
    .filter(ee.Filter.inList('csd_uid', TEST_CSD_UIDS));
print('Number of test CSDs', csdFc.size());
print('First test CSD feature', csdFc.first());

// --------------------- ERA5-Land Collection ---------------------
// Using monthly pre-aggregated ERA5-Land.
var ERA5_COLLECTION_ID = 'ECMWF/ERA5_LAND/MONTHLY_AGGR';
var era5 = ee.ImageCollection(ERA5_COLLECTION_ID);
print('ERA5-Land monthly collection (first image)');
print(era5.first());
print('Band names of first image', era5.first().bandNames());

// --------------------- Date Handling & Filtering ---------------------
var startDate = ee.Date.fromYMD(TEST_YEAR, TEST_MONTH, 1);
var endDate   = startDate.advance(1, 'month');
var era5Month = era5.filterDate(startDate, endDate);

// Get the single monthly image
var monthlyImage = era5Month.first();

// --------------------- Variable Band Names ---------------------
var TEMP_BAND       = 'temperature_2m';
var TEMP_MIN_BAND   = 'temperature_2m_min';
var TEMP_MAX_BAND   = 'temperature_2m_max';
var PRECIP_BAND     = 'total_precipitation_sum';

// --------------------- Unit Conversions ---------------------
// Temperature: Kelvin to degrees C
var K_TO_C = -273.15;
var tMeanC = monthlyImage.select(TEMP_BAND).add(K_TO_C).rename('climate_tmean_c');
var tMinC  = monthlyImage.select(TEMP_MIN_BAND).add(K_TO_C).rename('climate_tmin_c');
var tMaxC  = monthlyImage.select(TEMP_MAX_BAND).add(K_TO_C).rename('climate_tmax_c');

// Precipitation: meters to millimeters (1 m = 1000 mm)
var precipMm = monthlyImage.select(PRECIP_BAND).multiply(1000).rename('climate_precip_mm');

// --------------------- Combine Bands ---------------------
var climateImage = ee.Image.cat([tMeanC, tMinC, tMaxC, precipMm]);

// --------------------- Spatial Reduction ---------------------
var reduced = climateImage.reduceRegions({
  collection: csdFc,
  reducer: ee.Reducer.mean(),
  scale: ERA5_SCALE_METERS,
});
var withYearMonth = reduced.map(function(f) {
  var withTime = f.set('year', TEST_YEAR).set('month', TEST_MONTH);
  return ee.Feature(null, withTime.toDictionary([
    'csd_uid',
    'csd_name',
    'province_u',
    'province_n',
    'year',
    'month',
    'climate_tmean_c',
    'climate_tmin_c',
    'climate_tmax_c',
    'climate_precip_mm'
  ]));
});
print('Output preview', withYearMonth.first());

// --------------------- Export ---------------------
Export.table.toDrive({
  collection: withYearMonth,
  description: EXPORT_DESCRIPTION,
  folder: EXPORT_FOLDER,
  fileNamePrefix: EXPORT_DESCRIPTION + '_' + TEST_YEAR + '_' + TEST_MONTH,
  fileFormat: 'CSV'
});

// ------------------------------------------------------------
// Safety notes:
// * No private outbreak coordinates are included.
// * This is a smoke test template; full extraction requires validation.
// * Exported CSV is a QA artifact and should not be committed.
// ------------------------------------------------------------
