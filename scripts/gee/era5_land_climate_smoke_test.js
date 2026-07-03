// ERA5-Land GEE Smoke Test Script
// ------------------------------------------------------------
// Template for a tiny smoke test of ERA5-Land climate extraction.
// No private outbreak coordinates are included. The script must be
// edited before execution in the Earth Engine Code Editor.

// ----------------------- Configuration -----------------------
var CSD_ASSET_ID = 'users/YOUR_USERNAME/YOUR_CSD_ASSET'; // Replace with your CSD asset ID
var TEST_CSD_UIDS = ['PLACEHOLDER_1', 'PLACEHOLDER_2']; // 5-10 CSD uid strings
var TEST_YEAR = 2022; // Replace with desired year
var TEST_MONTH = 1;   // Replace with desired month (1-12)
var EXPORT_DESCRIPTION = 'era5_land_climate_smoke_test';
var EXPORT_FOLDER = 'hpai_gee_exports'; // Google Drive folder name
var INCLUDE_SNOW = false; // set true only after confirming snow_depth band exists
var ERA5_SCALE_METERS = 10000; // Approximate scale for ERA5-Land; verify if needed

// --------------------- Load CSD Feature Collection ---------------------
var csdFc = ee.FeatureCollection(CSD_ASSET_ID)
    .filter(ee.Filter.inList('csd_uid', TEST_CSD_UIDS));
print('Number of test CSDs', csdFc.size());
print('First test CSD feature', csdFc.first());

// --------------------- ERA5-Land Collection ---------------------
// Verify collection ID before running. Suggested ID:
// "ECMWF/ERA5_LAND/HOURLY"
var ERA5_COLLECTION_ID = 'ECMWF/ERA5_LAND/HOURLY'; // <-- verify
var era5 = ee.ImageCollection(ERA5_COLLECTION_ID);
print('ERA5-Land collection (first image)');
print(era5.first());
print('Band names of first image', era5.first().bandNames());

// --------------------- Date Handling ---------------------
var startDate = ee.Date.fromYMD(TEST_YEAR, TEST_MONTH, 1);
var endDate   = startDate.advance(1, 'month');
var era5Month = era5.filterDate(startDate, endDate);

// --------------------- Variable Band Names ---------------------
// Verify that these bands exist in the collection before running.
var TEMP_BAND   = 'temperature_2m';
var PRECIP_BAND = 'total_precipitation';
var SNOW_BAND   = 'snow_depth'; // optional - may be missing

// --------------------- Monthly Aggregation ---------------------
// Temperature aggregates (Kelvin)
var tempMean = era5Month.select(TEMP_BAND).mean();
var tempMin  = era5Month.select(TEMP_BAND).min();
var tempMax  = era5Month.select(TEMP_BAND).max();

// Precipitation aggregate (meters)
var precipSum = era5Month.select(PRECIP_BAND).sum();

// Optional snow depth aggregate (meters)
var snowMean = INCLUDE_SNOW ? era5Month.select(SNOW_BAND).mean() : null;

// --------------------- Unit Conversions ---------------------
// Temperature: Kelvin to degrees C
var K_TO_C = -273.15;
var tMeanC = tempMean.add(K_TO_C).rename('climate_tmean_c');
var tMinC  = tempMin.add(K_TO_C).rename('climate_tmin_c');
var tMaxC  = tempMax.add(K_TO_C).rename('climate_tmax_c');

// Precipitation: meters to millimeters (1 m = 1000 mm)
var precipMm = precipSum.multiply(1000).rename('climate_precip_mm');

// Snow depth: meters to centimeters (1 m = 100 cm)
var snowCm = INCLUDE_SNOW ? snowMean.multiply(100).rename('climate_snow_depth_cm') : null;

// --------------------- Combine Bands ---------------------
var bandList = [tMeanC, tMinC, tMaxC, precipMm];
if (INCLUDE_SNOW) {
  bandList.push(snowCm);
}
var climateImage = ee.Image.cat(bandList);

// --------------------- Spatial Reduction ---------------------
var reduced = climateImage.reduceRegions({
  collection: csdFc,
  reducer: ee.Reducer.mean(),
  scale: ERA5_SCALE_METERS,
});
var withYearMonth = reduced.map(function(f) {
  return f.set('year', TEST_YEAR).set('month', TEST_MONTH);
});

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
