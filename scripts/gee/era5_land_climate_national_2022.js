// ERA5-Land GEE National Pilot Extraction Script (All CSDs, 2022)
// ------------------------------------------------------------
// Extracts monthly climate features for all 5,161 Canadian CSDs
// across all 12 months of the year 2022.
//
// Comments:
// - This script uses monthly pre-aggregated ERA5-Land (ECMWF/ERA5_LAND/MONTHLY_AGGR).
// - Do not use raw hourly total_precipitation simple sum (causes double-counting).
// - Dropping geometry explicitly from exported rows.
// - Checking for missing values: climate_missing_flag = 1 if any field is null.
// - Snow and freeze-thaw are future extensions.

// ----------------------- Helper Functions -----------------------
function missingAsNumber(feature, propertyName) {
  return ee.Number(
    ee.Algorithms.If(
      ee.Algorithms.IsEqual(feature.get(propertyName), null),
      1,
      0
    )
  );
}

// ----------------------- Configuration -----------------------
var CSD_ASSET_ID = 'projects/avianinfluenza/assets/canada_csd_2021_boundaries';
var TEST_YEAR = 2022;
var EXPORT_DESCRIPTION = 'era5_land_climate_national_2022';
var EXPORT_FOLDER = 'hpai_gee_exports'; // Google Drive folder name
var ERA5_SCALE_METERS = 10000; // Native ERA5-Land resolution is ~9 km

// --------------------- Load CSD Feature Collection ---------------------
var csdFc = ee.FeatureCollection(CSD_ASSET_ID);
print('Total CSD count (Expected: 5161)', csdFc.size());

// --------------------- ERA5-Land Monthly Collection ---------------------
var ERA5_COLLECTION_ID = 'ECMWF/ERA5_LAND/MONTHLY_AGGR';
var era5 = ee.ImageCollection(ERA5_COLLECTION_ID);

// --------------------- Monthly Loop (1 to 12) ---------------------
var months = ee.List.sequence(1, 12);

// Verify image count per month (should be exactly 1 for MONTHLY_AGGR)
var monthlyImageCounts = months.map(function(month) {
  month = ee.Number(month);
  var startDate = ee.Date.fromYMD(TEST_YEAR, month, 1);
  var endDate   = startDate.advance(1, 'month');
  var count     = era5.filterDate(startDate, endDate).size();
  return ee.Feature(null, {
    'year': TEST_YEAR,
    'month': month,
    'image_count': count
  });
});
var monthlyImageCountsFc = ee.FeatureCollection(monthlyImageCounts);

var monthlyExtractions = months.map(function(month) {
  month = ee.Number(month);
  
  // Date range for this month
  var startDate = ee.Date.fromYMD(TEST_YEAR, month, 1);
  var endDate   = startDate.advance(1, 'month');
  
  // Filter collection
  var era5Month = era5.filterDate(startDate, endDate);
  var monthlyImage = era5Month.first();
  
  // Select and convert temperature from Kelvin to Celsius
  var K_TO_C = -273.15;
  var tMeanC = monthlyImage.select('temperature_2m').add(K_TO_C).rename('climate_tmean_c');
  var tMinC  = monthlyImage.select('temperature_2m_min').add(K_TO_C).rename('climate_tmin_c');
  var tMaxC  = monthlyImage.select('temperature_2m_max').add(K_TO_C).rename('climate_tmax_c');
  
  // Convert precipitation from meters to millimeters
  var precipMm = monthlyImage.select('total_precipitation_sum').multiply(1000).rename('climate_precip_mm');
  
  // Combine converted bands
  var climateImage = ee.Image.cat([tMeanC, tMinC, tMaxC, precipMm]);
  
  // Perform spatial reduction (polygon mean) over the CSDs
  var reduced = climateImage.reduceRegions({
    collection: csdFc,
    reducer: ee.Reducer.mean(),
    scale: ERA5_SCALE_METERS
  });
  
  // Process properties, drop geometry, and handle missing values
  var processed = reduced.map(function(f) {
    var missingFlag = missingAsNumber(f, 'climate_tmean_c')
      .max(missingAsNumber(f, 'climate_tmin_c'))
      .max(missingAsNumber(f, 'climate_tmax_c'))
      .max(missingAsNumber(f, 'climate_precip_mm'));
    
    var outDict = ee.Dictionary({
      'csd_uid': f.get('csd_uid'),
      'csd_name': f.get('csd_name'),
      'province_u': f.get('province_u'),
      'province_n': f.get('province_n'),
      'year': TEST_YEAR,
      'month': month,
      'climate_tmean_c': f.get('climate_tmean_c'),
      'climate_tmin_c': f.get('climate_tmin_c'),
      'climate_tmax_c': f.get('climate_tmax_c'),
      'climate_precip_mm': f.get('climate_precip_mm'),
      'climate_missing_flag': missingFlag
    });
    
    return ee.Feature(null, outDict);
  });
  
  return processed;
});

// Flatten the feature collections into a single table
var finalCollection = ee.FeatureCollection(monthlyExtractions).flatten();

// --------------------- Prints ---------------------
print('Monthly image counts (should all be 1)', monthlyImageCountsFc);
print('Expected row count: 61932');
print('Actual output row count', finalCollection.size());
print('First 10 output rows', finalCollection.limit(10));
print('First output feature', finalCollection.first());
print('First output feature property names', finalCollection.first().propertyNames());

// --------------------- Export to Google Drive ---------------------
Export.table.toDrive({
  collection: finalCollection,
  description: EXPORT_DESCRIPTION,
  folder: EXPORT_FOLDER,
  fileNamePrefix: EXPORT_DESCRIPTION,
  fileFormat: 'CSV',
  selectors: [
    'csd_uid',
    'csd_name',
    'province_u',
    'province_n',
    'year',
    'month',
    'climate_tmean_c',
    'climate_tmin_c',
    'climate_tmax_c',
    'climate_precip_mm',
    'climate_missing_flag'
  ]
});
