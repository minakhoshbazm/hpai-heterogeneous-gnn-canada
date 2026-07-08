// ERA5-Land GEE Fallback Smoke Test Script
// ------------------------------------------------------------
// Evaluates candidate fallback methods (polygon mean, centroid point sample,
// and buffers of 25 km, 50 km, and 100 km) on a select set of missing and
// control CSDs for January and July 2022.

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
var TEST_CSD_UIDS = [
  '1001120', '1001267', '1001270', '1001375', '1001494',
  '1003028', '1003026', '2401023', // Known missing
  '1001101', '3501005', '4801003'  // Controls
];
var TEST_YEAR = 2022;
var TEST_MONTHS = [1, 7];
var EXPORT_DESCRIPTION = 'era5_fallback_smoke_test';
var EXPORT_FOLDER = 'hpai_gee_exports';
var ERA5_SCALE_METERS = 10000;

// --------------------- Load CSD Feature Collection ---------------------
var csdFc = ee.FeatureCollection(CSD_ASSET_ID)
    .filter(ee.Filter.inList('csd_uid', TEST_CSD_UIDS));
print('Selected CSD count (Expected: 11)', csdFc.size());

// --------------------- ERA5-Land Monthly Collection ---------------------
var ERA5_COLLECTION_ID = 'ECMWF/ERA5_LAND/MONTHLY_AGGR';
var era5 = ee.ImageCollection(ERA5_COLLECTION_ID);

// --------------------- Main Execution Loop ---------------------
var monthsList = ee.List(TEST_MONTHS);

var monthlyExtractions = monthsList.map(function(month) {
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
  
  // Define fallback geometries
  var geomOriginal = csdFc;
  
  var geomCentroid = csdFc.map(function(f) {
    return ee.Feature(f.geometry().centroid(), f.toDictionary());
  });
  
  var geomBuffer25 = csdFc.map(function(f) {
    return ee.Feature(f.geometry().centroid().buffer(25000), f.toDictionary());
  });
  
  var geomBuffer50 = csdFc.map(function(f) {
    return ee.Feature(f.geometry().centroid().buffer(50000), f.toDictionary());
  });
  
  var geomBuffer100 = csdFc.map(function(f) {
    return ee.Feature(f.geometry().centroid().buffer(100000), f.toDictionary());
  });
  
  // Helper to run spatial reduction and format properties
  var extractAndFormat = function(fc, methodName) {
    var reduced = climateImage.reduceRegions({
      collection: fc,
      reducer: ee.Reducer.mean(),
      scale: ERA5_SCALE_METERS
    });
    
    return reduced.map(function(f) {
      var missingFlag = missingAsNumber(f, 'climate_tmean_c')
        .max(missingAsNumber(f, 'climate_tmin_c'))
        .max(missingAsNumber(f, 'climate_tmax_c'))
        .max(missingAsNumber(f, 'climate_precip_mm'));
      
      var outDict = ee.Dictionary({
        'csd_uid': f.get('csd_uid'),
        'csd_name': f.get('csd_name'),
        'province_u': f.get('province_u'),
        'year': TEST_YEAR,
        'month': month,
        'method_name': methodName,
        'climate_tmean_c': f.get('climate_tmean_c'),
        'climate_tmin_c': f.get('climate_tmin_c'),
        'climate_tmax_c': f.get('climate_tmax_c'),
        'climate_precip_mm': f.get('climate_precip_mm'),
        'method_missing_flag': missingFlag
      });
      
      return ee.Feature(null, outDict);
    });
  };
  
  // Run all 5 methods
  var resOriginal = extractAndFormat(geomOriginal, 'polygon_mean');
  var resCentroid = extractAndFormat(geomCentroid, 'centroid_sample');
  var resBuffer25 = extractAndFormat(geomBuffer25, 'buffer_25km');
  var resBuffer50 = extractAndFormat(geomBuffer50, 'buffer_50km');
  var resBuffer100 = extractAndFormat(geomBuffer100, 'buffer_100km');
  
  // Merge methods together for this month
  return resOriginal
    .merge(resCentroid)
    .merge(resBuffer25)
    .merge(resBuffer50)
    .merge(resBuffer100);
});

// Flatten the feature collections into a single table
var finalCollection = ee.FeatureCollection(monthlyExtractions).flatten();

// --------------------- Prints ---------------------
print('Expected row count: 110');
print('Actual output row count', finalCollection.size());
print('First 20 output rows', finalCollection.limit(20));
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
    'year',
    'month',
    'method_name',
    'climate_tmean_c',
    'climate_tmin_c',
    'climate_tmax_c',
    'climate_precip_mm',
    'method_missing_flag'
  ]
});
