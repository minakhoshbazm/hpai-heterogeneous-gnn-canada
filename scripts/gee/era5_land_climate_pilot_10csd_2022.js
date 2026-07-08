// ERA5-Land GEE Pilot Extraction Script (10 CSDs, 2022)
// ------------------------------------------------------------
// Extracts monthly climate features for 10 representative CSDs
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
var TEST_CSD_UIDS = [
  '1001101', '1101002', '1201001', '1301001', '2401023',
  '3501005', '4601035', '4701001', '4801003', '5929028'
];
var TEST_YEAR = 2022;
var EXPORT_DESCRIPTION = 'era5_land_climate_pilot_10csd_2022';
var EXPORT_FOLDER = 'hpai_gee_exports'; // Google Drive folder name
var ERA5_SCALE_METERS = 10000; // Native ERA5-Land resolution is ~9 km

// --------------------- Load CSD Feature Collection ---------------------
var csdFc = ee.FeatureCollection(CSD_ASSET_ID)
    .filter(ee.Filter.inList('csd_uid', TEST_CSD_UIDS));
print('Selected CSD count', csdFc.size());

// --------------------- ERA5-Land Monthly Collection ---------------------
var ERA5_COLLECTION_ID = 'ECMWF/ERA5_LAND/MONTHLY_AGGR';
var era5 = ee.ImageCollection(ERA5_COLLECTION_ID);

// --------------------- Monthly Loop (1 to 12) ---------------------
var months = ee.List.sequence(1, 12);

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
    
    var withProperties = f.set({
      'year': TEST_YEAR,
      'month': month,
      'climate_missing_flag': missingFlag
    });
    
    // Discard geometry by creating a Feature with null geometry
    return ee.Feature(null, withProperties.toDictionary([
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
    ]));
  });
  
  return processed;
});

// Flatten the feature collections into a single table
var finalCollection = ee.FeatureCollection(monthlyExtractions).flatten();

// --------------------- Prints ---------------------
print('Monthly output row count (Expected: 120)', finalCollection.size());
print('First 10 output rows', finalCollection.limit(10));

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
