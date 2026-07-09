// ERA5-Land Monthly Climate Extraction with Fallback (National 2024 - Export Only)
// -------------------------------------------------------------------------
// This script extracts monthly ERA5-Land climate features for all 5,161 CSDs
// across Canada for the entire year of 2024. It implements a hierarchical
// fallback cascade to resolve small, coastal, and island CSDs that have
// missing climate values due to land-masking constraints.
//
// Fallback Cascade:
// 1. polygon_mean (original CSD boundary)
// 2. buffer_25km  (25 km centroid buffer mean)
// 3. buffer_50km  (50 km centroid buffer mean)
// 4. buffer_100km (100 km centroid buffer mean)
// 5. missing      (assigned if still unresolved after 100 km buffer)
// -------------------------------------------------------------------------

// --------------------------- Configuration ---------------------------
var CSD_ASSET_ID = 'projects/avianinfluenza/assets/canada_csd_2021_boundaries';
var ERA5_COLLECTION_ID = 'ECMWF/ERA5_LAND/MONTHLY_AGGR';
var YEAR = 2024;
var ERA5_SCALE_METERS = 10000;
var EXPORT_DESCRIPTION = 'era5_land_climate_national_2024_with_fallback';
var EXPORT_FOLDER = 'hpai_gee_exports';

// ------------------------- Load Collections -------------------------
var csdFc = ee.FeatureCollection(CSD_ASSET_ID).map(function(f) {
  var centroid = f.geometry().centroid();
  var coords = centroid.coordinates();
  return f.set({
    'orig_centroid_lon': coords.get(0),
    'orig_centroid_lat': coords.get(1)
  });
});
var era5 = ee.ImageCollection(ERA5_COLLECTION_ID);

// ------------------------- Fallback Extraction -------------------------
var monthsList = ee.List.sequence(1, 12);

var monthlyExtractions = monthsList.map(function(monthNum) {
  var month = ee.Number(monthNum);
  var startDate = ee.Date.fromYMD(YEAR, month, 1);
  var endDate = startDate.advance(1, 'month');
  
  // Filter collection for the month
  var era5Month = era5.filterDate(startDate, endDate);
  var monthlyImage = era5Month.first();
  
  // Select and convert temperature from Kelvin to Celsius
  var K_TO_C = -273.15;
  var tMeanC = monthlyImage.select('temperature_2m').add(K_TO_C).rename('climate_tmean_c');
  var tMinC  = monthlyImage.select('temperature_2m_min').add(K_TO_C).rename('climate_tmin_c');
  var tMaxC  = monthlyImage.select('temperature_2m_max').add(K_TO_C).rename('climate_tmax_c');
  
  // Convert precipitation from meters to millimeters
  var precipMm = monthlyImage.select('total_precipitation_sum').multiply(1000).rename('climate_precip_mm');
  
  // Combine bands
  var climateImage = ee.Image.cat([tMeanC, tMinC, tMaxC, precipMm]);
  
  // Helper function to check for missing climate properties
  var computeMissingFlag = function(f) {
    var isNullMean = ee.Algorithms.IsEqual(f.get('climate_tmean_c'), null);
    var isNullMin = ee.Algorithms.IsEqual(f.get('climate_tmin_c'), null);
    var isNullMax = ee.Algorithms.IsEqual(f.get('climate_tmax_c'), null);
    var isNullPrecip = ee.Algorithms.IsEqual(f.get('climate_precip_mm'), null);
    
    var missingFlag = ee.Number(
      ee.Algorithms.If(isNullMean, 1,
      ee.Algorithms.If(isNullMin, 1,
      ee.Algorithms.If(isNullMax, 1,
      ee.Algorithms.If(isNullPrecip, 1, 0))))
    );
    
    return f.set('climate_missing_flag', missingFlag);
  };
  
  // --- TIER 1: polygon_mean (Default) ---
  var reduced1 = climateImage.reduceRegions({
    collection: csdFc,
    reducer: ee.Reducer.mean(),
    scale: ERA5_SCALE_METERS
  }).map(computeMissingFlag);
  
  var resolved1 = reduced1.filter(ee.Filter.eq('climate_missing_flag', 0))
    .map(function(f) {
      return f.set({
        'climate_extraction_method': 'polygon_mean',
        'climate_fallback_distance_km': 0
      });
    });
    
  var unresolved1 = reduced1.filter(ee.Filter.eq('climate_missing_flag', 1));
  
  // --- TIER 2: buffer_25km ---
  var geom2 = unresolved1.map(function(f) {
    var lon = ee.Number(f.get('orig_centroid_lon'));
    var lat = ee.Number(f.get('orig_centroid_lat'));
    var centroid = ee.Geometry.Point([lon, lat]);
    return f.setGeometry(centroid.buffer(25000));
  });
  
  var reduced2 = climateImage.reduceRegions({
    collection: geom2,
    reducer: ee.Reducer.mean(),
    scale: ERA5_SCALE_METERS
  }).map(computeMissingFlag);
  
  var resolved2 = reduced2.filter(ee.Filter.eq('climate_missing_flag', 0))
    .map(function(f) {
      return f.set({
        'climate_extraction_method': 'buffer_25km',
        'climate_fallback_distance_km': 25
      });
    });
    
  var unresolved2 = reduced2.filter(ee.Filter.eq('climate_missing_flag', 1));
  
  // --- TIER 3: buffer_50km ---
  var geom3 = unresolved2.map(function(f) {
    var lon = ee.Number(f.get('orig_centroid_lon'));
    var lat = ee.Number(f.get('orig_centroid_lat'));
    var centroid = ee.Geometry.Point([lon, lat]);
    return f.setGeometry(centroid.buffer(50000));
  });
  
  var reduced3 = climateImage.reduceRegions({
    collection: geom3,
    reducer: ee.Reducer.mean(),
    scale: ERA5_SCALE_METERS
  }).map(computeMissingFlag);
  
  var resolved3 = reduced3.filter(ee.Filter.eq('climate_missing_flag', 0))
    .map(function(f) {
      return f.set({
        'climate_extraction_method': 'buffer_50km',
        'climate_fallback_distance_km': 50
      });
    });
    
  var unresolved3 = reduced3.filter(ee.Filter.eq('climate_missing_flag', 1));
  
  // --- TIER 4: buffer_100km ---
  var geom4 = unresolved3.map(function(f) {
    var lon = ee.Number(f.get('orig_centroid_lon'));
    var lat = ee.Number(f.get('orig_centroid_lat'));
    var centroid = ee.Geometry.Point([lon, lat]);
    return f.setGeometry(centroid.buffer(100000));
  });
  
  var reduced4 = climateImage.reduceRegions({
    collection: geom4,
    reducer: ee.Reducer.mean(),
    scale: ERA5_SCALE_METERS
  }).map(computeMissingFlag);
  
  var resolved4 = reduced4.filter(ee.Filter.eq('climate_missing_flag', 0))
    .map(function(f) {
      return f.set({
        'climate_extraction_method': 'buffer_100km',
        'climate_fallback_distance_km': 100
      });
    });
    
  var unresolved4 = reduced4.filter(ee.Filter.eq('climate_missing_flag', 1))
    .map(function(f) {
      return f.set({
        'climate_extraction_method': 'missing',
        'climate_fallback_distance_km': null
      });
    });
    
  // --- Merge All Tiers ---
  var monthlyMerged = resolved1
    .merge(resolved2)
    .merge(resolved3)
    .merge(resolved4)
    .merge(unresolved4);
    
  // Drop geometry and keep target schema columns
  return monthlyMerged.map(function(f) {
    var outDict = ee.Dictionary({
      'csd_uid': f.get('csd_uid'),
      'csd_name': f.get('csd_name'),
      'province_u': f.get('province_u'),
      'province_n': f.get('province_n'),
      'year': YEAR,
      'month': month,
      'climate_tmean_c': f.get('climate_tmean_c'),
      'climate_tmin_c': f.get('climate_tmin_c'),
      'climate_tmax_c': f.get('climate_tmax_c'),
      'climate_precip_mm': f.get('climate_precip_mm'),
      'climate_missing_flag': f.get('climate_missing_flag'),
      'climate_extraction_method': f.get('climate_extraction_method'),
      'climate_fallback_distance_km': f.get('climate_fallback_distance_km')
    });
    return ee.Feature(null, outDict);
  });
});

// Flatten into a single feature collection
var finalCollection = ee.FeatureCollection(monthlyExtractions).flatten();

// --------------------------- Prints ---------------------------
print('Total CSD count:', csdFc.size());
// Print monthly image counts to verify each month has exactly 1 aggregate image
var monthlyImageCounts = monthsList.map(function(monthNum) {
  var month = ee.Number(monthNum);
  var startDate = ee.Date.fromYMD(YEAR, month, 1);
  var endDate = startDate.advance(1, 'month');
  var count = era5.filterDate(startDate, endDate).size();
  return ee.Feature(null, {
    'month': month,
    'image_count': count
  });
});
print('Monthly image counts (Expected 1 per month):', ee.FeatureCollection(monthlyImageCounts));
print('Expected row count: 61932');

// Heavy Console prints commented out to prevent GEE Console timeout
// print('Actual output row count:', finalCollection.size());
// print('First output feature:', finalCollection.first());
// print('First output feature property names:', finalCollection.first().propertyNames());
// print('Count by climate_extraction_method:', finalCollection.aggregate_histogram('climate_extraction_method'));

// ------------------------- Export to Drive -------------------------
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
    'climate_missing_flag',
    'climate_extraction_method',
    'climate_fallback_distance_km'
  ]
});
