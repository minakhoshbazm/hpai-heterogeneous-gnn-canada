// Dynamic World Land Cover CSD-Level Monthly Extraction (Ontario 2022 Pilot)
// -------------------------------------------------------------------------
// This script extracts monthly CSD-level land cover probabilities and
// seasonal pond proxy features from GOOGLE/DYNAMICWORLD/V1 for CSDs in
// Ontario only (province_u = "35") for the year 2022.
// Scale is set to 30m with tileScale = 4 to reduce computation overhead.
// -------------------------------------------------------------------------

// --------------------------- Configuration ---------------------------
var CSD_ASSET_ID = 'projects/avianinfluenza/assets/canada_csd_2021_boundaries';
var DW_COLLECTION_ID = 'GOOGLE/DYNAMICWORLD/V1';
var YEAR = 2022;
var PROVINCE_U = '35';
var EXPORT_DESCRIPTION = 'dynamic_world_ontario_2022_pilot';
var EXPORT_FOLDER = 'hpai_gee_exports';

// ------------------------- Load Collections -------------------------
var csdFc = ee.FeatureCollection(CSD_ASSET_ID)
    .filter(ee.Filter.eq('province_u', PROVINCE_U));
var dw = ee.ImageCollection(DW_COLLECTION_ID);

// ------------------------- Fallback Extraction -------------------------
var monthsList = ee.List.sequence(1, 12);

var monthlyExtractions = monthsList.map(function(monthNum) {
  var month = ee.Number(monthNum);
  var startDate = ee.Date.fromYMD(YEAR, month, 1);
  var endDate = startDate.advance(1, 'month');
  
  // Filter collection for the target month
  var dwMonth = dw.filterDate(startDate, endDate);
  var imageCount = dwMonth.size();
  
  // Generate mean composite and rename bands
  var meanImage = dwMonth.select(['water', 'flooded_vegetation', 'crops', 'grass', 'trees', 'built', 'bare'])
      .mean()
      .rename(['dw_water_mean', 'dw_flooded_vegetation_mean', 'dw_crops_mean', 'dw_grass_mean', 'dw_trees_mean', 'dw_built_mean', 'dw_bare_mean']);
      
  // Generate 95th percentile water probability image and rename (temporal reduction over the month)
  var waterP95Image = dwMonth.select('water')
      .reduce(ee.Reducer.percentile([95]))
      .rename(['dw_water_p95']);
      
  // Combine mean and p95 images
  var combinedImage = ee.Image.cat([meanImage, waterP95Image]);
  
  // Run a single spatial reduction over CSDs with scale=30 and tileScale=4
  var reduced = combinedImage.reduceRegions({
    collection: csdFc,
    reducer: ee.Reducer.mean(),
    scale: 30,
    tileScale: 4
  });
  
  // Post-process properties (compute pond proxy, missing flag, and format dictionary)
  var processed = reduced.map(function(f) {
    var isNullWater = ee.Algorithms.IsEqual(f.get('dw_water_mean'), null);
    var isNullP95 = ee.Algorithms.IsEqual(f.get('dw_water_p95'), null);
    var isNullFlood = ee.Algorithms.IsEqual(f.get('dw_flooded_vegetation_mean'), null);
    var isNullCrops = ee.Algorithms.IsEqual(f.get('dw_crops_mean'), null);
    var isNullGrass = ee.Algorithms.IsEqual(f.get('dw_grass_mean'), null);
    var isNullTrees = ee.Algorithms.IsEqual(f.get('dw_trees_mean'), null);
    var isNullBuilt = ee.Algorithms.IsEqual(f.get('dw_built_mean'), null);
    var isNullBare = ee.Algorithms.IsEqual(f.get('dw_bare_mean'), null);
    var isCountZero = ee.Algorithms.IsEqual(imageCount, 0);
    
    var missingFlag = ee.Number(
      ee.Algorithms.If(isCountZero, 1,
      ee.Algorithms.If(isNullWater, 1,
      ee.Algorithms.If(isNullP95, 1,
      ee.Algorithms.If(isNullFlood, 1,
      ee.Algorithms.If(isNullCrops, 1,
      ee.Algorithms.If(isNullGrass, 1,
      ee.Algorithms.If(isNullTrees, 1,
      ee.Algorithms.If(isNullBuilt, 1,
      ee.Algorithms.If(isNullBare, 1, 0)))))))))
    );
    
    var waterMean = ee.Number(f.get('dw_water_mean'));
    var waterP95 = ee.Number(f.get('dw_water_p95'));
    var proxy = ee.Algorithms.If(
      ee.Algorithms.IsEqual(waterMean, null),
      null,
      ee.Algorithms.If(
        ee.Algorithms.IsEqual(waterP95, null),
        null,
        waterP95.subtract(waterMean)
      )
    );
    
    var outDict = ee.Dictionary({
      'csd_uid': f.get('csd_uid'),
      'csd_name': f.get('csd_name'),
      'province_u': f.get('province_u'),
      'year': YEAR,
      'month': month,
      'dw_water_mean': f.get('dw_water_mean'),
      'dw_water_p95': f.get('dw_water_p95'),
      'dw_flooded_vegetation_mean': f.get('dw_flooded_vegetation_mean'),
      'dw_crops_mean': f.get('dw_crops_mean'),
      'dw_grass_mean': f.get('dw_grass_mean'),
      'dw_trees_mean': f.get('dw_trees_mean'),
      'dw_built_mean': f.get('dw_built_mean'),
      'dw_bare_mean': f.get('dw_bare_mean'),
      'dw_seasonal_pond_proxy': proxy,
      'dw_image_count': imageCount,
      'dw_missing_flag': missingFlag
    });
    
    return ee.Feature(null, outDict);
  });
  
  return processed;
});

var finalCollection = ee.FeatureCollection(monthlyExtractions).flatten();

// --------------------------- Prints ---------------------------
print('Selected province: Ontario (province_u = "35")');
print('CSD count for Ontario:', csdFc.size());
print('Expected row count text: Ontario CSD count * 12');
print('Year:', YEAR);

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
    'year',
    'month',
    'dw_water_mean',
    'dw_water_p95',
    'dw_flooded_vegetation_mean',
    'dw_crops_mean',
    'dw_grass_mean',
    'dw_trees_mean',
    'dw_built_mean',
    'dw_bare_mean',
    'dw_seasonal_pond_proxy',
    'dw_image_count',
    'dw_missing_flag'
  ]
});
