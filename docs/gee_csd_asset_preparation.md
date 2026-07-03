# Preparing the CSD Boundary Asset for the ERA5-Land GEE smoke test

## 1. Source Boundary File

- Use the official Canadian Census Subdivision (CSD) boundary file released by Statistics Canada (e.g., the 2021 Census shapefile or GeoJSON). This file contains a polygon for every CSD in Canada and is publicly available.

## 2. Required Fields

| Property      | Type   | Required? | Description |
|---------------|--------|-----------|-------------|
| `csd_uid`     | string | Yes       | 7-character CSD identifier, preserve leading zeros |
| `csd_name`    | string | Optional  | CSD name if available |
| `province_uid`| string | Optional  | Province or territory identifier if available |
| `province_name`| string | Optional | Province or territory name if available |

Only `csd_uid` is required for the smoke test; the other fields are optional but useful for downstream analysis.

## 3. Privacy Guidelines

- Upload only public CSD boundary geometry.  
- Do NOT upload CFIA labels, exact outbreak coordinates, or private QA maps.  
- The asset must contain no private or sensitive data; it is intended solely for geometric reference.

## 4. Geometry Requirements

- **Coordinate Reference System:** Export or upload the file in EPSG:4326 (WGS84). This is the native CRS expected by Google Earth Engine.  
- **Simplification:** Simplify geometries only if necessary for performance (e.g., to keep file size reasonable). If you simplify, log the tolerance you used (e.g., `tolerance = 0.0001` degrees) and keep a copy of the unsimplified version for reference.  
- **Validity:** Ensure polygons are valid and have no self-intersections; GEE will reject malformed geometries.

If you use a simplified boundary asset, save the simplification script or settings in this documentation.

## 5. Suggested Asset Name

```
users/YOUR_USERNAME/canada_csd_2021_boundaries
```
Replace `YOUR_USERNAME` with your GEE account name.

## 6. Selecting 5-10 Test CSDs

When you create the test list (`TEST_CSD_UIDS` in the smoke-test script), aim for a diverse sample:

1. Different provinces: choose at least one CSD from three separate provinces or territories.  
2. Urban example: include a small-to-medium urban CSD.  
3. Rural/agricultural example: include a predominantly rural CSD.  
4. Coastal or island example: if possible, pick a CSD on the coast or an island.  
5. Size variety: vary the physical size of the CSDs (small, medium, large).

Write the chosen `csd_uid`s into the JavaScript file as a simple array, e.g.:
```javascript
var TEST_CSD_UIDS = ['5905001', '5907001', '1203002', '1106003', '1304004'];
```

## 7. Post-upload Checks

After you have uploaded the asset to GEE, run these quick verification steps in the Code Editor:

1. Feature count:
```javascript
print('Number of CSD features', ee.FeatureCollection('users/YOUR_USERNAME/canada_csd_2021_boundaries').size());
```
2. First feature properties:
```javascript
var first = ee.FeatureCollection('users/YOUR_USERNAME/canada_csd_2021_boundaries').first();
print('First feature', first);
```
3. `csd_uid` value (verify it looks correct and preserves leading zeros):
```javascript
print('csd_uid value', first.get('csd_uid'));
```
> Note: Earth Engine may not display property types directly; verify that `csd_uid` values look correct and preserve leading zeros.
4. Geometry display:
```javascript
Map.addLayer(ee.FeatureCollection('users/YOUR_USERNAME/canada_csd_2021_boundaries'), {}, 'CSD boundaries');
Map.centerObject(first, 7);
```
If all checks pass, you are ready to run the ERA5-Land GEE smoke test.
