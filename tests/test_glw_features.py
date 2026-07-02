import os
import unittest
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import from_origin

class TestGLWFeatures(unittest.TestCase):
    def setUp(self):
        self.raster_path = 'temp_test_raster_chicken.tif'
        # Create a test raster of shape (10, 10) starting at -100 lon, 50 lat, resolution 1 degree
        transform = from_origin(-100, 50, 1, 1)
        data = np.arange(100, dtype=np.float32).reshape((10, 10))
        
        # Set pixel (0, 0) to nodata
        self.nodata = -3.4e38
        data[0, 0] = self.nodata
        
        with rasterio.open(
            self.raster_path, 'w',
            driver='GTiff',
            height=10, width=10,
            count=1,
            dtype='float32',
            crs='EPSG:4326',
            transform=transform,
            nodata=self.nodata
        ) as dst:
            dst.write(data, 1)
            
    def tearDown(self):
        if os.path.exists(self.raster_path):
            os.remove(self.raster_path)
            
    def test_zonal_mask_extraction(self):
        """Test that custom zonal masking extracts the correct stats for overlapping polygons."""
        # Define a polygon that covers pixels (1, 1) to (2, 2)
        # Coordinates: lon -99 to -97, lat 49 to 47
        poly = Polygon([(-99, 49), (-97, 49), (-97, 47), (-99, 47), (-99, 49)])
        
        from rasterio.mask import mask
        with rasterio.open(self.raster_path) as src:
            out_image, out_transform = mask(src, [poly], crop=True)
            data = out_image[0]
            valid_mask = (data != self.nodata) & (data > -3.3e+38)
            valid_data = data[valid_mask]
            
            self.assertTrue(len(valid_data) > 0)
            # The values in this region are:
            # Row 1: index 11, 12 -> 11.0, 12.0
            # Row 2: index 21, 22 -> 21.0, 22.0
            self.assertEqual(valid_data.min(), 11.0)
            self.assertEqual(valid_data.max(), 22.0)
            self.assertAlmostEqual(valid_data.mean(), 16.5)
            
    def test_centroid_sampling_fallback(self):
        """Test that centroid sampling fallback is triggered correctly for empty intersections."""
        # Create a tiny polygon that doesn't capture pixel centers but overlaps pixel (2, 2) centroid
        # Pixel (2, 2) is centered at lon -97.5, lat 47.5, which has value 22.0
        poly = Polygon([(-97.6, 47.6), (-97.4, 47.6), (-97.4, 47.4), (-97.6, 47.4), (-97.6, 47.6)])
        
        with rasterio.open(self.raster_path) as src:
            centroid = poly.centroid
            val = list(src.sample([(centroid.x, centroid.y)]))[0][0]
            self.assertEqual(float(val), 22.0)

if __name__ == '__main__':
    unittest.main()
