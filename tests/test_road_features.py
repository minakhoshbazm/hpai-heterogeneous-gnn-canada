import unittest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import pandas as pd

from src.hpai_gnn.data.extract_road_features import compute_centroid_distances

class TestRoadFeatures(unittest.TestCase):
    def test_compute_centroid_distances(self):
        # Create road geometry
        road_geoms = [
            LineString([(0, 0), (10, 0)]),  # Horizontal line along y=0
            LineString([(0, 10), (10, 10)]) # Horizontal line along y=10
        ]
        road_gdf = gpd.GeoDataFrame(geometry=road_geoms, crs="EPSG:3347")
        
        # Centroids to test
        centroids = [
            Point(5, 2),   # Distance to line 1 is 2 meters, to line 2 is 8 meters. Nearest is line 1.
            Point(5, 9),   # Distance to line 1 is 9 meters, to line 2 is 1 meter. Nearest is line 2.
            Point(5, 5),   # Distance is 5 meters.
            None,          # Empty geometry
            Point()        # Empty Point
        ]
        
        distances = compute_centroid_distances(centroids, road_gdf)
        
        # Check lengths
        self.assertEqual(len(distances), 5)
        
        # Check actual values (converting meters to km: 2m -> 0.002km)
        self.assertAlmostEqual(distances[0], 0.002)
        self.assertAlmostEqual(distances[1], 0.001)
        self.assertAlmostEqual(distances[2], 0.005)
        self.assertTrue(np.isnan(distances[3]))
        self.assertTrue(np.isnan(distances[4]))

    def test_empty_roads(self):
        road_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:3347")
        centroids = [Point(1, 1)]
        distances = compute_centroid_distances(centroids, road_gdf)
        self.assertEqual(distances, [np.nan])

    def test_overlay_splitting(self):
        # Verify line splitting via overlay works as expected
        poly_geom = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        csd_gdf = gpd.GeoDataFrame({'CSDUID': ['12345'], 'geometry': [poly_geom]}, crs="EPSG:3347")
        
        # Line that goes from (-5, 5) to (15, 5) - total length 20. But inside poly it is from (0,5) to (10,5) - length 10.
        road_geom = LineString([(-5, 5), (15, 5)])
        road_gdf = gpd.GeoDataFrame({'ROADCLASS': ['Local / Street'], 'geometry': [road_geom]}, crs="EPSG:3347")
        
        overlay_gdf = gpd.overlay(road_gdf, csd_gdf[['CSDUID', 'geometry']], how='intersection')
        self.assertEqual(len(overlay_gdf), 1)
        self.assertEqual(overlay_gdf.iloc[0]['CSDUID'], '12345')
        
        # Check length of the intersected segment (should be 10 meters)
        self.assertAlmostEqual(overlay_gdf.geometry.length.iloc[0], 10.0)

if __name__ == '__main__':
    unittest.main()
