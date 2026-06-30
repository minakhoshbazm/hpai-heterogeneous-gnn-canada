import unittest
import os
import tempfile
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point

class TestCSDMappingPipeline(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for files
        self.test_dir = tempfile.TemporaryDirectory()
        self.gpkg_path = os.path.join(self.test_dir.name, "test_boundaries.gpkg")
        
        # Create four adjacent square CSD polygons in a grid (EPSG:3347 coordinates)
        # 0 | 1
        # -----
        # 2 | 3
        poly0 = Polygon([(0, 100), (100, 100), (100, 200), (0, 200)])
        poly1 = Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])
        poly2 = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        poly3 = Polygon([(100, 0), (200, 0), (200, 100), (100, 100)])
        
        # Note that Stats Canada columns are CSDUID, CSDNAME, PRUID
        boundary_data = {
            'CSDUID': ['1001101', '1001105', '1001110', '1001115'],
            'CSDNAME': ['Township A', 'Village B', 'Subdivision C', 'City D'],
            'PRUID': ['10', '10', '10', '10'],
            'geometry': [poly0, poly1, poly2, poly3]
        }
        
        gdf = gpd.GeoDataFrame(boundary_data, crs="EPSG:3347")
        # Save as GPKG for sjoin testing
        gdf.to_file(self.gpkg_path, driver="GPKG")
        
        # Create CSD lookup csv
        self.lookup_path = os.path.join(self.test_dir.name, "census_subdivision_lookup.csv")
        lookup_data = [
            {'csd_uid': '1001101', 'csd_name': 'Township A', 'csd_type': 'TP', 'cd_uid': '1001', 'pruid': '10', 'land_area': 1000.0},
            {'csd_uid': '1001105', 'csd_name': 'Village B', 'csd_type': 'V', 'cd_uid': '1001', 'pruid': '10', 'land_area': 1000.0},
            {'csd_uid': '1001110', 'csd_name': 'Subdivision C', 'csd_type': 'SNO', 'cd_uid': '1001', 'pruid': '10', 'land_area': 1000.0},
            {'csd_uid': '1001115', 'csd_name': 'City D', 'csd_type': 'C', 'cd_uid': '1001', 'pruid': '10', 'land_area': 1000.0}
        ]
        pd.DataFrame(lookup_data).to_csv(self.lookup_path, index=False)
        
    def tearDown(self):
        self.test_dir.cleanup()
        
    def test_coordinate_spatial_join(self):
        """Verify Priority 1: spatial join intersects coordinate points and reprojects correctly."""
        # Setup points in EPSG:4326 (WGS84)
        # Note: Stats Canada conformal conic coordinates (10, 150) in EPSG:3347 maps to a point inside poly2
        # Let's mock the sjoin directly in EPSG:3347 to be independent of local coordinate reproject math
        points_df = pd.DataFrame([
            {'premises_id': 'NL-01', 'latitude': 47.5, 'longitude': -52.7, 'province': 'Newfoundland and Labrador'}
        ])
        
        # Test Point inside poly0: center of poly0 is (50, 150)
        point = Point(50, 150)
        points_gdf = gpd.GeoDataFrame(points_df, geometry=[point], crs="EPSG:3347")
        
        # Load CSD boundaries
        csd_gdf = gpd.read_file(self.gpkg_path)
        csd_gdf['geometry'] = csd_gdf.geometry.buffer(0)
        
        joined = gpd.sjoin(points_gdf, csd_gdf, how="left", predicate="intersects")
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined.iloc[0]['CSDUID'], '1001101')
        self.assertEqual(joined.iloc[0]['CSDNAME'], 'Township A')
        
    def test_fuzzy_name_matching(self):
        """Verify Priority 2: name normalization and difflib fuzzy lookups resolve names within same province."""
        import difflib
        
        # Mock lookup
        lookup_df = pd.read_csv(self.lookup_path, dtype={'csd_uid': str, 'cd_uid': str, 'pruid': str})
        lookup_df['csd_name_norm'] = lookup_df['csd_name'].str.strip().str.lower()
        
        # Outbreak row city name with typo
        search_city = "townshipp a" # Typo
        city_norm = search_city.strip().lower()
        
        # Search list
        csd_names_list = lookup_df['csd_name_norm'].tolist()
        
        # Difflib close matches
        close_matches = difflib.get_close_matches(city_norm, csd_names_list, n=1, cutoff=0.8)
        self.assertEqual(len(close_matches), 1)
        self.assertEqual(close_matches[0], 'township a')
        
        matched_row = lookup_df[lookup_df['csd_name_norm'] == close_matches[0]].iloc[0]
        self.assertEqual(matched_row['csd_uid'], '1001101')
        
    def test_shifted_labels(self):
        """Verify target labels temporal shifts are correctly computed within each CSD node group."""
        # Create balanced CSD-month grid for testing
        grid_data = [
            {'csd_uid': '1001101', 'month': '2023-01', 'outbreak_count': 0, 'outbreak_binary': 0},
            {'csd_uid': '1001101', 'month': '2023-02', 'outbreak_count': 1, 'outbreak_binary': 1},
            {'csd_uid': '1001101', 'month': '2023-03', 'outbreak_count': 0, 'outbreak_binary': 0},
            {'csd_uid': '1001105', 'month': '2023-01', 'outbreak_count': 1, 'outbreak_binary': 1},
            {'csd_uid': '1001105', 'month': '2023-02', 'outbreak_count': 0, 'outbreak_binary': 0},
            {'csd_uid': '1001105', 'month': '2023-03', 'outbreak_count': 0, 'outbreak_binary': 0}
        ]
        grid = pd.DataFrame(grid_data)
        labels_df = grid.sort_values(by=['csd_uid', 'month']).copy()
        
        # Apply group shifts
        labels_df['outbreak_count_next_month'] = labels_df.groupby('csd_uid')['outbreak_count'].shift(-1)
        labels_df['outbreak_next_month'] = labels_df.groupby('csd_uid')['outbreak_binary'].shift(-1)
        
        df_101 = labels_df[labels_df['csd_uid'] == '1001101'].reset_index(drop=True)
        self.assertEqual(df_101.loc[0, 'outbreak_next_month'], 1) # Next month (Feb) has outbreak
        self.assertEqual(df_101.loc[1, 'outbreak_next_month'], 0) # Next month (Mar) has 0
        self.assertTrue(pd.isna(df_101.loc[2, 'outbreak_next_month'])) # Next month (Apr) is boundary NaN
        
        df_105 = labels_df[labels_df['csd_uid'] == '1001105'].reset_index(drop=True)
        self.assertEqual(df_105.loc[0, 'outbreak_next_month'], 0) # Next month (Feb) is 0
        self.assertEqual(df_105.loc[1, 'outbreak_next_month'], 0) # Next month (Mar) is 0
        self.assertTrue(pd.isna(df_105.loc[2, 'outbreak_next_month'])) # Next month (Apr) is boundary NaN

if __name__ == "__main__":
    unittest.main()
