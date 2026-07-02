import unittest
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
import pandas as pd

class TestCSDGraph(unittest.TestCase):
    def test_queen_vs_rook_contiguity(self):
        # Create two adjacent squares sharing a boundary segment from (10, 0) to (10, 10) (Rook contiguity)
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10), (10, 0)])
        
        # Create a third square touching poly1 only at the corner point (10, 10) (Queen contiguity)
        poly3 = Polygon([(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)])
        
        # Perform boundary length checks
        boundary1 = poly1.boundary
        boundary2 = poly2.boundary
        boundary3 = poly3.boundary
        
        # Rook boundary length check
        intersection_rook = boundary1.intersection(boundary2)
        shared_len_rook = float(intersection_rook.length)
        self.assertGreater(shared_len_rook, 0.0)
        self.assertAlmostEqual(shared_len_rook, 10.0)
        
        # Queen boundary length check
        intersection_queen = boundary1.intersection(boundary3)
        shared_len_queen = float(intersection_queen.length)
        self.assertEqual(shared_len_queen, 0.0)
        self.assertTrue(poly1.intersects(poly3)) # Touches at point
        
    def test_undirected_ordering(self):
        # Verify that sorting source and target ensures source < target
        uid_a = "3520005"
        uid_b = "2404015"
        
        source, target = sorted([uid_a, uid_b])
        self.assertEqual(source, "2404015")
        self.assertEqual(target, "3520005")
        self.assertTrue(source < target)

if __name__ == '__main__':
    unittest.main()
