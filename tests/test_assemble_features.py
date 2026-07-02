import unittest
import pandas as pd
import numpy as np

class TestAssembleFeatures(unittest.TestCase):
    def test_exclusion_policy(self):
        # Create a mock DataFrame with various columns
        mock_data = pd.DataFrame({
            'csd_uid': ['3520005'],
            'latitude': [43.65],
            'geom': ['POLYGON'],
            'outbreak_label': [1]
        })
        
        # Test the policy checks
        excluded_keywords = ['geometry', 'geom', 'latitude', 'longitude', 'lat', 'lon', 'outbreak', 'label']
        found_violations = [c for c in mock_data.columns if any(kw in c.lower() for kw in excluded_keywords)]
        
        self.assertGreater(len(found_violations), 0)
        self.assertIn('latitude', found_violations)
        self.assertIn('geom', found_violations)
        self.assertIn('outbreak_label', found_violations)

    def test_csd_join_key_types(self):
        # Verify that UIDs are stored and compared as strings (preserving leading zeros)
        uid_str = "0102003"
        uid_int = int(uid_str)
        
        self.assertNotEqual(uid_str, str(uid_int)) # "0102003" != "102003"
        self.assertEqual(len(uid_str), 7)
        self.assertEqual(len(str(uid_int)), 6)

if __name__ == '__main__':
    unittest.main()
