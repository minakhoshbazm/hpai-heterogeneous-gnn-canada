import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from scripts.analyze_era5_fallback_smoke_test import analyze_fallback_smoke_test

class TestAnalyzeEra5FallbackSmokeTest(unittest.TestCase):
    def setUp(self):
        # Create a valid tiny synthetic dataset containing 1 CSD, 1 month, 5 methods (5 rows total)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(self.temp_dir.name, "input.csv")
        self.output_path = os.path.join(self.temp_dir.name, "output.csv")
        
        # Base row structure
        self.methods = ['polygon_mean', 'centroid_sample', 'buffer_25km', 'buffer_50km', 'buffer_100km']
        rows = []
        for idx, m in enumerate(self.methods):
            # Centroid sample and buffer 25km are missing, buffer 50km and 100km are populated, polygon mean is missing
            is_missing = m in ['polygon_mean', 'centroid_sample', 'buffer_25km']
            rows.append({
                'system:index': f"idx_{idx}",
                '.geo': '{"type":"Point"}',
                'geo': '{"type":"Point"}',
                'csd_uid': '1001120',
                'csd_name': "St. Shott's",
                'province_u': 10,
                'year': 2022,
                'month': 1,
                'method_name': m,
                'climate_tmean_c': np.nan if is_missing else 5.0,
                'climate_tmin_c': np.nan if is_missing else 0.0,
                'climate_tmax_c': np.nan if is_missing else 10.0,
                'climate_precip_mm': np.nan if is_missing else 50.0,
                'method_missing_flag': 1 if is_missing else 0
            })
            
        self.base_df = pd.DataFrame(rows)
        self.base_df.to_csv(self.input_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_analysis(self):
        # Base df has 5 rows, expected_rows = 5
        analyze_fallback_smoke_test(
            self.input_path, self.output_path, expected_rows=5,
            expected_methods=self.methods, expected_months={1}
        )
        self.assertTrue(os.path.exists(self.output_path))
        
        # Verify first successful method is buffer_50km
        df = pd.read_csv(self.output_path, dtype={'csd_uid': str})
        self.assertEqual(len(df), 1)
        self.assertEqual(df['first_successful_method'].iloc[0], 'buffer_50km')
        self.assertEqual(df['climate_tmean_c'].iloc[0], 5.0)

    def test_invalid_row_count(self):
        df = self.base_df.iloc[:-1] # 4 rows
        df.to_csv(self.input_path, index=False)
        with self.assertRaises(ValueError):
            analyze_fallback_smoke_test(
                self.input_path, self.output_path, expected_rows=5,
                expected_methods=df['method_name'].tolist(), expected_months={1}
            )

    def test_invalid_method_name(self):
        df = self.base_df.copy()
        df.loc[0, 'method_name'] = 'invalid_method'
        df.to_csv(self.input_path, index=False)
        with self.assertRaises(ValueError):
            analyze_fallback_smoke_test(
                self.input_path, self.output_path, expected_rows=5,
                expected_methods=self.methods, expected_months={1}
            )

    def test_duplicate_method_rows(self):
        df = self.base_df.copy()
        # Create duplicate for polygon_mean
        dup_row = df.iloc[0].copy()
        df = pd.concat([df, pd.DataFrame([dup_row])], ignore_index=True)
        # Drop another row to keep count as 5 (drops centroid_sample)
        df = df.drop(index=1)
        df.to_csv(self.input_path, index=False)
        # Expected methods list doesn't include centroid_sample anymore because we dropped it
        remaining_methods = ['polygon_mean', 'buffer_25km', 'buffer_50km', 'buffer_100km']
        with self.assertRaises(ValueError):
            analyze_fallback_smoke_test(
                self.input_path, self.output_path, expected_rows=5,
                expected_methods=remaining_methods, expected_months={1}
            )

    def test_invalid_missing_flag(self):
        df = self.base_df.copy()
        df.loc[0, 'method_missing_flag'] = 2
        df.to_csv(self.input_path, index=False)
        with self.assertRaises(ValueError):
            analyze_fallback_smoke_test(
                self.input_path, self.output_path, expected_rows=5,
                expected_methods=self.methods, expected_months={1}
            )

    def test_mismatched_missing_flag_logic(self):
        df = self.base_df.copy()
        # Set missing flag to 0 for a missing row
        df.loc[0, 'method_missing_flag'] = 0
        df.to_csv(self.input_path, index=False)
        with self.assertRaises(ValueError):
            analyze_fallback_smoke_test(
                self.input_path, self.output_path, expected_rows=5,
                expected_methods=self.methods, expected_months={1}
            )

    def test_negative_precipitation(self):
        df = self.base_df.copy()
        # buffer_50km (idx 3) is populated, set it to -5
        df.loc[3, 'climate_precip_mm'] = -5.0
        df.to_csv(self.input_path, index=False)
        with self.assertRaises(ValueError):
            analyze_fallback_smoke_test(
                self.input_path, self.output_path, expected_rows=5,
                expected_methods=self.methods, expected_months={1}
            )

    def test_invalid_temperature_ordering(self):
        df = self.base_df.copy()
        # buffer_50km (idx 3) is populated, set tmin > tmean
        df.loc[3, 'climate_tmin_c'] = 8.0
        df.loc[3, 'climate_tmean_c'] = 5.0
        df.to_csv(self.input_path, index=False)
        with self.assertRaises(ValueError):
            analyze_fallback_smoke_test(
                self.input_path, self.output_path, expected_rows=5,
                expected_methods=self.methods, expected_months={1}
            )

if __name__ == '__main__':
    unittest.main()
