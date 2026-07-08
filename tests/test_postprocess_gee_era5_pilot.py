import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from scripts.postprocess_gee_era5_pilot import postprocess_pilot_csv

class TestPostprocessGeeEra5Pilot(unittest.TestCase):
    def setUp(self):
        # Create a valid base dataset of 10 CSDs across 12 months (120 rows total)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(self.temp_dir.name, "input.csv")
        self.output_path = os.path.join(self.temp_dir.name, "output.csv")
        
        # 10 mock CSDs
        self.csds = [f"{i:07d}" for i in range(10)]
        self.csd_names = [f"CSD_{csd}" for csd in self.csds]
        
        rows = []
        for csd, name in zip(self.csds, self.csd_names):
            for m in range(1, 13):
                # Les Iles-de-la-Madeleine is mock CSD 4 (similar to CSD 2401023), set all fields to NaN, missing flag to 1
                is_missing = (csd == "0000004")
                rows.append({
                    'system:index': f"idx_{csd}_{m}",
                    '.geo': '{"type":"MultiPoint"}',
                    'geo': '{"type":"MultiPoint"}',
                    'csd_uid': csd,
                    'csd_name': name,
                    'province_u': 10,
                    'province_n': 'NL',
                    'year': 2022,
                    'month': m,
                    'climate_tmean_c': np.nan if is_missing else 5.0,
                    'climate_tmin_c': np.nan if is_missing else 0.0,
                    'climate_tmax_c': np.nan if is_missing else 10.0,
                    'climate_precip_mm': np.nan if is_missing else 50.0,
                    'climate_missing_flag': 1 if is_missing else 0
                })
        self.base_df = pd.DataFrame(rows)
        self.base_df.to_csv(self.input_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_postprocessing(self):
        # The base DataFrame should pass validation successfully
        postprocess_pilot_csv(self.input_path, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))
        
        # Verify columns and data types
        df = pd.read_csv(self.output_path, dtype={'csd_uid': str})
        self.assertEqual(len(df), 120)
        self.assertNotIn('system:index', df.columns)
        self.assertNotIn('.geo', df.columns)
        self.assertNotIn('geo', df.columns)
        
        # Verify csd_uid is a string and preserves leading zeros
        self.assertTrue(pd.api.types.is_string_dtype(df['csd_uid']))
        self.assertIsInstance(df['csd_uid'].iloc[0], str)
        self.assertEqual(df['csd_uid'].iloc[0], "0000000")
        
        # Verify Les Iles-de-la-Madeleine (mock CSD 4) has flag 1
        mock_4 = df[df['csd_uid'] == "0000004"]
        self.assertTrue((mock_4['climate_missing_flag'] == 1).all())
        self.assertTrue(mock_4['climate_tmean_c'].isna().all())

    def test_invalid_row_count(self):
        # Modify row count
        df = self.base_df.iloc[:-1] # 119 rows
        df.to_csv(self.input_path, index=False)
        with self.assertRaises(ValueError):
            postprocess_pilot_csv(self.input_path, self.output_path)

    def test_invalid_csd_count(self):
        # Modify unique CSD count by setting one CSD to another
        df = self.base_df.copy()
        df.loc[df['csd_uid'] == '0000009', 'csd_uid'] = '0000008'
        df.to_csv(self.input_path, index=False)
        with self.assertRaises(ValueError):
            postprocess_pilot_csv(self.input_path, self.output_path)

    def test_invalid_months(self):
        # Modify months (set month 12 to 11)
        df = self.base_df.copy()
        df.loc[df['month'] == 12, 'month'] = 11
        df.to_csv(self.input_path, index=False)
        with self.assertRaises(ValueError):
            postprocess_pilot_csv(self.input_path, self.output_path)

    def test_duplicate_csd_months(self):
        # Introduce a duplicate row
        df = self.base_df.copy()
        dup_row = df.iloc[0].copy()
        df = pd.concat([df, pd.DataFrame([dup_row])], ignore_index=True)
        # Ensure it has 120 rows by dropping another row
        df = df.drop(index=1)
        df.to_csv(self.input_path, index=False)
        with self.assertRaises(ValueError):
            postprocess_pilot_csv(self.input_path, self.output_path)

    def test_invalid_missing_flag(self):
        # Set missing flag to 2
        df = self.base_df.copy()
        df.loc[0, 'climate_missing_flag'] = 2
        df.to_csv(self.input_path, index=False)
        with self.assertRaises(ValueError):
            postprocess_pilot_csv(self.input_path, self.output_path)

    def test_mismatched_missing_flag_logic(self):
        # Set missing flag to 0 for a missing row
        df = self.base_df.copy()
        df.loc[df['csd_uid'] == '0000004', 'climate_missing_flag'] = 0
        df.to_csv(self.input_path, index=False)
        with self.assertRaises(ValueError):
            postprocess_pilot_csv(self.input_path, self.output_path)

    def test_negative_precipitation(self):
        # Set precip to -5
        df = self.base_df.copy()
        df.loc[0, 'climate_precip_mm'] = -5.0
        df.to_csv(self.input_path, index=False)
        with self.assertRaises(ValueError):
            postprocess_pilot_csv(self.input_path, self.output_path)

    def test_invalid_temperature_ordering(self):
        # Set tmin > tmean
        df = self.base_df.copy()
        df.loc[0, 'climate_tmin_c'] = 8.0
        df.loc[0, 'climate_tmean_c'] = 5.0
        df.to_csv(self.input_path, index=False)
        with self.assertRaises(ValueError):
            postprocess_pilot_csv(self.input_path, self.output_path)

if __name__ == '__main__':
    unittest.main()
