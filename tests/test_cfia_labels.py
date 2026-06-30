import unittest
import pandas as pd
import numpy as np

class TestCFIALabelPipeline(unittest.TestCase):
    def test_import(self):
        """Verify that the hpai_gnn modules are importable."""
        try:
            import hpai_gnn
            import hpai_gnn.data
            import hpai_gnn.data.clean_cfia_outbreaks as cleaner
            self.assertTrue(hasattr(cleaner, 'clean_and_map_outbreaks'))
        except ImportError as e:
            self.fail(f"Failed to import cleaner module: {e}")

    def test_date_prioritization(self):
        """Verify that date selection logic correctly prioritizes date fields."""
        # Create synthetic rows representing different date presence patterns
        synthetic_data = [
            # Case 1: NCFAD date is available (primary)
            {
                'ncfad_confirmation_date': '2023-05-15',
                'cahsn_confirmation_date': '2023-05-12',
                'cfia_notification_date': '2023-05-10',
                'date_first_clinical_signs': '2023-05-08'
            },
            # Case 2: NCFAD missing, CAHSN available
            {
                'ncfad_confirmation_date': None,
                'cahsn_confirmation_date': '2023-06-12',
                'cfia_notification_date': '2023-06-10',
                'date_first_clinical_signs': '2023-06-08'
            },
            # Case 3: NCFAD & CAHSN missing, Notification available
            {
                'ncfad_confirmation_date': None,
                'cahsn_confirmation_date': None,
                'cfia_notification_date': '2023-07-10',
                'date_first_clinical_signs': '2023-07-08'
            },
            # Case 4: Only Clinical signs available
            {
                'ncfad_confirmation_date': None,
                'cahsn_confirmation_date': None,
                'cfia_notification_date': None,
                'date_first_clinical_signs': '2023-08-08'
            }
        ]
        
        df = pd.DataFrame(synthetic_data)
        for col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
        def select_event_date(row):
            if not pd.isna(row['ncfad_confirmation_date']):
                return row['ncfad_confirmation_date']
            if not pd.isna(row['cahsn_confirmation_date']):
                return row['cahsn_confirmation_date']
            if not pd.isna(row['cfia_notification_date']):
                return row['cfia_notification_date']
            if not pd.isna(row['date_first_clinical_signs']):
                return row['date_first_clinical_signs']
            return pd.NaT

        df['event_date'] = df.apply(select_event_date, axis=1)
        
        # Verify results
        self.assertEqual(df.loc[0, 'event_date'], pd.Timestamp('2023-05-15')) # Case 1
        self.assertEqual(df.loc[1, 'event_date'], pd.Timestamp('2023-06-12')) # Case 2
        self.assertEqual(df.loc[2, 'event_date'], pd.Timestamp('2023-07-10')) # Case 3
        self.assertEqual(df.loc[3, 'event_date'], pd.Timestamp('2023-08-08')) # Case 4

    def test_label_shifting_without_leakage(self):
        """Verify that shifting shifts labels to next month within each CD group without spatial leakage."""
        # Create a synthetic dataset of two CDs over 3 months
        # cd_uid 1001: Month 1 (0), Month 2 (1), Month 3 (0)
        # cd_uid 3506: Month 1 (1), Month 2 (0), Month 3 (0)
        synthetic_records = [
            {'cd_uid': '1001', 'month': '2023-01', 'outbreak_count': 0, 'outbreak_binary': 0},
            {'cd_uid': '1001', 'month': '2023-02', 'outbreak_count': 1, 'outbreak_binary': 1},
            {'cd_uid': '1001', 'month': '2023-03', 'outbreak_count': 0, 'outbreak_binary': 0},
            {'cd_uid': '3506', 'month': '2023-01', 'outbreak_count': 1, 'outbreak_binary': 1},
            {'cd_uid': '3506', 'month': '2023-02', 'outbreak_count': 0, 'outbreak_binary': 0},
            {'cd_uid': '3506', 'month': '2023-03', 'outbreak_count': 0, 'outbreak_binary': 0},
        ]
        
        df = pd.DataFrame(synthetic_records)
        df = df.sort_values(by=['cd_uid', 'month'])
        
        # Compute shift
        df['outbreak_count_next_month'] = df.groupby('cd_uid')['outbreak_count'].shift(-1)
        df['outbreak_next_month'] = df.groupby('cd_uid')['outbreak_binary'].shift(-1)
        
        # Verify shifts for cd_uid 1001
        df_1001 = df[df['cd_uid'] == '1001'].reset_index(drop=True)
        self.assertEqual(df_1001.loc[0, 'outbreak_next_month'], 1) # Next month (Feb) is 1
        self.assertEqual(df_1001.loc[1, 'outbreak_next_month'], 0) # Next month (Mar) is 0
        self.assertTrue(pd.isna(df_1001.loc[2, 'outbreak_next_month'])) # Next month (Apr) is NaN (future)
        
        # Verify shifts for cd_uid 3506 (should not contain leaked data from 1001)
        df_3506 = df[df['cd_uid'] == '3506'].reset_index(drop=True)
        self.assertEqual(df_3506.loc[0, 'outbreak_next_month'], 0) # Next month (Feb) is 0
        self.assertEqual(df_3506.loc[1, 'outbreak_next_month'], 0) # Next month (Mar) is 0
        self.assertTrue(pd.isna(df_3506.loc[2, 'outbreak_next_month'])) # Next month (Apr) is NaN (future)

if __name__ == "__main__":
    unittest.main()
