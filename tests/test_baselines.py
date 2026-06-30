import os
import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from hpai_gnn.features.mock_data import generate_mock_features
from hpai_gnn.features.features import (
    load_feature_dataset,
    train_val_test_split_temporal,
    scale_features,
    prepare_tabular_data
)
from hpai_gnn.models.baselines import get_baseline_model
from hpai_gnn.evaluation.metrics import calculate_metrics

class TestBaselinesPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a tiny mock labels file for fast testing
        cls.test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp_test_dir"))
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.labels_path = os.path.join(cls.test_dir, "test_labels.csv")
        cls.features_path = os.path.join(cls.test_dir, "test_features.csv")
        
        # 3 CDs, 12 months (2022-01 to 2022-12)
        records = []
        for cd in ['1001', '2002', '3003']:
            for month in [f"2022-{str(m).zfill(2)}" for m in range(1, 13)]:
                records.append({
                    'cd_uid': cd,
                    'cd_name': f"CD_{cd}",
                    'province_name': "TestProvince",
                    'month': month,
                    'outbreak_count_current_month': 1 if np.random.rand() > 0.8 else 0,
                    'outbreak_binary_current_month': 1 if np.random.rand() > 0.8 else 0,
                    'outbreak_count_next_month': 0.0,
                    'outbreak_next_month': 1.0 if np.random.rand() > 0.8 else 0.0
                })
        
        pd.DataFrame(records).to_csv(cls.labels_path, index=False)
        
    @classmethod
    def tearDownClass(cls):
        # Clean up files and directory
        if os.path.exists(cls.labels_path):
            os.remove(cls.labels_path)
        if os.path.exists(cls.features_path):
            os.remove(cls.features_path)
        if os.path.exists(cls.test_dir):
            try:
                os.rmdir(cls.test_dir)
            except OSError:
                pass

    def test_mock_generation(self):
        """Verify mock feature generation writes correctly shaped file."""
        generate_mock_features(self.labels_path, self.features_path)
        self.assertTrue(os.path.exists(self.features_path))
        
        df = pd.read_csv(self.features_path)
        self.assertEqual(len(df), 36) # 3 CDs * 12 months
        
        expected_cols = [
            'cd_uid', 'cd_name', 'province_name', 'month',
            'ndvi_anomaly', 'ndwi_anomaly', 'temp_anomaly', 'precip_anomaly', 'water_fraction',
            'mallard_abundance_proxy', 'canada_goose_abundance_proxy',
            'outbreak_lag_1', 'outbreak_lag_2', 'outbreak_next_month'
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_temporal_splitting_completeness(self):
        """Verify train/val/test splits don't share overlap and contain expected years."""
        # Synthesize a dataset across years 2021-2023
        data = [
            {'month': '2021-05', 'val': 1.0},
            {'month': '2021-12', 'val': 2.0},
            {'month': '2022-06', 'val': 3.0},
            {'month': '2023-01', 'val': 4.0}
        ]
        df = pd.DataFrame(data)
        train_df, val_df, test_df = train_val_test_split_temporal(
            df, train_years=[2021], val_years=[2022], test_years=[2023]
        )
        
        self.assertEqual(len(train_df), 2)
        self.assertEqual(len(val_df), 1)
        self.assertEqual(len(test_df), 1)
        
        # Verify years
        self.assertTrue((train_df['month'].str.startswith('2021')).all())
        self.assertTrue((val_df['month'].str.startswith('2022')).all())
        self.assertTrue((test_df['month'].str.startswith('2023')).all())

    def test_feature_scaling_isolation(self):
        """Verify scaling fit parameters are isolated to training sets."""
        train_data = pd.DataFrame({'feat': [10.0, 20.0, 30.0]})
        val_data = pd.DataFrame({'feat': [5.0]})
        test_data = pd.DataFrame({'feat': [40.0]})
        
        train_scaled, val_scaled, test_scaled, scaler = scale_features(
            train_data, val_data, test_data, ['feat'], method='standard'
        )
        
        # Scaler mean should be trained mean: (10 + 20 + 30) / 3 = 20.0
        self.assertEqual(scaler.mean_[0], 20.0)
        self.assertAlmostEqual(train_scaled.loc[1, 'feat'], 0.0) # 20.0 scaled should be 0.0 (mean)

    def test_baseline_instantiation(self):
        """Verify factory returns standard model interfaces."""
        lr = get_baseline_model('logistic_regression')
        rf = get_baseline_model('random_forest')
        
        self.assertTrue(hasattr(lr, 'fit'))
        self.assertTrue(hasattr(lr, 'predict'))
        self.assertTrue(hasattr(rf, 'fit'))
        self.assertTrue(hasattr(rf, 'predict'))

    def test_metrics_edge_cases(self):
        """Verify metrics don't break with single-class datasets."""
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        y_prob = [0.1, 0.2, 0.15, 0.05]
        
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        self.assertEqual(metrics['roc_auc'], 0.5)
        self.assertEqual(metrics['sensitivity'], 0.0)
        self.assertEqual(metrics['specificity'], 1.0)

if __name__ == "__main__":
    unittest.main()
