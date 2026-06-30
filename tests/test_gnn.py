import unittest
import os
import tempfile
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

try:
    import torch
    import torch_geometric
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

@unittest.skipUnless(HAS_TORCH, "PyTorch and PyTorch Geometric are required for GNN tests.")
class TestSpatialGNN(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        self.gpkg_path = os.path.join(self.test_dir.name, "test_boundaries.gpkg")
        
        # Create four adjacent square polygons in a grid (EPSG:3347 unit is meters)
        # 0 | 1
        # -----
        # 2 | 3
        poly0 = Polygon([(0, 100), (100, 100), (100, 200), (0, 200)])
        poly1 = Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])
        poly2 = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        poly3 = Polygon([(100, 0), (200, 0), (200, 100), (100, 100)])
        
        data = {
            'cd_uid': ['0001', '0002', '0003', '0004'],
            'cd_name': ['Div 1', 'Div 2', 'Div 3', 'Div 4'],
            'province_code': ['10', '10', '10', '10'],
            'province_name_en': ['NL', 'NL', 'NL', 'NL'],
            'province_name_bilingual': ['NL', 'NL', 'NL', 'NL'],
            'province_id': ['10', '10', '10', '10'],
            'province_name': ['NL', 'NL', 'NL', 'NL'],
            'land_area': [10000.0, 10000.0, 10000.0, 10000.0],
            'geometry': [poly0, poly1, poly2, poly3]
        }
        
        gdf = gpd.GeoDataFrame(data, crs="EPSG:3347")
        gdf.to_file(self.gpkg_path, driver="GPKG")
        
        # Create a mock feature CSV file for testing data loader
        self.feature_path = os.path.join(self.test_dir.name, "test_features.csv")
        months = ["2022-01", "2022-02"]
        records = []
        for month in months:
            for cd in ['0001', '0002', '0003', '0004']:
                records.append({
                    'month': month,
                    'cd_uid': cd,
                    'poultry_density_proxy': np.random.uniform(0.1, 1.0),
                    'road_density_proxy': np.random.uniform(0.1, 1.0),
                    'ndvi_anomaly': np.random.uniform(0.1, 1.0),
                    'ndwi_anomaly': np.random.uniform(0.1, 1.0),
                    'temp_anomaly': np.random.uniform(0.1, 1.0),
                    'precip_anomaly': np.random.uniform(0.1, 1.0),
                    'water_fraction': np.random.uniform(0.1, 1.0),
                    'mallard_abundance_proxy': np.random.uniform(0.1, 1.0),
                    'canada_goose_abundance_proxy': np.random.uniform(0.1, 1.0),
                    'outbreak_lag_1': 0,
                    'outbreak_lag_2': 0,
                    'outbreak_next_month': int(np.random.rand() > 0.5)
                })
        pd.DataFrame(records).to_csv(self.feature_path, index=False)
        
    def tearDown(self):
        self.test_dir.cleanup()
        
    def test_spatial_graph_construction(self):
        """Verify Queen adjacency graph creation and dimensions from spatial_graph.py."""
        from hpai_gnn.graph.spatial_graph import build_spatial_graph
        
        edge_index, edge_attr, node_to_idx = build_spatial_graph(
            self.gpkg_path, sorted_cd_uids=None, inverse_distance_weights=True
        )
        
        self.assertEqual(len(node_to_idx), 4)
        self.assertEqual(node_to_idx['0001'], 0)
        self.assertEqual(node_to_idx['0004'], 3)
        
        self.assertEqual(edge_index.dim(), 2)
        self.assertEqual(edge_index.size(0), 2)
        self.assertEqual(edge_attr.size(1), 1)
        self.assertEqual(edge_index.size(1), edge_attr.size(0))
        
        # Verify reciprocal symmetry (undirected graph representation)
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        for u, v in edges:
            self.assertIn((v, u), edges)
            
    def test_gnn_model_forward_pass(self):
        """Verify GCN, GraphSAGE, and GAT model construction and shape validation."""
        from hpai_gnn.models.gnn_baselines import SpatialGNN
        
        in_channels = 11
        hidden_channels = 16
        x = torch.randn(4, in_channels)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_weight = torch.rand(4, 1)
        
        # Test GCN
        gcn = SpatialGNN(in_channels, hidden_channels, out_channels=1, num_layers=2, model_type='GCN')
        logits_gcn = gcn(x, edge_index, edge_weight)
        self.assertEqual(logits_gcn.shape, (4, 1))
        
        # Test GraphSAGE
        sage = SpatialGNN(in_channels, hidden_channels, out_channels=1, num_layers=2, model_type='SAGE')
        logits_sage = sage(x, edge_index)
        self.assertEqual(logits_sage.shape, (4, 1))
        
        # Test GAT
        gat = SpatialGNN(in_channels, hidden_channels, out_channels=1, num_layers=2, model_type='GAT', heads=2)
        logits_gat = gat(x, edge_index, edge_weight)
        self.assertEqual(logits_gat.shape, (4, 1))
        
    def test_gnn_data_preparation(self):
        """Verify GNN dataset preparation and monthly spatiotemporal snapshot mapping."""
        from hpai_gnn.training.train_gnn import prepare_gnn_data
        
        feature_cols = [
            'poultry_density_proxy', 'road_density_proxy',
            'ndvi_anomaly', 'ndwi_anomaly', 'temp_anomaly', 'precip_anomaly', 'water_fraction',
            'mallard_abundance_proxy', 'canada_goose_abundance_proxy',
            'outbreak_lag_1', 'outbreak_lag_2'
        ]
        
        train_snapshots, val_snapshots, test_snapshots, scaler, _, _ = prepare_gnn_data(
            feature_path=self.feature_path,
            gpkg_path=self.gpkg_path,
            train_years=[2022],
            val_years=[],
            test_years=[],
            feature_cols=feature_cols,
            target_col='outbreak_next_month'
        )
        
        # 2 months in 2022
        self.assertEqual(len(train_snapshots), 2)
        self.assertEqual(len(val_snapshots), 0)
        self.assertEqual(len(test_snapshots), 0)
        
        # Validate data object properties
        snapshot = train_snapshots[0]
        self.assertTrue(hasattr(snapshot, 'x'))
        self.assertTrue(hasattr(snapshot, 'edge_index'))
        self.assertTrue(hasattr(snapshot, 'y'))
        self.assertEqual(snapshot.x.shape, (4, len(feature_cols)))
        self.assertEqual(snapshot.y.shape, (4, 1))

if __name__ == "__main__":
    unittest.main()
