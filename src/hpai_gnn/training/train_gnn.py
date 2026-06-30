"""
Training and evaluation pipeline for Stage 2 Spatial GNN models.
"""

import os
import yaml
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

from hpai_gnn.graph.spatial_graph import build_spatial_graph
from hpai_gnn.models.gnn_baselines import SpatialGNN
from hpai_gnn.evaluation.metrics import calculate_metrics, print_metrics_summary, save_metrics_report

def prepare_gnn_data(feature_path, gpkg_path, train_years, val_years, test_years, 
                     feature_cols, target_col='outbreak_next_month', use_weights=False):
    """
    Loads spatial boundary graph, prepares, splits, and scales the spatiotemporal features,
    and constructs monthly PyTorch Geometric Data snapshots.
    """
    # 1. Build spatial graph (node indices alphabetical by cd_uid)
    print(f"Constructing spatial graph using Queen contiguity from: {gpkg_path}")
    edge_index, edge_attr, node_to_idx = build_spatial_graph(
        gpkg_path, sorted_cd_uids=None, inverse_distance_weights=use_weights
    )
    num_nodes = len(node_to_idx)
    print(f"Spatial graph built with {num_nodes} nodes and {edge_index.size(1)} edges.")
    
    # 2. Load feature data and sort by month then by cd_uid to align node order
    df = pd.read_csv(feature_path)
    df['cd_uid'] = df['cd_uid'].astype(str)
    df = df.sort_values(by=['month', 'cd_uid']).reset_index(drop=True)
    
    # 3. Temporal split
    df['year'] = df['month'].apply(lambda x: int(x.split('-')[0]))
    train_df = df[df['year'].isin(train_years)].copy()
    val_df = df[df['year'].isin(val_years)].copy()
    test_df = df[df['year'].isin(test_years)].copy()
    
    # Clean up temp columns
    train_df = train_df.drop(columns=['year'])
    val_df = val_df.drop(columns=['year'])
    test_df = test_df.drop(columns=['year'])
    
    # 4. Handle target NaNs
    train_df = train_df.dropna(subset=[target_col])
    val_df = val_df.dropna(subset=[target_col]) if len(val_df) > 0 else val_df
    test_df = test_df.dropna(subset=[target_col]) if len(test_df) > 0 else test_df
    
    # 5. Scale features fit ONLY on train split
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    if len(val_df) > 0:
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    if len(test_df) > 0:
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        
    # Helper to construct Data objects
    def build_snapshots(split_df):
        if len(split_df) == 0:
            return []
            
        snapshots = []
        # Group by month
        grouped = split_df.groupby('month')
        for month, group in grouped:
            # Sort by cd_uid to verify ordering consistency
            group_sorted = group.sort_values(by='cd_uid').reset_index(drop=True)
            if len(group_sorted) != num_nodes:
                raise ValueError(f"Month {month} has {len(group_sorted)} records; expected exactly {num_nodes}.")
                
            x_tensor = torch.tensor(group_sorted[feature_cols].values, dtype=torch.float32)
            y_tensor = torch.tensor(group_sorted[target_col].values, dtype=torch.float32).unsqueeze(1)
            
            data_obj = Data(
                x=x_tensor, 
                edge_index=edge_index, 
                edge_weight=edge_attr, 
                y=y_tensor
            )
            data_obj.month = month
            snapshots.append(data_obj)
            
        return snapshots
        
    train_snapshots = build_snapshots(train_df)
    val_snapshots = build_snapshots(val_df)
    test_snapshots = build_snapshots(test_df)
    
    return train_snapshots, val_snapshots, test_snapshots, scaler, edge_index, edge_attr

def train_one_epoch(model, optimizer, criterion, snapshots, device):
    model.train()
    total_loss = 0.0
    for data in snapshots:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_weight = data.edge_weight.to(device) if data.edge_weight is not None else None
        y = data.y.to(device)
        
        optimizer.zero_grad()
        logits = model(x, edge_index, edge_weight)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(snapshots)

def evaluate_model(model, snapshots, device):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for data in snapshots:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_weight = data.edge_weight.to(device) if data.edge_weight is not None else None
            
            logits = model(x, edge_index, edge_weight)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets = data.y.numpy().flatten()
            
            all_probs.extend(probs)
            all_targets.extend(targets)
            
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)
    
    return calculate_metrics(all_targets, all_preds, all_probs)

def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Stage 2 Homogeneous Spatial GNNs")
    parser.add_argument('--config', type=str, default='configs/gnn_config.yaml',
                        help='Path to the yaml configuration file')
    args = parser.parse_args()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    config_file = os.path.join(base_dir, args.config)
    if not os.path.exists(config_file):
        config_file = args.config
        
    print(f"Loading configuration from: {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        
    # Set random seeds for reproducibility
    seed = config.get('random_seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    paths = config.get('paths', {})
    processed_dir = paths.get('processed_data_dir', 'data/processed')
    output_dir = paths.get('output_dir', 'outputs/gnn_baselines')
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to features and boundary GPKG
    feature_path = os.path.join(base_dir, processed_dir, 'features', 'mock_features.csv')
    spatial_config = config.get('spatial', {})
    gpkg_path = os.path.join(base_dir, spatial_config.get('boundaries_gpkg', 'data/processed/boundaries/census_divisions_epsg3347.gpkg'))
    use_weights = spatial_config.get('use_inverse_distance_weights', False)
    
    # Define model type
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'GCN')
    
    # Training splits
    train_years = [2021, 2022, 2023]
    val_years = [2024]
    test_years = [2025]
    
    feature_cols = [
        'poultry_density_proxy', 'road_density_proxy',
        'ndvi_anomaly', 'ndwi_anomaly', 'temp_anomaly', 'precip_anomaly', 'water_fraction',
        'mallard_abundance_proxy', 'canada_goose_abundance_proxy',
        'outbreak_lag_1', 'outbreak_lag_2'
    ]
    target_col = 'outbreak_next_month'
    
    # Load and build spatiotemporal datasets
    train_snapshots, val_snapshots, test_snapshots, scaler, _, _ = prepare_gnn_data(
        feature_path=feature_path,
        gpkg_path=gpkg_path,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        feature_cols=feature_cols,
        target_col=target_col,
        use_weights=use_weights
    )
    
    print(f"Spatiotemporal snapshot splits:")
    print(f"  - Train months: {len(train_snapshots)}")
    print(f"  - Val months:   {len(val_snapshots)}")
    print(f"  - Test months:  {len(test_snapshots)}")
    
    # Calculate imbalance ratio to weight loss function
    all_train_y = torch.cat([d.y for d in train_snapshots], dim=0)
    pos_count = (all_train_y == 1).sum().item()
    neg_count = (all_train_y == 0).sum().item()
    pos_weight_val = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"Train targets - Neg: {neg_count:.0f}, Pos: {pos_count:.0f}, Imbalance ratio: {pos_weight_val:.2f}")
    
    # Setup device
    device_str = config.get('device', 'cpu')
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Training on device: {device}")
    
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Instantiate Spatial GNN model
    in_channels = len(feature_cols)
    hidden_channels = model_config.get('hidden_channels', 32)
    num_layers = model_config.get('num_layers', 2)
    dropout = model_config.get('dropout', 0.2)
    heads = model_config.get('heads', 1)
    
    model = SpatialGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=1,
        num_layers=num_layers,
        model_type=model_type,
        dropout=dropout,
        heads=heads
    ).to(device)
    
    print(f"Instantiated model {model_type}:")
    print(model)
    
    # Optimizer settings
    train_config = config.get('training', {})
    lr = train_config.get('learning_rate', 0.005)
    weight_decay = train_config.get('weight_decay', 0.0005)
    epochs = train_config.get('epochs', 100)
    patience = train_config.get('early_stopping_patience', 15)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Early stopping loop
    best_val_pr_auc = -1.0
    best_model_state = None
    patience_counter = 0
    
    print("Beginning training loop...")
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, optimizer, criterion, train_snapshots, device)
        val_metrics = evaluate_model(model, val_snapshots, device)
        val_pr_auc = val_metrics['pr_auc']
        
        if epoch % 5 == 0 or epoch == 1:
            train_metrics = evaluate_model(model, train_snapshots, device)
            print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f} | Train PR-AUC: {train_metrics['pr_auc']:.4f} | Val PR-AUC: {val_pr_auc:.4f}")
            
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val PR-AUC: {best_val_pr_auc:.4f}")
                break
                
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Move back to device
        model.to(device)
        
    # Evaluate final best model on all splits
    print("\nTraining completed. Evaluating best model on all splits...")
    results = {}
    for split_name, snapshots in [('train', train_snapshots), ('val', val_snapshots), ('test', test_snapshots)]:
        if len(snapshots) == 0:
            continue
        metrics = evaluate_model(model, snapshots, device)
        results[split_name] = metrics
        print_metrics_summary(metrics, title=f"GNN {model_type} - {split_name.upper()} Split")
        
    # Save best model checkpoint
    model_save_path = os.path.join(output_dir, f"{model_type.lower()}_model.pt")
    torch.save(best_model_state, model_save_path)
    print(f"Saved trained {model_type} state dict to {model_save_path}")
    
    # Save scaler
    scaler_save_path = os.path.join(output_dir, f"{model_type.lower()}_scaler.pkl")
    with open(scaler_save_path, 'wb') as f_out:
        pickle.dump(scaler, f_out)
    print(f"Saved feature scaler to {scaler_save_path}")
    
    # Save metrics report JSON
    metrics_save_path = os.path.join(output_dir, f"{model_type.lower()}_metrics.json")
    save_metrics_report(results, metrics_save_path)
    
if __name__ == "__main__":
    main()
