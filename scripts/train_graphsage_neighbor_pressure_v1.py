#!/usr/bin/env python3
"""
scripts/train_graphsage_neighbor_pressure_v1.py

Trains GraphSAGE GNN models using environment + lags + neighbor outbreak pressure features.
Uses validation split and early stopping.
"""

import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        return self.lin(x).squeeze(-1)

def evaluate_metrics(y_true, y_scores):
    ap = average_precision_score(y_true, y_scores)
    roc = roc_auc_score(y_true, y_scores)
    
    sorted_indices = np.argsort(y_scores)[::-1]
    
    metrics = {
        'pr_auc': ap,
        'roc_auc': roc
    }
    
    for k in [50, 100, 200]:
        top_k = sorted_indices[:k]
        hits = y_true[top_k].sum()
        precision = hits / k
        recall = hits / y_true.sum() if y_true.sum() > 0 else 0.0
        metrics[f'precision_{k}'] = precision
        metrics[f'recall_{k}'] = recall
        
    return metrics

def get_snapshot_predictions(model, snapshots, feature_cols, csd_to_idx, num_nodes, edge_index):
    model.eval()
    all_y_true = []
    all_y_scores = []
    
    with torch.no_grad():
        for data in snapshots:
            out = model(data.x, edge_index)
            probs = torch.sigmoid(out).cpu().numpy()
            
            mask_np = data.mask.cpu().numpy()
            all_y_true.extend(data.y.cpu().numpy()[mask_np])
            all_y_scores.extend(probs[mask_np])
            
    return np.array(all_y_true), np.array(all_y_scores)

def main():
    model_table_path = 'data/processed/modeling/model_table_v1_environment_with_lags_and_neighbor_pressure.csv'
    edges_path = 'data/processed/graph/csd_adjacency_edges_v1_model_scope.csv'
    metrics_out = 'results/baselines/graphsage_neighbor_pressure_v1_metrics.csv'
    summary_out = 'results/baselines/graphsage_neighbor_pressure_v1_summary.csv'

    print("Loading data...")
    df_model = pd.read_csv(model_table_path)
    df_edges = pd.read_csv(edges_path)

    # Exclude metadata and non-numeric columns
    exclude_cols = {
        'csd_uid', 'csd_name', 'province_code', 'province_name', 
        'year', 'month', 'outbreak_next_month',
        'dw_fill_method', 'climate_extraction_method', 'glw2020_chicken_extraction_method',
        'glw2015_duck_extraction_method', 'road_data_coverage'
    }
    feature_cols = [c for c in df_model.columns if c not in exclude_cols]
    print(f"Num features: {len(feature_cols)}")

    # Sort data chronologically to ensure consistent snapshots
    df_model = df_model.sort_values(by=['year', 'month', 'csd_uid']).reset_index(drop=True)

    # Train/Val/Test splits
    # Train: 2022-01 through 2024-09
    # Val: 2024-10 through 2024-12
    # Test: 2025
    df_train = df_model[
        df_model['year'].isin([2022, 2023]) | 
        ((df_model['year'] == 2024) & (df_model['month'] <= 9))
    ].copy()
    
    df_val = df_model[
        (df_model['year'] == 2024) & (df_model['month'] >= 10)
    ].copy()
    
    df_test = df_model[df_model['year'] == 2025].copy()

    # Fit scaler on Train features only
    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols].fillna(0.0))
    df_val[feature_cols] = scaler.transform(df_val[feature_cols].fillna(0.0))
    df_test[feature_cols] = scaler.transform(df_test[feature_cols].fillna(0.0))

    # CSD mapping
    unique_csds = sorted(df_model['csd_uid'].unique())
    csd_to_idx = {csd: i for i, csd in enumerate(unique_csds)}
    num_nodes = len(unique_csds)

    # Build adjacency edge index
    edge_src = df_edges['source_csd_uid'].map(csd_to_idx).values
    edge_dst = df_edges['target_csd_uid'].map(csd_to_idx).values
    
    edge_index = torch.tensor(
        np.stack([np.concatenate([edge_src, edge_dst]), np.concatenate([edge_dst, edge_src])], axis=0),
        dtype=torch.long
    )

    # Calculate class weight for BCE loss using train split only
    train_pos = df_train['outbreak_next_month'].sum()
    train_neg = len(df_train) - train_pos
    pos_weight = torch.tensor([train_neg / train_pos], dtype=torch.float)

    # Group train, val, and test data into monthly snapshot PyG Data lists
    def build_snapshots(df):
        snapshots = []
        for (y, m), group in df.groupby(['year', 'month']):
            x_npy = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)
            y_npy = np.full((num_nodes,), -1.0, dtype=np.float32)
            
            node_indices = group['csd_uid'].map(csd_to_idx).values
            x_npy[node_indices] = group[feature_cols].values
            y_npy[node_indices] = group['outbreak_next_month'].values.astype(np.float32)
            
            x_tensor = torch.tensor(x_npy, dtype=torch.float)
            y_tensor = torch.tensor(y_npy, dtype=torch.float)
            mask_tensor = (y_tensor >= 0.0)
            
            snapshots.append(Data(x=x_tensor, y=y_tensor, mask=mask_tensor, edge_index=edge_index))
        return snapshots

    train_snapshots = build_snapshots(df_train)
    val_snapshots = build_snapshots(df_val)
    test_snapshots = build_snapshots(df_test)

    seeds = [11, 22, 33]
    seed_results = []
    max_epochs = 100
    patience = 10

    for s in seeds:
        print(f"\nTraining GraphSAGE with seed {s} and neighbor pressure features...")
        set_seed(s)
        
        model = GraphSAGEModel(in_channels=len(feature_cols), hidden_channels=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_ap = -1.0
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_loss = 0.0
            for data in train_snapshots:
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = criterion(out[data.mask], data.y[data.mask])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            epoch_loss /= len(train_snapshots)
            
            # Evaluate on validation split
            val_true, val_scores = get_snapshot_predictions(model, val_snapshots, feature_cols, csd_to_idx, num_nodes, edge_index)
            val_ap = average_precision_score(val_true, val_scores)
            
            # Check validation improvement
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {epoch_loss:.4f} | Val PR-AUC: {val_ap:.4f} | Best Val PR-AUC: {best_val_ap:.4f}")
                
            if epochs_no_improve >= patience:
                print(f"  Early stopping triggered at epoch {epoch}. Restoring best model state.")
                break

        # Restore best model state
        model.load_state_dict(best_model_state)
        
        # Evaluate on Test set
        test_true, test_scores = get_snapshot_predictions(model, test_snapshots, feature_cols, csd_to_idx, num_nodes, edge_index)
        metrics = evaluate_metrics(test_true, test_scores)
        metrics['val_pr_auc'] = best_val_ap
        metrics['seed'] = s
        seed_results.append(metrics)
        
        print(f"  Seed {s} Results: Val PR-AUC={best_val_ap:.4f}, Test PR-AUC={metrics['pr_auc']:.4f}, Test ROC-AUC={metrics['roc_auc']:.4f}, Precision@50={metrics['precision_50']:.3f}")

    df_metrics = pd.DataFrame(seed_results)
    
    # Calculate Mean & Standard Deviation
    mean_row = df_metrics.drop(columns='seed').mean()
    std_row = df_metrics.drop(columns='seed').std()
    
    summary_data = {
        'Metric': list(mean_row.index),
        'Mean': list(mean_row.values),
        'Std': list(std_row.values)
    }
    df_summary = pd.DataFrame(summary_data)

    print("\n=========================================================================")
    print("      GraphSAGE with Neighbor Outbreak Pressure Evaluation")
    print("=========================================================================")
    print(df_metrics.to_string(index=False, formatters={
        'val_pr_auc': '{:,.4f}'.format,
        'pr_auc': '{:,.4f}'.format,
        'roc_auc': '{:,.4f}'.format,
        'precision_50': '{:,.3f}'.format,
        'recall_50': '{:,.4f}'.format,
        'precision_100': '{:,.3f}'.format,
        'recall_100': '{:,.4f}'.format,
        'precision_200': '{:,.3f}'.format,
        'recall_200': '{:,.4f}'.format,
    }))
    print("\n------------------------- Summary Statistics -------------------------")
    print(df_summary.to_string(index=False, formatters={
        'Mean': '{:,.4f}'.format,
        'Std': '{:,.4f}'.format
    }))
    print("=========================================================================")

    # Print comparison
    print("\n--- Baseline Comparisons ---")
    mean_pr_auc = mean_row['pr_auc']
    print(f"- GNN + Neighbor Pressure Mean PR-AUC: {mean_pr_auc:.4f} (Std: {std_row['pr_auc']:.4f})")
    print(f"- GNN baseline (No Neighbor Pressure):  Mean PR-AUC 0.0735 (Std: 0.0122)")
    print(f"- Neighbor-pressure Logistic Reg AP:    0.0653")
    print(f"- Neighbor-pressure Random Forest AP:   0.0490")

    print("\n--- Interpretation ---")
    if mean_pr_auc > 0.0735:
        print("- GraphSAGE improved: neighbor-pressure features add useful spatial information beyond graph message passing.")
    else:
        print("- GraphSAGE did not improve: manual neighbor-pressure features may be partly redundant with graph message passing.")

    print("\n--- Validation Note ---")
    print("- Current validation is temporal: train on past years and test on 2025.")
    print("- Spatial validation is still needed later, such as leave-one-province-out or regional block holdout.")
    print("- Spatial validation was NOT implemented in this script.")

    # Save
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    df_metrics.to_csv(metrics_out, index=False)
    df_summary.to_csv(summary_out, index=False)
    print(f"\nSaved metrics to: {metrics_out}")
    print(f"Saved summary to: {summary_out}")

if __name__ == '__main__':
    main()
