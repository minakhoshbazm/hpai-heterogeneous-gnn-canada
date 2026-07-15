#!/usr/bin/env python3
"""
scripts/train_graphsage_baseline_v1.py

Trains a 2-layer GraphSAGE baseline on CSD graph monthly snapshots.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

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
    
    # Sort indices by prediction scores descending
    sorted_indices = np.argsort(y_scores)[::-1]
    
    metrics = {
        'PR-AUC': ap,
        'ROC-AUC': roc
    }
    
    for k in [50, 100, 200]:
        top_k = sorted_indices[:k]
        hits = y_true[top_k].sum()
        precision = hits / k
        recall = hits / y_true.sum() if y_true.sum() > 0 else 0.0
        metrics[f'Precision@{k}'] = precision
        metrics[f'Recall@{k}'] = recall
        
    return metrics

def main():
    model_table_path = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
    edges_path = 'data/processed/graph/csd_adjacency_edges_v1_model_scope.csv'
    metrics_out = 'results/baselines/graphsage_v1_metrics.csv'
    preds_out = 'results/baselines/graphsage_v1_test_predictions.csv'

    print("Loading data...")
    df_model = pd.read_csv(model_table_path)
    df_edges = pd.read_csv(edges_path)

    # Determine numeric feature columns
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

    # Train/Test splits
    df_train = df_model[df_model['year'].isin([2022, 2023, 2024])].copy()
    df_test = df_model[df_model['year'] == 2025].copy()

    print(f"Train rows: {len(df_train)}")
    print(f"Test rows: {len(df_test)}")

    # Fit scaler on Train features only
    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols].fillna(0.0))
    df_test[feature_cols] = scaler.transform(df_test[feature_cols].fillna(0.0))

    # Get set of all CSDs in the model scope to create index mapping
    unique_csds = sorted(df_model['csd_uid'].unique())
    csd_to_idx = {csd: i for i, csd in enumerate(unique_csds)}
    num_nodes = len(unique_csds)
    print(f"Total CSD nodes in model scope: {num_nodes}")

    # Build adjacency edge index
    edge_src = df_edges['source_csd_uid'].map(csd_to_idx).values
    edge_dst = df_edges['target_csd_uid'].map(csd_to_idx).values
    
    # PyG expects directed edges, build both directions for undirected graph
    edge_index = torch.tensor(
        np.stack([np.concatenate([edge_src, edge_dst]), np.concatenate([edge_dst, edge_src])], axis=0),
        dtype=torch.long
    )

    # Calculate class weight for BCE loss
    train_pos = df_train['outbreak_next_month'].sum()
    train_neg = len(df_train) - train_pos
    pos_weight = torch.tensor([train_neg / train_pos], dtype=torch.float)
    print(f"BCE Pos Weight: {pos_weight.item():.4f} (Pos: {train_pos}, Neg: {train_neg})")

    # Group train and test data into monthly snapshot PyG Data lists
    def build_snapshots(df):
        snapshots = []
        for (y, m), group in df.groupby(['year', 'month']):
            # Align node feature matrix with the canonical csd_to_idx ordering
            # Unrepresented CSDs in this month get zero feature row and negative target to mask them
            x_npy = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)
            y_npy = np.full((num_nodes,), -1.0, dtype=np.float32)
            
            # Map CSD UIDs to their node indices
            node_indices = group['csd_uid'].map(csd_to_idx).values
            x_npy[node_indices] = group[feature_cols].values
            y_npy[node_indices] = group['outbreak_next_month'].values.astype(np.float32)
            
            x_tensor = torch.tensor(x_npy, dtype=torch.float)
            y_tensor = torch.tensor(y_npy, dtype=torch.float)
            mask_tensor = (y_tensor >= 0.0) # Mask representing CSDs active in this month
            
            snapshots.append(Data(x=x_tensor, y=y_tensor, mask=mask_tensor, edge_index=edge_index))
        return snapshots

    train_snapshots = build_snapshots(df_train)
    test_snapshots = build_snapshots(df_test)

    # Initialize model
    model = GraphSAGEModel(in_channels=len(feature_cols), hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop
    epochs = 30
    print("Training GraphSAGE...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for data in train_snapshots:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            # Apply mask to loss calculation
            loss = criterion(out[data.mask], data.y[data.mask])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= len(train_snapshots)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f}")

    # Evaluation on 2025 Test set
    model.eval()
    all_y_true = []
    all_y_scores = []
    
    # Store predictions matching the test dataframe rows
    test_rows_pred = []

    with torch.no_grad():
        for (y, m), group in df_test.groupby(['year', 'month']):
            # Predict snapshot
            x_npy = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)
            node_indices = group['csd_uid'].map(csd_to_idx).values
            x_npy[node_indices] = group[feature_cols].values
            
            x_tensor = torch.tensor(x_npy, dtype=torch.float)
            out = model(x_tensor, edge_index)
            probs = torch.sigmoid(out).cpu().numpy()
            
            # Map predictions back to the group dataframe order
            group_probs = probs[node_indices]
            
            # Save predictions
            pred_df = pd.DataFrame({
                'csd_uid': group['csd_uid'],
                'year': y,
                'month': m,
                'outbreak_next_month': group['outbreak_next_month'],
                'prediction_score': group_probs
            })
            test_rows_pred.append(pred_df)
            
            all_y_true.extend(group['outbreak_next_month'].values)
            all_y_scores.extend(group_probs)

    y_true_all = np.array(all_y_true)
    y_scores_all = np.array(all_y_scores)
    
    # Metrics calculation
    metrics = evaluate_metrics(y_true_all, y_scores_all)

    # Print results
    print("\n==============================================")
    print("GraphSAGE V1 Evaluation Metrics:")
    for metric_name, val in metrics.items():
        print(f"  {metric_name}: {val:.4f}")
    
    print("\nComparison against baselines:")
    print("  Env + Lag Logistic Regression: PR-AUC 0.0635, Precision@50 0.1200")
    print("  Time-Persistence Baseline:     PR-AUC 0.0405, Precision@50 0.2000")
    print("==============================================")

    # Save outputs
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_out, index=False)
    pd.concat(test_rows_pred, ignore_index=True).to_csv(preds_out, index=False)
    print(f"\nSaved metrics to: {metrics_out}")
    print(f"Saved predictions to: {preds_out}")

if __name__ == '__main__':
    main()
