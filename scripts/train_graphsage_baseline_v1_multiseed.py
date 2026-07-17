#!/usr/bin/env python3
"""
scripts/train_graphsage_baseline_v1_multiseed.py

Runs the GraphSAGE baseline GNN model across multiple seeds (11, 22, 33)
to check result stability.
"""

import os
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
    
    # Sort indices by prediction scores descending
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

def main():
    model_table_path = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
    edges_path = 'data/processed/graph/csd_adjacency_edges_v1_model_scope.csv'
    metrics_out = 'results/baselines/graphsage_v1_multiseed_metrics.csv'
    summary_out = 'results/baselines/graphsage_v1_multiseed_summary.csv'

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

    # Train/Test splits
    df_train = df_model[df_model['year'].isin([2022, 2023, 2024])].copy()
    df_test = df_model[df_model['year'] == 2025].copy()

    # Fit scaler on Train features only
    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols].fillna(0.0))
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

    # Calculate class weight for BCE loss
    train_pos = df_train['outbreak_next_month'].sum()
    train_neg = len(df_train) - train_pos
    pos_weight = torch.tensor([train_neg / train_pos], dtype=torch.float)

    # Group train and test data into monthly snapshot PyG Data lists
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
    test_snapshots = build_snapshots(df_test)

    seeds = [11, 22, 33]
    seed_results = []
    epochs = 30

    for s in seeds:
        print(f"\nTraining GraphSAGE with seed {s}...")
        set_seed(s)
        
        model = GraphSAGEModel(in_channels=len(feature_cols), hidden_channels=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(1, epochs + 1):
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
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f}")

        # Evaluate
        model.eval()
        all_y_true = []
        all_y_scores = []
        
        with torch.no_grad():
            for (y, m), group in df_test.groupby(['year', 'month']):
                x_npy = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)
                node_indices = group['csd_uid'].map(csd_to_idx).values
                x_npy[node_indices] = group[feature_cols].values
                
                x_tensor = torch.tensor(x_npy, dtype=torch.float)
                out = model(x_tensor, edge_index)
                probs = torch.sigmoid(out).cpu().numpy()
                
                all_y_true.extend(group['outbreak_next_month'].values)
                all_y_scores.extend(probs[node_indices])

        y_true_all = np.array(all_y_true)
        y_scores_all = np.array(all_y_scores)
        
        metrics = evaluate_metrics(y_true_all, y_scores_all)
        metrics['seed'] = s
        seed_results.append(metrics)
        
        print(f"  Seed {s} Results: PR-AUC={metrics['pr_auc']:.4f}, Precision@50={metrics['precision_50']:.3f}, Recall@200={metrics['recall_200']:.4f}")

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
    print("                     Multi-Seed GraphSAGE Evaluation")
    print("=========================================================================")
    print(df_metrics.to_string(index=False, formatters={
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

    # Print conclusion
    print("\n--- Conclusions ---")
    mean_pr_auc = mean_row['pr_auc']
    print(f"- Mean GNN PR-AUC: {mean_pr_auc:.4f} (vs Env + Lag Logistic Regression: 0.0635)")
    better_baseline = mean_pr_auc > 0.0635
    print(f"- Consistently better than baseline: {better_baseline}")
    print(f"- Standard Deviation of PR-AUC: {std_row['pr_auc']:.4f}")
    stable = std_row['pr_auc'] < 0.015
    print(f"- Results stable enough to report: {stable}")

    # Save
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    df_metrics.to_csv(metrics_out, index=False)
    df_summary.to_csv(summary_out, index=False)
    print(f"\nSaved multiseed metrics to: {metrics_out}")
    print(f"Saved summary statistics to: {summary_out}")

if __name__ == '__main__':
    main()
