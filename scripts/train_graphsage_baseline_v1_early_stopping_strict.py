"""
Train Corrected Strict GraphSAGE GNN National Baseline (Early Stopping + Multiseed).

Purpose:
Fits multi-seed GraphSAGE GNN on the national temporal split:
  - Train: 2022-01 through 2024-09
  - Val:   2024-10 through 2024-12
  - Test:  2025
  - Features: STRICT 50 features (ENV_LAG_FEATURES). ccs_uid and raw outbreak_current_month explicitly excluded.

Inputs:
  data/processed/modeling/model_table_v1_environment_with_lags.csv
  data/processed/graph/csd_adjacency_edges_v1_model_scope.csv

Outputs:
  results/baselines/graphsage_v1_early_stopping_strict_multiseed_metrics.csv
  results/baselines/graphsage_v1_early_stopping_strict_summary.csv
  results/baselines/graphsage_v1_early_stopping_strict_test_predictions.csv
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

MODEL_TABLE_PATH = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
EDGES_PATH = 'data/processed/graph/csd_adjacency_edges_v1_model_scope.csv'
METRICS_OUT = 'results/baselines/graphsage_v1_early_stopping_strict_multiseed_metrics.csv'
SUMMARY_OUT = 'results/baselines/graphsage_v1_early_stopping_strict_summary.csv'
PREDS_OUT = 'results/baselines/graphsage_v1_early_stopping_strict_test_predictions.csv'

EXCLUDE_EXPLICIT_GNN = {
    'csd_uid', 'csd_name', 'province_code', 'province_name', 'ccs_uid',
    'year', 'month', 'outbreak_next_month',
    'dw_fill_method', 'climate_extraction_method', 'glw2020_chicken_extraction_method',
    'glw2015_duck_extraction_method', 'road_data_coverage',
    'outbreak_current_month'
}

APPROVED_LAG_FEATURES = [
    'same_csd_outbreak_current_month',
    'same_csd_outbreak_prev_1_month',
    'same_csd_outbreak_count_past_3_months',
    'same_csd_outbreak_count_past_6_months',
    'same_csd_outbreak_count_past_12_months',
    'province_outbreak_count_current_month',
    'province_outbreak_count_past_3_months'
]

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

def topk_metrics(y_true, probs, k_list, total_positives):
    sorted_idx = np.argsort(probs)[::-1]
    sorted_y = y_true[sorted_idx]
    results = {}
    for k in k_list:
        top_k = sorted_y[:k]
        pos_in_top = int(top_k.sum())
        precision = pos_in_top / k
        recall = pos_in_top / total_positives if total_positives > 0 else 0.0
        results[f'precision_{k}'] = precision
        results[f'recall_{k}'] = recall
    return results

def get_snapshot_predictions(model, snapshots, edge_index):
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
    print("==========================================================================================")
    print("  Train Strict GraphSAGE Baseline Model (Early Stopping + Multiseed)")
    print("==========================================================================================")

    df_model = pd.read_csv(MODEL_TABLE_PATH)
    df_edges = pd.read_csv(EDGES_PATH)

    feature_cols = [c for c in df_model.columns if c not in EXCLUDE_EXPLICIT_GNN and pd.api.types.is_numeric_dtype(df_model[c])]
    feature_cols = sorted(feature_cols)

    # ------------------ STRICT FEATURE AUDIT ASSERTS ------------------
    print("\nExecuting Strict Feature Audit Assertions for National GraphSAGE GNN...")
    print(f"Num selected features: {len(feature_cols)} (Expected: 50)")
    print("Features selected:", feature_cols)

    assert len(feature_cols) == 50, f"ERROR: Feature count is {len(feature_cols)}, expected exactly 50!"
    assert 'ccs_uid' not in feature_cols, "ERROR: ccs_uid is present in GraphSAGE feature list!"
    uid_features = [c for c in feature_cols if 'uid' in c.lower()]
    assert len(uid_features) == 0, f"ERROR: Features containing 'uid' detected in GraphSAGE: {uid_features}"
    assert 'outbreak_current_month' not in feature_cols, "ERROR: Raw outbreak_current_month detected in GraphSAGE features!"

    for lag in APPROVED_LAG_FEATURES:
        assert lag in feature_cols, f"ERROR: Approved lag feature {lag} missing from GraphSAGE!"

    print("SUCCESS: National GraphSAGE feature list strictly audited (50 features: 43 env + 7 lags, ccs_uid & outbreak_current_month excluded).")

    # Sort data chronologically
    df_model = df_model.sort_values(by=['year', 'month', 'csd_uid']).reset_index(drop=True)

    # Train/Val/Test splits
    df_train = df_model[
        df_model['year'].isin([2022, 2023]) | 
        ((df_model['year'] == 2024) & (df_model['month'] <= 9))
    ].copy()
    
    df_val = df_model[
        ((df_model['year'] == 2024) & (df_model['month'] >= 10))
    ].copy()
    
    df_test = df_model[df_model['year'] == 2025].copy()

    train_rows, val_rows, test_rows = len(df_train), len(df_val), len(df_test)
    train_pos = int(df_train['outbreak_next_month'].sum())
    val_pos = int(df_val['outbreak_next_month'].sum())
    test_pos = int(df_test['outbreak_next_month'].sum())

    print(f"\nDataset Split Summary:")
    print(f"  Train split: {train_rows:,} rows, {train_pos} positives ({train_pos / train_rows * 100:.4f}%)")
    print(f"  Val split:   {val_rows:,} rows, {val_pos} positives ({val_pos / val_rows * 100:.4f}%)")
    print(f"  Test split:  {test_rows:,} rows, {test_pos} positives ({test_pos / test_rows * 100:.4f}%)")

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

    # Class weight for BCE loss using train split only
    train_neg = train_rows - train_pos
    pos_weight = torch.tensor([train_neg / train_pos], dtype=torch.float)

    # Group train, val, and test data into monthly snapshot PyG Data lists
    def build_snapshots(df_sub):
        snapshots = []
        for (y, m), group in df_sub.groupby(['year', 'month']):
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

    test_preds_dict = {
        'csd_uid': df_test['csd_uid'].values,
        'csd_name': df_test['csd_name'].values,
        'province_code': df_test['province_code'].values,
        'year': df_test['year'].values,
        'month': df_test['month'].values,
        'actual': df_test['outbreak_next_month'].values
    }

    for s in seeds:
        print(f"\nTraining GraphSAGE with seed {s} and early stopping...")
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
            
            # Evaluate on val
            val_true, val_scores = get_snapshot_predictions(model, val_snapshots, edge_index)
            val_ap = average_precision_score(val_true, val_scores)
            
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

        # Restore best model state and evaluate on Test
        model.load_state_dict(best_model_state)
        test_true, test_scores = get_snapshot_predictions(model, test_snapshots, edge_index)
        
        ap = average_precision_score(test_true, test_scores)
        roc = roc_auc_score(test_true, test_scores)
        k_metrics = topk_metrics(test_true, test_scores, [50, 100, 200], test_pos)

        res = {
            'seed': s,
            'val_pr_auc': best_val_ap,
            'pr_auc': ap,
            'roc_auc': roc,
            'precision_50': k_metrics['precision_50'],
            'recall_50': k_metrics['recall_50'],
            'precision_100': k_metrics['precision_100'],
            'recall_100': k_metrics['recall_100'],
            'precision_200': k_metrics['precision_200'],
            'recall_200': k_metrics['recall_200']
        }
        seed_results.append(res)
        test_preds_dict[f'strict_graphsage_seed_{s}_prob'] = test_scores
        
        print(f"  Seed {s} Results: Val PR-AUC={best_val_ap:.4f}, Test PR-AUC={ap:.4f}, Test ROC-AUC={roc:.4f}, Precision@50={k_metrics['precision_50']:.3f}")

    df_metrics = pd.DataFrame(seed_results)
    
    # Summary across seeds
    mean_row = df_metrics.drop(columns='seed').mean()
    std_row = df_metrics.drop(columns='seed').std()
    
    df_summary = pd.DataFrame({
        'Metric': list(mean_row.index),
        'Mean': list(mean_row.values),
        'Std': list(std_row.values)
    })

    print("\n=========================================================================")
    print("        STRICT NATIONAL GRAPHSAGE MULTISEED EVALUATION")
    print("=========================================================================")
    print(df_metrics.to_string(index=False))
    print("\n------------------------- Summary Statistics -------------------------")
    print(df_summary.to_string(index=False))
    print("=========================================================================")

    # Save outputs
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(PREDS_OUT), exist_ok=True)

    df_metrics.to_csv(METRICS_OUT, index=False)
    df_summary.to_csv(SUMMARY_OUT, index=False)
    pd.DataFrame(test_preds_dict).to_csv(PREDS_OUT, index=False)

    print(f"\nSaved multiseed metrics to: {METRICS_OUT}")
    print(f"Saved summary to: {SUMMARY_OUT}")
    print(f"Saved predictions to: {PREDS_OUT}")

if __name__ == '__main__':
    main()
