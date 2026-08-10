"""
Run Strict Regional Spatial-Temporal Validation for GraphSAGE GNN Model (Strict Identifiers Exclusion).

Purpose:
Performs leave-one-region-out spatial-temporal validation for GraphSAGE:
  - Training Graph: Induced subgraph of nodes OUTSIDE the held-out regional block (2022-01 to 2024-09).
  - Validation Graph: Induced subgraph of nodes OUTSIDE the held-out regional block (2024-10 to 2024-12).
  - Test Graph: Induced subgraph of nodes INSIDE the held-out regional block ONLY (2025).
  - Edge Masking: NO edges cross between the held-out block and outside regions during test inference.
  - Features: Retains 7 approved lag/history features, but strictly excludes all identifiers (csd_uid, ccs_uid, etc.) and raw outbreak_current_month.

Regional Blocks:
  - Atlantic: NL (10), PEI (11), NS (12), NB (13)
  - Quebec_Ontario: QC (24), ON (35)
  - Prairies: MB (46), SK (47), AB (48)
  - British_Columbia: BC (59)

Inputs:
  data/processed/modeling/model_table_v1_environment_with_lags.csv
  data/processed/graph/csd_adjacency_edges_v1_model_scope.csv

Outputs:
  results/baselines/regional_spatiotemporal_graphsage_multiseed_metrics.csv
  results/baselines/regional_spatiotemporal_graphsage_summary.csv
  results/baselines/regional_spatiotemporal_graphsage_predictions.csv
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
METRICS_OUT = 'results/baselines/regional_spatiotemporal_graphsage_multiseed_metrics.csv'
SUMMARY_OUT = 'results/baselines/regional_spatiotemporal_graphsage_summary.csv'
PREDS_OUT = 'results/baselines/regional_spatiotemporal_graphsage_predictions.csv'

REGIONAL_BLOCKS = {
    'Atlantic': [10, 11, 12, 13],
    'Quebec_Ontario': [24, 35],
    'Prairies': [46, 47, 48],
    'British_Columbia': [59]
}

EXCLUDE_EXPLICIT_GNN = {
    'csd_uid', 'csd_name', 'province_code', 'province_name', 'ccs_uid',
    'year', 'month', 'outbreak_next_month',
    'dw_fill_method', 'climate_extraction_method', 'glw2020_chicken_extraction_method',
    'glw2015_duck_extraction_method', 'road_data_coverage',
    'outbreak_current_month'
}

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
    print("  Strict Regional Spatial-Temporal Validation for GraphSAGE Model (STRICT EXCLUSIONS)")
    print("==========================================================================================")

    df_model = pd.read_csv(MODEL_TABLE_PATH)
    df_edges = pd.read_csv(EDGES_PATH)

    feature_cols = [c for c in df_model.columns if c not in EXCLUDE_EXPLICIT_GNN and pd.api.types.is_numeric_dtype(df_model[c])]
    feature_cols = sorted(feature_cols)

    # ------------------ FEATURE AUDIT ASSERTS ------------------
    print("\nExecuting Feature Audit Assertions for GraphSAGE GNN...")
    print(f"Num selected features (environment + 7 lags): {len(feature_cols)}")
    print("Features selected:", feature_cols)

    assert 'ccs_uid' not in feature_cols, "ERROR: ccs_uid is present in GraphSAGE feature list!"
    uid_features = [c for c in feature_cols if 'uid' in c.lower()]
    assert len(uid_features) == 0, f"ERROR: Features containing 'uid' detected in GraphSAGE: {uid_features}"
    assert 'outbreak_current_month' not in feature_cols, "ERROR: Raw outbreak_current_month detected in GraphSAGE features!"

    approved_lags = [
        'same_csd_outbreak_current_month', 'same_csd_outbreak_prev_1_month',
        'same_csd_outbreak_count_past_3_months', 'same_csd_outbreak_count_past_6_months',
        'same_csd_outbreak_count_past_12_months', 'province_outbreak_count_current_month',
        'province_outbreak_count_past_3_months'
    ]
    for lag in approved_lags:
        assert lag in feature_cols, f"ERROR: Approved lag feature {lag} missing from GraphSAGE!"

    assert len(feature_cols) == 50, f"Expected 50 features for GraphSAGE (43 env + 7 lags), got {len(feature_cols)}"
    print("SUCCESS: GraphSAGE feature list strictly audited (50 features: 43 env + 7 lags, ccs_uid & outbreak_current_month excluded).")

    # Sort data chronologically
    df_model = df_model.sort_values(by=['year', 'month', 'csd_uid']).reset_index(drop=True)

    seeds = [11, 22, 33]
    all_seed_results = []
    all_preds_list = []

    for block_name, prov_codes in REGIONAL_BLOCKS.items():
        print(f"\n==========================================================================================")
        print(f"Processing Regional Block: {block_name} (Provinces: {prov_codes})")
        print(f"==========================================================================================")

        block_csds = set(df_model[df_model['province_code'].isin(prov_codes)]['csd_uid'].unique())
        other_csds = set(df_model[~df_model['province_code'].isin(prov_codes)]['csd_uid'].unique())

        df_train = df_model[
            (~df_model['csd_uid'].isin(block_csds)) &
            (df_model['year'].isin([2022, 2023]) | ((df_model['year'] == 2024) & (df_model['month'] <= 9)))
        ].copy()

        df_val = df_model[
            (~df_model['csd_uid'].isin(block_csds)) &
            ((df_model['year'] == 2024) & (df_model['month'] >= 10))
        ].copy()

        df_test = df_model[
            (df_model['csd_uid'].isin(block_csds)) &
            (df_model['year'] == 2025)
        ].copy()

        train_rows, val_rows, test_rows = len(df_train), len(df_val), len(df_test)
        train_pos = int(df_train['outbreak_next_month'].sum())
        val_pos = int(df_val['outbreak_next_month'].sum())
        test_pos = int(df_test['outbreak_next_month'].sum())
        test_prevalence = test_pos / test_rows if test_rows > 0 else 0.0

        small_sample_warning = ""
        if test_pos < 10:
            small_sample_warning = f"WARNING: Test positive count ({test_pos}) is very small (< 10); metrics may be unstable."

        print(f"  Train: {train_rows:,} rows | Positives: {train_pos}")
        print(f"  Val:   {val_rows:,} rows | Positives: {val_pos}")
        print(f"  Test:  {test_rows:,} rows | Positives: {test_pos} (Prevalence: {test_prevalence:.6f})")
        if small_sample_warning:
            print(f"  {small_sample_warning}")

        scaler = StandardScaler()
        df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols].fillna(0.0))
        df_val[feature_cols] = scaler.transform(df_val[feature_cols].fillna(0.0))
        df_test[feature_cols] = scaler.transform(df_test[feature_cols].fillna(0.0))

        other_csd_list = sorted(other_csds)
        other_csd_to_idx = {csd: i for i, csd in enumerate(other_csd_list)}
        num_other_nodes = len(other_csd_list)

        block_csd_list = sorted(block_csds)
        block_csd_to_idx = {csd: i for i, csd in enumerate(block_csd_list)}
        num_block_nodes = len(block_csd_list)

        # Build induced subgraphs
        train_edges = df_edges[
            df_edges['source_csd_uid'].isin(other_csds) & df_edges['target_csd_uid'].isin(other_csds)
        ]
        tr_src = train_edges['source_csd_uid'].map(other_csd_to_idx).values
        tr_dst = train_edges['target_csd_uid'].map(other_csd_to_idx).values
        train_edge_index = torch.tensor(
            np.stack([np.concatenate([tr_src, tr_dst]), np.concatenate([tr_dst, tr_src])], axis=0),
            dtype=torch.long
        )

        test_edges = df_edges[
            df_edges['source_csd_uid'].isin(block_csds) & df_edges['target_csd_uid'].isin(block_csds)
        ]
        te_src = test_edges['source_csd_uid'].map(block_csd_to_idx).values
        te_dst = test_edges['target_csd_uid'].map(block_csd_to_idx).values
        test_edge_index = torch.tensor(
            np.stack([np.concatenate([te_src, te_dst]), np.concatenate([te_dst, te_src])], axis=0),
            dtype=torch.long
        )

        train_neg = train_rows - train_pos
        pos_weight = torch.tensor([train_neg / train_pos], dtype=torch.float)

        def build_snapshots(df_sub, csd_to_idx_map, n_nodes, e_idx):
            snapshots = []
            for (y, m), group in df_sub.groupby(['year', 'month']):
                x_npy = np.zeros((n_nodes, len(feature_cols)), dtype=np.float32)
                y_npy = np.full((n_nodes,), -1.0, dtype=np.float32)
                
                node_indices = group['csd_uid'].map(csd_to_idx_map).values
                x_npy[node_indices] = group[feature_cols].values
                y_npy[node_indices] = group['outbreak_next_month'].values.astype(np.float32)
                
                x_tensor = torch.tensor(x_npy, dtype=torch.float)
                y_tensor = torch.tensor(y_npy, dtype=torch.float)
                mask_tensor = (y_tensor >= 0.0)
                
                snapshots.append(Data(x=x_tensor, y=y_tensor, mask=mask_tensor, edge_index=e_idx))
            return snapshots

        train_snapshots = build_snapshots(df_train, other_csd_to_idx, num_other_nodes, train_edge_index)
        val_snapshots = build_snapshots(df_val, other_csd_to_idx, num_other_nodes, train_edge_index)
        test_snapshots = build_snapshots(df_test, block_csd_to_idx, num_block_nodes, test_edge_index)

        block_preds = pd.DataFrame({
            'regional_block': block_name,
            'csd_uid': df_test['csd_uid'].values,
            'csd_name': df_test['csd_name'].values,
            'province_code': df_test['province_code'].values,
            'year': df_test['year'].values,
            'month': df_test['month'].values,
            'actual': df_test['outbreak_next_month'].values
        })

        max_epochs = 100
        patience = 10

        for s in seeds:
            print(f"  Training GraphSAGE on {block_name} (Seed {s})...")
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

                val_true, val_scores = get_snapshot_predictions(model, val_snapshots, train_edge_index)
                val_ap = average_precision_score(val_true, val_scores)

                if val_ap > best_val_ap:
                    best_val_ap = val_ap
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    break

            model.load_state_dict(best_model_state)
            test_true, test_scores = get_snapshot_predictions(model, test_snapshots, test_edge_index)

            ap = average_precision_score(test_true, test_scores)
            roc_auc = roc_auc_score(test_true, test_scores)
            k_metrics = topk_metrics(test_true, test_scores, [50, 100, 200], test_pos)

            pr_auc_lift = ap / test_prevalence if test_prevalence > 0 else 0.0

            res_rec = {
                'regional_block': block_name,
                'seed': s,
                'train_rows': train_rows,
                'train_positives': train_pos,
                'val_rows': val_rows,
                'val_positives': val_pos,
                'test_rows': test_rows,
                'test_positives': test_pos,
                'test_prevalence': test_prevalence,
                'val_pr_auc': best_val_ap,
                'pr_auc': ap,
                'pr_auc_lift': pr_auc_lift,
                'roc_auc': roc_auc,
                'precision_50': k_metrics['precision_50'],
                'recall_50': k_metrics['recall_50'],
                'precision_100': k_metrics['precision_100'],
                'recall_100': k_metrics['recall_100'],
                'precision_200': k_metrics['precision_200'],
                'recall_200': k_metrics['recall_200'],
                'small_sample_warning': small_sample_warning
            }
            all_seed_results.append(res_rec)
            block_preds[f'graphsage_prob_seed_{s}'] = test_scores

            print(f"    Seed {s} Results -> Val PR-AUC: {best_val_ap:.4f} | Test PR-AUC: {ap:.4f} (Lift: {pr_auc_lift:.2f}x) | ROC-AUC: {roc_auc:.4f}")

        all_preds_list.append(block_preds)

    df_metrics = pd.DataFrame(all_seed_results)
    df_preds = pd.concat(all_preds_list, ignore_index=True)

    metric_cols_to_avg = ['val_pr_auc', 'pr_auc', 'pr_auc_lift', 'roc_auc', 'precision_50', 'recall_50', 'precision_100', 'recall_100', 'precision_200', 'recall_200']
    summary_rows = []

    for block_name, grp in df_metrics.groupby('regional_block'):
        row_dict = {
            'regional_block': block_name,
            'test_positives': grp['test_positives'].iloc[0],
            'test_prevalence': grp['test_prevalence'].iloc[0],
            'small_sample_warning': grp['small_sample_warning'].iloc[0]
        }
        for col in metric_cols_to_avg:
            row_dict[f'{col}_mean'] = grp[col].mean()
            row_dict[f'{col}_std'] = grp[col].std()
        summary_rows.append(row_dict)

    df_summary = pd.DataFrame(summary_rows)

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(PREDS_OUT), exist_ok=True)

    df_metrics.to_csv(METRICS_OUT, index=False)
    df_summary.to_csv(SUMMARY_OUT, index=False)
    df_preds.to_csv(PREDS_OUT, index=False)

    print("\n==========================================================================================")
    print("   STRICT REGIONAL SPATIAL-TEMPORAL GRAPHSAGE MULTISEED SUMMARY RESULTS (CORRECTED)")
    print("==========================================================================================")
    disp_cols = ['regional_block', 'test_positives', 'test_prevalence', 'pr_auc_mean', 'pr_auc_std', 'pr_auc_lift_mean', 'roc_auc_mean', 'precision_50_mean', 'recall_50_mean']
    print(df_summary[disp_cols].to_string(index=False))

    print(f"\nSaved multiseed metrics to: {METRICS_OUT}")
    print(f"Saved summary to: {SUMMARY_OUT}")
    print(f"Saved predictions to: {PREDS_OUT}")

if __name__ == '__main__':
    main()
