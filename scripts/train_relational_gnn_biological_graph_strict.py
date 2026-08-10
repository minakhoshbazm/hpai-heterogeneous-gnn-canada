#!/usr/bin/env python3
"""
scripts/train_relational_gnn_biological_graph_strict.py

Trains and evaluates a biologically informed Multi-Relation Relational GNN (RGCNConv) for HPAI risk prediction:
- Relation 0: geographic_proximity (d <= 50 km)
- Relation 1: shared_water_habitat (d <= 100 km & water habitat similarity >= 0.70)
- Relation 2: poultry_exposure_similarity (d <= 100 km & poultry similarity >= 0.70)

Enforces strict 50-feature set (43 environment + 7 approved lag features).
Excludes eBird, GBIF, bird observations, 2025 labels, and future information.

Selection & Evaluation Protocol:
- Tuning Phase: Train 2022-01..2024-09, Val 2024-10..2024-12 across seeds [11, 22, 33]. Selects optimal configuration based strictly on mean validation PR-AUC.
- Final Evaluation Phase: Retrains selected model on all 2022-2024 data and evaluates on 2025 test set across seeds [11, 22, 33] and 3-seed ensemble.
- Benchmark Comparison: Compares against strict LR (0.0654), strict MLP (0.0601), border GraphSAGE (0.0570), and kNN-3 GraphSAGE (0.0669).

Outputs:
- results/baselines/relational_gnn_biological_graph_validation_metrics.csv
- results/baselines/relational_gnn_biological_graph_summary.csv
- results/baselines/relational_gnn_biological_graph_test_predictions.csv
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
from torch_geometric.nn import RGCNConv, SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

MODEL_TABLE_PATH = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
REL_GEO_PATH = 'data/processed/graph/csd_relation_geographic_proximity_v1.csv'
REL_WATER_PATH = 'data/processed/graph/csd_relation_shared_water_habitat_v1.csv'
REL_POULTRY_PATH = 'data/processed/graph/csd_relation_poultry_exposure_similarity_v1.csv'
KNN3_PATH = 'data/processed/graph/csd_graph_knn_3_v1.csv'

METRICS_VAL_OUT = 'results/baselines/relational_gnn_biological_graph_validation_metrics.csv'
SUMMARY_TEST_OUT = 'results/baselines/relational_gnn_biological_graph_summary.csv'
PREDS_TEST_OUT = 'results/baselines/relational_gnn_biological_graph_test_predictions.csv'

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

class RelationalGNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, num_relations=3):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        return self.lin(x).squeeze(-1)

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

def get_snapshot_predictions_relational(model, snapshots, edge_index, edge_type):
    model.eval()
    all_y_true = []
    all_y_scores = []
    
    with torch.no_grad():
        for data in snapshots:
            out = model(data.x, edge_index, edge_type)
            probs = torch.sigmoid(out).cpu().numpy()
            mask_np = data.mask.cpu().numpy()
            all_y_true.extend(data.y.cpu().numpy()[mask_np])
            all_y_scores.extend(probs[mask_np])
            
    return np.array(all_y_true), np.array(all_y_scores)

def get_snapshot_predictions_sage(model, snapshots, edge_index):
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

def build_multi_relation_edge_tensors(csd_to_idx):
    rel_paths = [REL_GEO_PATH, REL_WATER_PATH, REL_POULTRY_PATH]
    all_src, all_dst, all_types = [], [], []

    for r_idx, path in enumerate(rel_paths):
        df_r = pd.read_csv(path, dtype={'source_csd_uid': str, 'target_csd_uid': str})
        for _, row in df_r.iterrows():
            u, v = row['source_csd_uid'], row['target_csd_uid']
            if u in csd_to_idx and v in csd_to_idx:
                iu, iv = csd_to_idx[u], csd_to_idx[v]
                # Add undirected pair with relation type r_idx
                all_src.extend([iu, iv])
                all_dst.extend([iv, iu])
                all_types.extend([r_idx, r_idx])

    edge_index = torch.tensor(np.stack([all_src, all_dst], axis=0), dtype=torch.long)
    edge_type = torch.tensor(all_types, dtype=torch.long)
    return edge_index, edge_type

def build_sage_edge_index(csv_path, csd_to_idx):
    df_e = pd.read_csv(csv_path, dtype=str)
    src = df_e['source_csd_uid'].map(csd_to_idx).values
    dst = df_e['target_csd_uid'].map(csd_to_idx).values
    edge_index = torch.tensor(
        np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0),
        dtype=torch.long
    )
    return edge_index

def build_snapshots(df_sub, csd_to_idx, num_nodes, feature_cols):
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
        
        snapshots.append(Data(x=x_tensor, y=y_tensor, mask=mask_tensor))
    return snapshots

def main():
    print("==========================================================================================")
    print("  Biologically Informed Multi-Relation Relational GNN (RGCN) vs. Baseline Evaluation")
    print("==========================================================================================")

    df_model = pd.read_csv(MODEL_TABLE_PATH, dtype={'csd_uid': str})

    # Strict feature audit assertions
    feature_cols = [c for c in df_model.columns if c not in EXCLUDE_EXPLICIT_GNN and pd.api.types.is_numeric_dtype(df_model[c])]
    feature_cols = sorted(feature_cols)

    print("\n--- Executing Strict 50-Feature Audit Assertions ---")
    print(f"Num features: {len(feature_cols)} (Expected: 50)")
    assert len(feature_cols) == 50, f"ERROR: Feature count is {len(feature_cols)}, expected 50!"
    assert 'ccs_uid' not in feature_cols, "ERROR: ccs_uid present!"
    uid_feats = [c for c in feature_cols if 'uid' in c.lower()]
    assert len(uid_feats) == 0, f"ERROR: Features with 'uid': {uid_feats}"
    assert 'outbreak_current_month' not in feature_cols, "ERROR: raw outbreak_current_month present!"
    for lag in APPROVED_LAG_FEATURES:
        assert lag in feature_cols, f"ERROR: Missing lag {lag}"
    print("SUCCESS: 50-Feature list strictly validated.")

    model_csds = sorted(df_model['csd_uid'].unique())
    num_nodes = len(model_csds)
    csd_to_idx = {csd: i for i, csd in enumerate(model_csds)}

    # Build Edge Tensors
    rel_edge_index, rel_edge_type = build_multi_relation_edge_tensors(csd_to_idx)
    knn3_edge_index = build_sage_edge_index(KNN3_PATH, csd_to_idx)

    print(f"Multi-Relation Graph Total Edges: {rel_edge_index.shape[1] // 2:,} across 3 relations.")
    print(f"kNN-3 Comparison Graph Total Edges: {knn3_edge_index.shape[1] // 2:,}.")

    # Sort data chronologically
    df_model = df_model.sort_values(by=['year', 'month', 'csd_uid']).reset_index(drop=True)

    # Splits for Tuning Phase
    df_train_tune = df_model[
        df_model['year'].isin([2022, 2023]) | 
        ((df_model['year'] == 2024) & (df_model['month'] <= 9))
    ].copy()
    
    df_val_tune = df_model[
        ((df_model['year'] == 2024) & (df_model['month'] >= 10))
    ].copy()

    # Split for Final Evaluation Phase
    df_full_train = df_model[df_model['year'].isin([2022, 2023, 2024])].copy()
    df_test = df_model[df_model['year'] == 2025].copy()

    train_tune_pos = int(df_train_tune['outbreak_next_month'].sum())
    val_tune_pos = int(df_val_tune['outbreak_next_month'].sum())
    full_train_pos = int(df_full_train['outbreak_next_month'].sum())
    test_pos = int(df_test['outbreak_next_month'].sum())

    # Scalers
    scaler_tune = StandardScaler()
    df_train_tune[feature_cols] = scaler_tune.fit_transform(df_train_tune[feature_cols].fillna(0.0))
    df_val_tune[feature_cols] = scaler_tune.transform(df_val_tune[feature_cols].fillna(0.0))

    scaler_full = StandardScaler()
    df_full_train[feature_cols] = scaler_full.fit_transform(df_full_train[feature_cols].fillna(0.0))
    df_test[feature_cols] = scaler_full.transform(df_test[feature_cols].fillna(0.0))

    pos_weight_tune = torch.tensor([(len(df_train_tune) - train_tune_pos) / train_tune_pos], dtype=torch.float)
    pos_weight_full = torch.tensor([(len(df_full_train) - full_train_pos) / full_train_pos], dtype=torch.float)

    # Snapshots
    train_tune_snaps = build_snapshots(df_train_tune, csd_to_idx, num_nodes, feature_cols)
    val_tune_snaps = build_snapshots(df_val_tune, csd_to_idx, num_nodes, feature_cols)
    full_train_snaps = build_snapshots(df_full_train, csd_to_idx, num_nodes, feature_cols)
    test_snaps = build_snapshots(df_test, csd_to_idx, num_nodes, feature_cols)

    seeds = [11, 22, 33]
    max_epochs = 100
    patience = 10

    # ----------------------------------------------------------------------------------------
    # PHASE 1: Tuning Phase (Relational GNN vs kNN-3 GraphSAGE on Validation Split)
    # ----------------------------------------------------------------------------------------
    print("\n==========================================================================================")
    print("  PHASE 1: Tuning & Model Selection (Validation Split 2024-10..2024-12)")
    print("==========================================================================================")

    # 1A. Tune Relational GNN (RGCN)
    print("\n---> Training Relational GNN (RGCN) on 3 Biological Relations...")
    rgcn_val_aps = []
    rgcn_best_epochs = []

    for s in seeds:
        set_seed(s)
        model = RelationalGNNModel(in_channels=len(feature_cols), hidden_channels=32, num_relations=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tune)

        best_val_ap = -1.0
        best_epoch = 0
        no_improve = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            for data in train_tune_snaps:
                optimizer.zero_grad()
                out = model(data.x, rel_edge_index, rel_edge_type)
                loss = criterion(out[data.mask], data.y[data.mask])
                loss.backward()
                optimizer.step()

            val_true, val_scores = get_snapshot_predictions_relational(model, val_tune_snaps, rel_edge_index, rel_edge_type)
            val_ap = average_precision_score(val_true, val_scores)

            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        rgcn_val_aps.append(best_val_ap)
        rgcn_best_epochs.append(best_epoch)
        print(f"    Seed {s} | RGCN Best Val PR-AUC: {best_val_ap:.4f} (at Epoch {best_epoch})")

    rgcn_mean_val_ap = float(np.mean(rgcn_val_aps))
    rgcn_optimal_epochs = int(round(np.mean(rgcn_best_epochs)))

    # Save validation metrics
    val_records = [
        {
            'model_name': 'Biologically Informed Relational GNN (RGCN)',
            'val_pr_auc_seed_11': rgcn_val_aps[0],
            'val_pr_auc_seed_22': rgcn_val_aps[1],
            'val_pr_auc_seed_33': rgcn_val_aps[2],
            'mean_val_pr_auc': rgcn_mean_val_ap,
            'std_val_pr_auc': float(np.std(rgcn_val_aps)),
            'optimal_epochs': rgcn_optimal_epochs
        }
    ]
    df_val_res = pd.DataFrame(val_records)
    os.makedirs(os.path.dirname(METRICS_VAL_OUT), exist_ok=True)
    df_val_res.to_csv(METRICS_VAL_OUT, index=False)

    print(f"\nRelational GNN Mean Validation PR-AUC: {rgcn_mean_val_ap:.4f} (Optimal Epochs: {rgcn_optimal_epochs})")

    # ----------------------------------------------------------------------------------------
    # PHASE 2: Final Evaluation Phase (Retrain RGCN on 2022-2024, Test on 2025)
    # ----------------------------------------------------------------------------------------
    print("\n==========================================================================================")
    print("  PHASE 2: Retraining Relational GNN on All 2022-2024 Data & Testing on 2025")
    print("==========================================================================================")

    test_seed_results = []
    test_preds_dict = {
        'csd_uid': df_test['csd_uid'].values,
        'csd_name': df_test['csd_name'].values,
        'province_code': df_test['province_code'].values,
        'year': df_test['year'].values,
        'month': df_test['month'].values,
        'actual': df_test['outbreak_next_month'].values
    }

    for s in seeds:
        print(f"  Retraining Relational GNN with seed {s} for {rgcn_optimal_epochs} epochs...")
        set_seed(s)
        model = RelationalGNNModel(in_channels=len(feature_cols), hidden_channels=32, num_relations=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)

        for epoch in range(1, rgcn_optimal_epochs + 1):
            model.train()
            for data in full_train_snaps:
                optimizer.zero_grad()
                out = model(data.x, rel_edge_index, rel_edge_type)
                loss = criterion(out[data.mask], data.y[data.mask])
                loss.backward()
                optimizer.step()

        test_true, test_scores = get_snapshot_predictions_relational(model, test_snaps, rel_edge_index, rel_edge_type)
        ap = average_precision_score(test_true, test_scores)
        roc = roc_auc_score(test_true, test_scores)
        k_metrics = topk_metrics(test_true, test_scores, [50, 100, 200], test_pos)

        test_seed_results.append({
            'seed': s,
            'val_pr_auc': rgcn_mean_val_ap,
            'pr_auc': ap,
            'roc_auc': roc,
            'precision_50': k_metrics['precision_50'],
            'recall_50': k_metrics['recall_50'],
            'precision_100': k_metrics['precision_100'],
            'recall_100': k_metrics['recall_100'],
            'precision_200': k_metrics['precision_200'],
            'recall_200': k_metrics['recall_200']
        })
        test_preds_dict[f'rgcn_seed_{s}_prob'] = test_scores
        print(f"    Seed {s} | 2025 Test PR-AUC: {ap:.4f} | ROC-AUC: {roc:.4f} | Precision@50: {k_metrics['precision_50']:.4f}")

    # Compute 3-seed Ensemble
    ens_scores = np.mean([test_preds_dict[f'rgcn_seed_{s}_prob'] for s in seeds], axis=0)
    test_preds_dict['rgcn_ensemble_prob'] = ens_scores

    ens_ap = average_precision_score(df_test['outbreak_next_month'].values, ens_scores)
    ens_roc = roc_auc_score(df_test['outbreak_next_month'].values, ens_scores)
    ens_k_metrics = topk_metrics(df_test['outbreak_next_month'].values, ens_scores, [50, 100, 200], test_pos)

    df_test_seeds = pd.DataFrame(test_seed_results)
    mean_test = df_test_seeds.drop(columns=['seed']).mean()
    std_test = df_test_seeds.drop(columns=['seed']).std()

    df_summary = pd.DataFrame([
        {
            'model_variant': 'Relational GNN (RGCN Multiseed Mean)',
            'val_pr_auc': rgcn_mean_val_ap,
            'pr_auc': mean_test['pr_auc'],
            'pr_auc_std': std_test['pr_auc'],
            'roc_auc': mean_test['roc_auc'],
            'roc_auc_std': std_test['roc_auc'],
            'precision_50': mean_test['precision_50'],
            'recall_50': mean_test['recall_50'],
            'precision_100': mean_test['precision_100'],
            'recall_100': mean_test['recall_100'],
            'precision_200': mean_test['precision_200'],
            'recall_200': mean_test['recall_200']
        },
        {
            'model_variant': 'Relational GNN (RGCN 3-Seed Ensemble)',
            'val_pr_auc': rgcn_mean_val_ap,
            'pr_auc': ens_ap,
            'pr_auc_std': 0.0,
            'roc_auc': ens_roc,
            'roc_auc_std': 0.0,
            'precision_50': ens_k_metrics['precision_50'],
            'recall_50': ens_k_metrics['recall_50'],
            'precision_100': ens_k_metrics['precision_100'],
            'recall_100': ens_k_metrics['recall_100'],
            'precision_200': ens_k_metrics['precision_200'],
            'recall_200': ens_k_metrics['recall_200']
        }
    ])

    # Save summary and predictions
    os.makedirs(os.path.dirname(SUMMARY_TEST_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(PREDS_TEST_OUT), exist_ok=True)

    df_summary.to_csv(SUMMARY_TEST_OUT, index=False)
    pd.DataFrame(test_preds_dict).to_csv(PREDS_TEST_OUT, index=False)

    print("\n==========================================================================================")
    print("      FINAL BENCHMARK COMPARISON TABLE (2025 Held-Out Test Split)")
    print("==========================================================================================")
    print(df_summary[['model_variant', 'pr_auc', 'pr_auc_std', 'roc_auc', 'precision_50', 'recall_50', 'precision_100', 'recall_100']].to_string(index=False))

    print("\nBenchmark Context Comparison:")
    print("  - Strict Env + Lag Logistic Regression: PR-AUC = 0.0654")
    print("  - Strict MLP (No Graph):                PR-AUC = 0.0601")
    print("  - Border-Only Strict GraphSAGE:         PR-AUC = 0.0570")
    print("  - kNN-3 GraphSAGE Baseline:             PR-AUC = 0.0669")
    print(f"  - Relational GNN (RGCN Multiseed Mean):  PR-AUC = {mean_test['pr_auc']:.4f} +/- {std_test['pr_auc']:.4f}")
    print(f"  - Relational GNN (RGCN 3-Seed Ensemble): PR-AUC = {ens_ap:.4f}")
    print("==========================================================================================")
    print(f"Saved selected model summary to: {SUMMARY_TEST_OUT}")
    print(f"Saved test predictions to:       {PREDS_TEST_OUT}")

if __name__ == '__main__':
    main()
