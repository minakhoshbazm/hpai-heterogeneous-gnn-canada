#!/usr/bin/env python3
"""
scripts/train_graphsage_candidate_spatial_graphs_strict.py

Executes a controlled spatial graph-structure selection protocol for GraphSAGE:
1. Constructs and exports candidate spatial graph edge lists in EPSG:3347 metres:
   - border_only: existing shared-border adjacency graph
   - knn_3: symmetrized 3-nearest-neighbour centroid graph
   - border_plus_knn_5: union of shared-border adjacency and symmetrized 5-nearest-neighbour centroid edges
   Saves edge lists under data/processed/graph/.

2. Graph Selection Protocol (Tuning Phase):
   - Train split: 2022-01 through 2024-09
   - Val split:   2024-10 through 2024-12
   - Evaluates all candidate graphs across seeds [11, 22, 33] using strict 50 features.
   - Selects the single best graph topology using MEAN VALIDATION PR-AUC only.

3. Final Evaluation Phase:
   - Retrains ONLY the selected graph candidate on all 2022-2024 data.
   - Evaluates once on 2025 test split across seeds [11, 22, 33].

4. Exports results:
   - results/baselines/graphsage_candidate_graph_validation_metrics.csv
   - results/baselines/graphsage_selected_spatial_graph_summary.csv
   - results/baselines/graphsage_selected_spatial_graph_test_predictions.csv
"""

import os
import copy
import random
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

MODEL_TABLE_PATH = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
BOUNDARY_SHP_PATH = 'data/raw/boundaries/lcsd000b21a_e.shp'
BORDER_EDGES_PATH = 'data/processed/graph/csd_adjacency_edges_v1_model_scope.csv'

GRAPH_OUT_DIR = 'data/processed/graph'
METRICS_VAL_OUT = 'results/baselines/graphsage_candidate_graph_validation_metrics.csv'
SUMMARY_TEST_OUT = 'results/baselines/graphsage_selected_spatial_graph_summary.csv'
PREDS_TEST_OUT = 'results/baselines/graphsage_selected_spatial_graph_test_predictions.csv'

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

def construct_and_save_graphs(df_model):
    print("\n--- Constructing Candidate Graph Topologies (EPSG:3347 metres) ---")
    model_csds = sorted(df_model['csd_uid'].unique())
    num_model_csds = len(model_csds)
    csd_to_idx = {csd: i for i, csd in enumerate(model_csds)}

    # 1. Load boundary shapefile and reproject to EPSG:3347
    gdf = gpd.read_file(BOUNDARY_SHP_PATH)
    if "CSDUID" in gdf.columns:
        gdf["csd_uid"] = gdf["CSDUID"].astype(str)
    else:
        gdf["csd_uid"] = gdf["csd_uid"].astype(str)

    gdf_model = gdf[gdf["csd_uid"].isin(set(model_csds))].copy()
    if gdf_model.crs is None or gdf_model.crs.to_epsg() != 3347:
        gdf_model = gdf_model.to_crs(epsg=3347)

    gdf_model["centroid"] = gdf_model.geometry.centroid
    centroids_df = gdf_model.set_index("csd_uid").reindex(model_csds)
    coords = np.column_stack([centroids_df["centroid"].x.values, centroids_df["centroid"].y.values])

    # 2. Shared-border edges
    df_border = pd.read_csv(BORDER_EDGES_PATH, dtype=str)
    border_edges = set()
    for _, row in df_border.iterrows():
        u, v = row['source_csd_uid'], row['target_csd_uid']
        if u in csd_to_idx and v in csd_to_idx and u != v:
            iu, iv = csd_to_idx[u], csd_to_idx[v]
            border_edges.add((min(iu, iv), max(iu, iv)))

    # 3. KNN construction
    kdtree = KDTree(coords)
    
    def get_knn_edges(k_val):
        _, indices = kdtree.query(coords, k=k_val+1)
        knn_set = set()
        for i in range(num_model_csds):
            for n_idx in indices[i, 1:]:
                if i != n_idx:
                    knn_set.add((min(i, n_idx), max(i, n_idx)))
        return knn_set

    knn_3_edges = get_knn_edges(3)
    knn_5_edges = get_knn_edges(5)
    border_plus_knn_5_edges = border_edges.union(knn_5_edges)

    candidate_dict = {
        'border_only': border_edges,
        'knn_3': knn_3_edges,
        'border_plus_knn_5': border_plus_knn_5_edges
    }

    # Save edge list CSVs
    os.makedirs(GRAPH_OUT_DIR, exist_ok=True)
    graph_files = {}
    graph_topo_stats = {}

    for name, edges in candidate_dict.items():
        out_file = os.path.join(GRAPH_OUT_DIR, f"csd_graph_{name}_v1.csv")
        rows = [{'source_csd_uid': model_csds[u], 'target_csd_uid': model_csds[v]} for u, v in sorted(edges)]
        df_edges_out = pd.DataFrame(rows)
        df_edges_out.to_csv(out_file, index=False)
        graph_files[name] = out_file

        # Topo stats
        G = nx.Graph()
        G.add_nodes_from(range(num_model_csds))
        G.add_edges_from(edges)
        degs = np.array([d for _, d in G.degree()])

        graph_topo_stats[name] = {
            'total_edges': len(edges),
            'unique_nodes': int(np.sum(degs > 0)),
            'isolated_nodes': int(np.sum(degs == 0)),
            'connected_components': nx.number_connected_components(G),
            'mean_degree': round(float(np.mean(degs)), 2),
            'median_degree': round(float(np.median(degs)), 1),
            'min_degree': int(np.min(degs)),
            'max_degree': int(np.max(degs))
        }

        print(f"Saved {name}: {len(edges):,} edges to {out_file}")

    return graph_files, graph_topo_stats, csd_to_idx, num_model_csds

def build_pyg_edge_index(edges_csv_path, csd_to_idx):
    df_e = pd.read_csv(edges_csv_path, dtype=str)
    src = df_e['source_csd_uid'].map(csd_to_idx).values
    dst = df_e['target_csd_uid'].map(csd_to_idx).values
    edge_index = torch.tensor(
        np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0),
        dtype=torch.long
    )
    return edge_index

def build_snapshots(df_sub, csd_to_idx, num_nodes, feature_cols, edge_index):
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

def main():
    print("==========================================================================================")
    print("  Controlled Spatial Graph Structure Selection & Final Evaluation (GraphSAGE Strict)")
    print("==========================================================================================")

    df_model = pd.read_csv(MODEL_TABLE_PATH, dtype={'csd_uid': str})

    # Feature audit assertions
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

    # Construct candidate graphs
    graph_files, graph_topo_stats, csd_to_idx, num_nodes = construct_and_save_graphs(df_model)

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

    # Split for Final Retraining Phase
    df_full_train = df_model[df_model['year'].isin([2022, 2023, 2024])].copy()
    df_test = df_model[df_model['year'] == 2025].copy()

    train_tune_pos = int(df_train_tune['outbreak_next_month'].sum())
    val_tune_pos = int(df_val_tune['outbreak_next_month'].sum())
    full_train_pos = int(df_full_train['outbreak_next_month'].sum())
    test_pos = int(df_test['outbreak_next_month'].sum())

    # Fit scalers
    scaler_tune = StandardScaler()
    df_train_tune[feature_cols] = scaler_tune.fit_transform(df_train_tune[feature_cols].fillna(0.0))
    df_val_tune[feature_cols] = scaler_tune.transform(df_val_tune[feature_cols].fillna(0.0))

    scaler_full = StandardScaler()
    df_full_train[feature_cols] = scaler_full.fit_transform(df_full_train[feature_cols].fillna(0.0))
    df_test[feature_cols] = scaler_full.transform(df_test[feature_cols].fillna(0.0))

    pos_weight_tune = torch.tensor([(len(df_train_tune) - train_tune_pos) / train_tune_pos], dtype=torch.float)
    pos_weight_full = torch.tensor([(len(df_full_train) - full_train_pos) / full_train_pos], dtype=torch.float)

    seeds = [11, 22, 33]
    max_epochs = 100
    patience = 10

    # ----------------------------------------------------------------------------------------
    # PHASE 1: Controlled Tuning & Spatial Graph Selection (Mean Validation PR-AUC)
    # ----------------------------------------------------------------------------------------
    print("\n==========================================================================================")
    print("  PHASE 1: Controlled Spatial Graph Selection Tuning (Validation 2024-10..2024-12)")
    print("==========================================================================================")

    val_tuning_results = []
    best_epochs_tracker = {}

    for g_name, g_path in graph_files.items():
        print(f"\n---> Evaluating Candidate Graph: {g_name}")
        edge_index = build_pyg_edge_index(g_path, csd_to_idx)
        train_snaps = build_snapshots(df_train_tune, csd_to_idx, num_nodes, feature_cols, edge_index)
        val_snaps = build_snapshots(df_val_tune, csd_to_idx, num_nodes, feature_cols, edge_index)

        seed_val_aps = []
        seed_best_epochs = []

        for s in seeds:
            set_seed(s)
            model = GraphSAGEModel(in_channels=len(feature_cols), hidden_channels=32)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tune)

            best_val_ap = -1.0
            best_epoch = 0
            best_model_state = None
            no_improve = 0

            for epoch in range(1, max_epochs + 1):
                model.train()
                for data in train_snaps:
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index)
                    loss = criterion(out[data.mask], data.y[data.mask])
                    loss.backward()
                    optimizer.step()

                val_true, val_scores = get_snapshot_predictions(model, val_snaps, edge_index)
                val_ap = average_precision_score(val_true, val_scores)

                if val_ap > best_val_ap:
                    best_val_ap = val_ap
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    break

            seed_val_aps.append(best_val_ap)
            seed_best_epochs.append(best_epoch)
            print(f"    Seed {s} | Best Val PR-AUC: {best_val_ap:.4f} (at Epoch {best_epoch})")

        mean_val_ap = float(np.mean(seed_val_aps))
        std_val_ap = float(np.std(seed_val_aps))
        mean_epoch = int(round(np.mean(seed_best_epochs)))
        best_epochs_tracker[g_name] = mean_epoch

        topo = graph_topo_stats[g_name]
        val_tuning_results.append({
            'graph_name': g_name,
            'total_edges': topo['total_edges'],
            'mean_degree': topo['mean_degree'],
            'max_degree': topo['max_degree'],
            'isolated_nodes': topo['isolated_nodes'],
            'connected_components': topo['connected_components'],
            'val_pr_auc_seed_11': seed_val_aps[0],
            'val_pr_auc_seed_22': seed_val_aps[1],
            'val_pr_auc_seed_33': seed_val_aps[2],
            'mean_val_pr_auc': mean_val_ap,
            'std_val_pr_auc': std_val_ap,
            'mean_optimal_epochs': mean_epoch
        })

    df_val_tuning = pd.DataFrame(val_tuning_results).sort_values(by='mean_val_pr_auc', ascending=False).reset_index(drop=True)
    
    # Save validation tuning metrics
    os.makedirs(os.path.dirname(METRICS_VAL_OUT), exist_ok=True)
    df_val_tuning.to_csv(METRICS_VAL_OUT, index=False)
    print(f"\nSaved validation tuning comparison to: {METRICS_VAL_OUT}")

    # SELECT WINNING GRAPH BASED ON MEAN VALIDATION PR-AUC ONLY
    selected_graph = df_val_tuning.iloc[0]['graph_name']
    selected_val_ap = df_val_tuning.iloc[0]['mean_val_pr_auc']
    selected_optimal_epochs = best_epochs_tracker[selected_graph]

    print("\n==========================================================================================")
    print(f"  GRAPH SELECTION RESULT: Winner = '{selected_graph}' (Mean Val PR-AUC: {selected_val_ap:.4f})")
    print(f"  Selection Protocol: Selected strictly by mean validation PR-AUC across seeds [11, 22, 33].")
    print("==========================================================================================")

    # ----------------------------------------------------------------------------------------
    # PHASE 2: Final Evaluation Phase (Retrain Selected Graph on 2022-2024, Test on 2025)
    # ----------------------------------------------------------------------------------------
    print(f"\n--- PHASE 2: Retraining ONLY Selected Graph ('{selected_graph}') on all 2022-2024 Data ---")
    selected_edge_index = build_pyg_edge_index(graph_files[selected_graph], csd_to_idx)
    full_train_snaps = build_snapshots(df_full_train, csd_to_idx, num_nodes, feature_cols, selected_edge_index)
    test_snaps = build_snapshots(df_test, csd_to_idx, num_nodes, feature_cols, selected_edge_index)

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
        print(f"  Retraining '{selected_graph}' with seed {s} for {selected_optimal_epochs} epochs...")
        set_seed(s)
        model = GraphSAGEModel(in_channels=len(feature_cols), hidden_channels=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)

        for epoch in range(1, selected_optimal_epochs + 1):
            model.train()
            for data in full_train_snaps:
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = criterion(out[data.mask], data.y[data.mask])
                loss.backward()
                optimizer.step()

        test_true, test_scores = get_snapshot_predictions(model, test_snaps, selected_edge_index)
        ap = average_precision_score(test_true, test_scores)
        roc = roc_auc_score(test_true, test_scores)
        k_metrics = topk_metrics(test_true, test_scores, [50, 100, 200], test_pos)

        test_seed_results.append({
            'seed': s,
            'selected_graph': selected_graph,
            'val_pr_auc': selected_val_ap,
            'pr_auc': ap,
            'roc_auc': roc,
            'precision_50': k_metrics['precision_50'],
            'recall_50': k_metrics['recall_50'],
            'precision_100': k_metrics['precision_100'],
            'recall_100': k_metrics['recall_100'],
            'precision_200': k_metrics['precision_200'],
            'recall_200': k_metrics['recall_200']
        })
        test_preds_dict[f'graphsage_selected_{selected_graph}_seed_{s}_prob'] = test_scores
        print(f"    Seed {s} | 2025 Test PR-AUC: {ap:.4f} | ROC-AUC: {roc:.4f} | Precision@50: {k_metrics['precision_50']:.4f}")

    df_test_seeds = pd.DataFrame(test_seed_results)
    mean_test = df_test_seeds.drop(columns=['seed', 'selected_graph']).mean()
    std_test = df_test_seeds.drop(columns=['seed', 'selected_graph']).std()

    df_summary = pd.DataFrame({
        'Metric': list(mean_test.index),
        'Mean': list(mean_test.values),
        'Std': list(std_test.values)
    })

    # Save summary and predictions
    os.makedirs(os.path.dirname(SUMMARY_TEST_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(PREDS_TEST_OUT), exist_ok=True)

    df_summary.to_csv(SUMMARY_TEST_OUT, index=False)
    pd.DataFrame(test_preds_dict).to_csv(PREDS_TEST_OUT, index=False)

    print("\n==========================================================================================")
    print(f"  FINAL SUMMARY: Selected Spatial Graph ('{selected_graph}') 2025 Evaluation")
    print("==========================================================================================")
    print(df_summary.to_string(index=False))
    print("\nBenchmark Context Comparison (2025 Test Split):")
    print("  - Strict Env + Lag Logistic Regression: PR-AUC = 0.0654")
    print("  - Strict MLP (No Graph):                PR-AUC = 0.0601")
    print("  - Border-Only Strict GraphSAGE:         PR-AUC = 0.0570")
    print(f"  - Selected Spatial GraphSAGE ({selected_graph}): PR-AUC = {mean_test['pr_auc']:.4f} +/- {std_test['pr_auc']:.4f}")
    print("==========================================================================================")
    print(f"Saved selected graph summary to: {SUMMARY_TEST_OUT}")
    print(f"Saved test predictions to: {PREDS_TEST_OUT}")

if __name__ == '__main__':
    main()
