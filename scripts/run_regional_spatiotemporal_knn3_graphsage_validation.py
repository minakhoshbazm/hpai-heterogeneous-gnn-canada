#!/usr/bin/env python3
"""
scripts/run_regional_spatiotemporal_knn3_graphsage_validation.py

Runs strict leave-one-region-out spatial-temporal validation for the selected kNN-3 GraphSAGE model across 4 Canadian regional blocks:
1. Atlantic: NL (10), PEI (11), NS (12), NB (13)
2. Quebec_Ontario: QC (24), ON (35)
3. Prairies: MB (46), SK (47), AB (48)
4. British_Columbia: BC (59)

For each held-out block:
- Train split: 2022-01 to 2024-09 outside the held-out block.
- Val split:   2024-10 to 2024-12 outside the held-out block.
- Test split:  2025 inside the held-out block.
- kNN-3 spatial graphs are constructed separately within training nodes and within test nodes in EPSG:3347 metres. Zero cross-region edges are permitted.
- Evaluates seeds [11, 22, 33] with identical architecture, scaling, class weighting, and early stopping.

Outputs:
- results/baselines/regional_spatiotemporal_knn3_graphsage_multiseed_metrics.csv
- results/baselines/regional_spatiotemporal_knn3_graphsage_summary.csv
- results/baselines/regional_spatiotemporal_knn3_graphsage_predictions.csv
"""

import os
import copy
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

MODEL_TABLE_PATH = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
BOUNDARY_SHP_PATH = 'data/raw/boundaries/lcsd000b21a_e.shp'
LR_REGIONAL_METRICS_PATH = 'results/baselines/regional_spatiotemporal_environment_lag_lr_strict_metrics.csv'

METRICS_SEED_OUT = 'results/baselines/regional_spatiotemporal_knn3_graphsage_multiseed_metrics.csv'
SUMMARY_OUT = 'results/baselines/regional_spatiotemporal_knn3_graphsage_summary.csv'
PREDS_OUT = 'results/baselines/regional_spatiotemporal_knn3_graphsage_predictions.csv'

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

def build_knn3_edge_index(csd_list, centroids_df):
    """Builds a symmetrized kNN-3 edge_index for a specific subset of CSDs."""
    num_csds = len(csd_list)
    csd_to_idx = {csd: i for i, csd in enumerate(csd_list)}
    
    # Subselect centroids
    sub_centroids = centroids_df.reindex(csd_list)
    coords = np.column_stack([sub_centroids["cx"].values, sub_centroids["cy"].values])

    # KDTree query k=3 (excluding self)
    kdtree = KDTree(coords)
    k_val = min(3, num_csds - 1)
    
    if k_val < 1:
        # Fallback for empty/single node graph
        return torch.zeros((2, 0), dtype=torch.long)

    distances, indices = kdtree.query(coords, k=k_val+1)
    
    edge_src, edge_dst = [], []
    for i in range(num_csds):
        for neighbor_idx in indices[i, 1:]:
            if i != neighbor_idx:
                edge_src.append(i)
                edge_dst.append(neighbor_idx)
                # Ensure symmetry
                edge_src.append(neighbor_idx)
                edge_dst.append(i)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    return edge_index, csd_to_idx

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
    print("   Strict Regional Spatial-Temporal Validation for Selected kNN-3 GraphSAGE Model")
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

    # 1. Load CSD centroids in EPSG:3347
    print("\nLoading shapefile and computing CSD centroids in EPSG:3347...")
    gdf = gpd.read_file(BOUNDARY_SHP_PATH)
    if "CSDUID" in gdf.columns:
        gdf["csd_uid"] = gdf["CSDUID"].astype(str)
    else:
        gdf["csd_uid"] = gdf["csd_uid"].astype(str)

    if gdf.crs is None or gdf.crs.to_epsg() != 3347:
        gdf = gdf.to_crs(epsg=3347)

    gdf["centroid"] = gdf.geometry.centroid
    centroids_df = pd.DataFrame({
        "csd_uid": gdf["csd_uid"],
        "cx": gdf["centroid"].x,
        "cy": gdf["centroid"].y
    }).set_index("csd_uid")

    # Chronological sort
    df_model = df_model.sort_values(by=['year', 'month', 'csd_uid']).reset_index(drop=True)

    seeds = [11, 22, 33]
    max_epochs = 100
    patience = 10

    all_seed_results = []
    all_summary_results = []
    all_predictions = []

    for block_name, prov_codes in REGIONAL_BLOCKS.items():
        print(f"\n==========================================================================================")
        print(f"Processing Regional Block: {block_name} (Provinces: {prov_codes})")
        print(f"==========================================================================================")

        # Splitting data
        df_train_sub = df_model[
            (df_model['year'].isin([2022, 2023]) | ((df_model['year'] == 2024) & (df_model['month'] <= 9))) &
            (~df_model['province_code'].isin(prov_codes))
        ].copy()

        df_val_sub = df_model[
            ((df_model['year'] == 2024) & (df_model['month'] >= 10)) &
            (~df_model['province_code'].isin(prov_codes))
        ].copy()

        df_test_sub = df_model[
            (df_model['year'] == 2025) &
            (df_model['province_code'].isin(prov_codes))
        ].copy()

        train_rows, val_rows, test_rows = len(df_train_sub), len(df_val_sub), len(df_test_sub)
        train_pos = int(df_train_sub['outbreak_next_month'].sum())
        val_pos = int(df_val_sub['outbreak_next_month'].sum())
        test_pos = int(df_test_sub['outbreak_next_month'].sum())
        test_prev = test_pos / test_rows if test_rows > 0 else 0.0

        small_sample_warning = ""
        if test_pos < 10:
            small_sample_warning = f"WARNING: Test positive count ({test_pos}) is very small (< 10); metrics may be unstable."

        print(f"  Train split (outside {block_name}): {train_rows:,} rows, {train_pos} positives")
        print(f"  Val split   (outside {block_name}): {val_rows:,} rows, {val_pos} positives")
        print(f"  Test split  (INSIDE  {block_name}): {test_rows:,} rows, {test_pos} positives (Prevalence: {test_prev:.6f})")
        if small_sample_warning:
            print(f"  {small_sample_warning}")

        # Scaler fit on Train split only
        scaler = StandardScaler()
        df_train_sub[feature_cols] = scaler.fit_transform(df_train_sub[feature_cols].fillna(0.0))
        df_val_sub[feature_cols] = scaler.transform(df_val_sub[feature_cols].fillna(0.0))
        df_test_sub[feature_cols] = scaler.transform(df_test_sub[feature_cols].fillna(0.0))

        pos_weight = torch.tensor([(train_rows - train_pos) / train_pos], dtype=torch.float)

        # Build ISOLATED kNN-3 Graphs
        train_csds = sorted(df_model[~df_model['province_code'].isin(prov_codes)]['csd_uid'].unique())
        test_csds = sorted(df_model[df_model['province_code'].isin(prov_codes)]['csd_uid'].unique())

        train_edge_index, train_csd_map = build_knn3_edge_index(train_csds, centroids_df)
        test_edge_index, test_csd_map = build_knn3_edge_index(test_csds, centroids_df)

        print(f"  Train CSDs: {len(train_csds):,}, Train Graph Edges: {train_edge_index.shape[1] // 2:,}")
        print(f"  Test CSDs:  {len(test_csds):,}, Test Graph Edges:  {test_edge_index.shape[1] // 2:,}")

        train_snaps = build_snapshots(df_train_sub, train_csd_map, len(train_csds), feature_cols, train_edge_index)
        val_snaps = build_snapshots(df_val_sub, train_csd_map, len(train_csds), feature_cols, train_edge_index)
        test_snaps = build_snapshots(df_test_sub, test_csd_map, len(test_csds), feature_cols, test_edge_index)

        block_seed_results = []
        block_preds_dict = {
            'regional_block': block_name,
            'csd_uid': df_test_sub['csd_uid'].values,
            'csd_name': df_test_sub['csd_name'].values,
            'province_code': df_test_sub['province_code'].values,
            'year': df_test_sub['year'].values,
            'month': df_test_sub['month'].values,
            'actual': df_test_sub['outbreak_next_month'].values
        }

        for s in seeds:
            set_seed(s)
            model = GraphSAGEModel(in_channels=len(feature_cols), hidden_channels=32)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

                val_true, val_scores = get_snapshot_predictions(model, val_snaps, train_edge_index)
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

            # Evaluate best model state on held-out test block
            model.load_state_dict(best_model_state)
            test_true, test_scores = get_snapshot_predictions(model, test_snaps, test_edge_index)

            ap = average_precision_score(test_true, test_scores)
            roc = roc_auc_score(test_true, test_scores) if len(np.unique(test_true)) > 1 else 0.5
            pr_lift = ap / test_prev if test_prev > 0 else 0.0
            k_metrics = topk_metrics(test_true, test_scores, [50, 100, 200], test_pos)

            seed_rec = {
                'regional_block': block_name,
                'seed': s,
                'train_rows': train_rows,
                'train_positives': train_pos,
                'test_rows': test_rows,
                'test_positives': test_pos,
                'test_prevalence': test_prev,
                'val_pr_auc': best_val_ap,
                'best_epoch': best_epoch,
                'pr_auc': ap,
                'pr_auc_lift': pr_lift,
                'roc_auc': roc,
                'precision_50': k_metrics['precision_50'],
                'recall_50': k_metrics['recall_50'],
                'precision_100': k_metrics['precision_100'],
                'recall_100': k_metrics['recall_100'],
                'precision_200': k_metrics['precision_200'],
                'recall_200': k_metrics['recall_200'],
                'small_sample_warning': small_sample_warning
            }
            all_seed_results.append(seed_rec)
            block_seed_results.append(seed_rec)
            block_preds_dict[f'knn3_graphsage_seed_{s}_prob'] = test_scores

            print(f"    Seed {s} | Val PR-AUC: {best_val_ap:.4f} (Ep {best_epoch}) | Test PR-AUC: {ap:.4f} | ROC-AUC: {roc:.4f} | Lift: {pr_lift:.2f}x")

        # Save predictions for block
        df_block_preds = pd.DataFrame(block_preds_dict)
        all_predictions.append(df_block_preds)

        # Block summary across seeds
        df_bs = pd.DataFrame(block_seed_results)
        summary_rec = {
            'regional_block': block_name,
            'province_codes': str(prov_codes),
            'train_rows': train_rows,
            'train_positives': train_pos,
            'test_rows': test_rows,
            'test_positives': test_pos,
            'test_prevalence': test_prev,
            'pr_auc_mean': df_bs['pr_auc'].mean(),
            'pr_auc_std': df_bs['pr_auc'].std(),
            'pr_auc_lift_mean': df_bs['pr_auc_lift'].mean(),
            'roc_auc_mean': df_bs['roc_auc'].mean(),
            'roc_auc_std': df_bs['roc_auc'].std(),
            'precision_50_mean': df_bs['precision_50'].mean(),
            'recall_50_mean': df_bs['recall_50'].mean(),
            'precision_100_mean': df_bs['precision_100'].mean(),
            'recall_100_mean': df_bs['recall_100'].mean(),
            'precision_200_mean': df_bs['precision_200'].mean(),
            'recall_200_mean': df_bs['recall_200'].mean(),
            'small_sample_warning': small_sample_warning
        }
        all_summary_results.append(summary_rec)

    # Save CSV outputs
    df_seed_metrics = pd.DataFrame(all_seed_results)
    df_summary = pd.DataFrame(all_summary_results)
    df_preds_all = pd.concat(all_predictions, ignore_index=True)

    os.makedirs(os.path.dirname(METRICS_SEED_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(PREDS_OUT), exist_ok=True)

    df_seed_metrics.to_csv(METRICS_SEED_OUT, index=False)
    df_summary.to_csv(SUMMARY_OUT, index=False)
    df_preds_all.to_csv(PREDS_OUT, index=False)

    # Load Strict LR regional metrics for comparison
    print("\n==========================================================================================")
    print("      STRICT REGIONAL SPATIAL-TEMPORAL EVALUATION SUMMARY & COMPARISON")
    print("==========================================================================================")
    print("\nkNN-3 GraphSAGE Regional Results (Mean across seeds [11, 22, 33]):")
    print(df_summary[['regional_block', 'test_positives', 'test_prevalence', 'pr_auc_mean', 'pr_auc_std', 'pr_auc_lift_mean', 'roc_auc_mean', 'precision_50_mean', 'recall_50_mean']].to_string(index=False))

    if os.path.exists(LR_REGIONAL_METRICS_PATH):
        df_lr_reg = pd.read_csv(LR_REGIONAL_METRICS_PATH)
        print("\n--- Comparison Against Strict Environment + Lag Logistic Regression ---")
        merged_comp = pd.merge(
            df_summary[['regional_block', 'test_positives', 'pr_auc_mean', 'roc_auc_mean']],
            df_lr_reg[['regional_block', 'pr_auc', 'roc_auc']].rename(columns={'pr_auc': 'lr_pr_auc', 'roc_auc': 'lr_roc_auc'}),
            on='regional_block'
        )
        merged_comp['pr_auc_diff'] = merged_comp['pr_auc_mean'] - merged_comp['lr_pr_auc']
        print(merged_comp[['regional_block', 'test_positives', 'lr_pr_auc', 'pr_auc_mean', 'pr_auc_diff', 'lr_roc_auc', 'roc_auc_mean']].to_string(index=False))

    print(f"\nSaved multiseed metrics to: {METRICS_SEED_OUT}")
    print(f"Saved summary to:           {SUMMARY_OUT}")
    print(f"Saved predictions to:       {PREDS_OUT}")

if __name__ == '__main__':
    main()
