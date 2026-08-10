#!/usr/bin/env python3
"""
scripts/run_regional_spatiotemporal_relational_gnn_validation.py

Part 3: Strict Regional Spatial-Temporal Validation for Relational GNN.

Evaluates the fixed, time-safe Relational GNN across 4 Canadian regional blocks:
1. Atlantic: NL (10), PEI (11), NS (12), NB (13)
2. Quebec_Ontario: QC (24), ON (35)
3. Prairies: MB (46), SK (47), AB (48)
4. British_Columbia: BC (59)

For each held-out regional block:
- train: 2022-01 through 2024-09 outside the held-out block
- validation: 2024-10 through 2024-12 outside the held-out block
- test: 2025 inside the held-out block
- no cross-region edges permitted in any relation
- construct month-specific snapshot graph relations independently within train/val nodes and within test nodes using expanding historical observations up to month M
- fixed architecture & hyperparameters (seeds [11, 22, 33])

Compares side-by-side with strict regional LR and kNN-3 GraphSAGE results.
Explicitly marks Atlantic as unreliable if test positive count is 5.

Outputs:
- results/baselines/regional_spatiotemporal_relational_gnn_multiseed_metrics.csv
- results/baselines/regional_spatiotemporal_relational_gnn_summary.csv
- results/baselines/regional_spatiotemporal_relational_gnn_predictions.csv
"""

import os
import math
import copy
import random
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv

MODEL_TABLE_PATH = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
CENTROIDS_CSV_PATH = 'data/processed/graph/csd_centroids_epsg3347.csv'

LR_REGIONAL_METRICS_PATH = 'results/baselines/regional_spatiotemporal_environment_lag_lr_strict_metrics.csv'
KNN3_REGIONAL_SUMMARY_PATH = 'results/baselines/regional_spatiotemporal_knn3_graphsage_summary.csv'

METRICS_SEED_OUT = 'results/baselines/regional_spatiotemporal_relational_gnn_multiseed_metrics.csv'
SUMMARY_OUT = 'results/baselines/regional_spatiotemporal_relational_gnn_summary.csv'
PREDS_OUT = 'results/baselines/regional_spatiotemporal_relational_gnn_predictions.csv'

REGIONAL_BLOCKS = {
    'Atlantic': [10, 11, 12, 13],
    'Quebec_Ontario': [24, 35],
    'Prairies': [46, 47, 48],
    'British_Columbia': [59]
}

WATER_HABITAT_COLS = [
    'dw_water_mean',
    'dw_water_p95',
    'dw_flooded_vegetation_mean',
    'dw_seasonal_pond_proxy'
]

POULTRY_EXPOSURE_COLS = [
    'ccs_broilers_roasters_cornish_farms_reporting',
    'ccs_broilers_roasters_cornish_production_kg',
    'ccs_turkey_farms_reporting',
    'ccs_turkey_production_kg',
    'glw2020_chicken_density_mean',
    'glw2020_chicken_density_max',
    'glw2015_duck_density_mean',
    'glw2015_duck_density_max'
]

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

def get_snapshot_predictions_relational(model, snapshots):
    model.eval()
    all_y_true = []
    all_y_scores = []
    
    with torch.no_grad():
        for data in snapshots:
            out = model(data.x, data.edge_index, data.edge_type)
            probs = torch.sigmoid(out).cpu().numpy()
            mask_np = data.mask.cpu().numpy()
            all_y_true.extend(data.y.cpu().numpy()[mask_np])
            all_y_scores.extend(probs[mask_np])
            del out, probs
            
    return np.array(all_y_true), np.array(all_y_scores)

def build_fixed_subset_geo_and_poultry_edges(csd_list, df_model, centroids_df):
    num_csds = len(csd_list)
    if num_csds <= 1:
        return {}, np.zeros((0, 2)), [], ([], []), ([], [])

    csd_to_idx = {csd: i for i, csd in enumerate(csd_list)}
    sub_centroids = centroids_df.reindex(csd_list)
    coords = np.column_stack([sub_centroids["cx"].values, sub_centroids["cy"].values])
    kdtree = KDTree(coords)
    
    # Relation 0: geo <= 50km
    pairs_50km = sorted(list(kdtree.query_pairs(50000.0)))
    geo_src, geo_dst = [], []
    for u, v in pairs_50km:
        geo_src.extend([u, v])
        geo_dst.extend([v, u])
        
    # Relation 2: poultry <= 100km & sim >= 0.70
    pairs_100km = list(kdtree.query_pairs(100000.0))
    poultry_src, poultry_dst = [], []
    
    if len(pairs_100km) > 0:
        csd_poultry_df = df_model.groupby('csd_uid')[POULTRY_EXPOSURE_COLS].first().reindex(csd_list)
        poultry_raw = np.log1p(np.maximum(0.0, csd_poultry_df.fillna(0.0).values))
        scaler_p = StandardScaler()
        p_scaled = scaler_p.fit_transform(poultry_raw)
        p_norms = np.linalg.norm(p_scaled, axis=1, keepdims=True)
        p_norms[p_norms == 0] = 1e-8
        p_unit = p_scaled / p_norms
        
        tau_poultry = 0.70
        for u, v in pairs_100km:
            p_sim = float(np.dot(p_unit[u], p_unit[v]))
            if p_sim >= tau_poultry:
                poultry_src.extend([u, v])
                poultry_dst.extend([v, u])
                
    return csd_to_idx, coords, pairs_100km, (geo_src, geo_dst), (poultry_src, poultry_dst)

def precompute_subset_water_unit_vecs(df_model, csd_list):
    df_sub = df_model[df_model['csd_uid'].isin(set(csd_list))].copy()
    df_sorted = df_sub.sort_values(by=['csd_uid', 'year', 'month']).copy()
    df_sorted[WATER_HABITAT_COLS] = df_sorted.groupby('csd_uid')[WATER_HABITAT_COLS].expanding().mean().reset_index(level=0, drop=True)
    
    water_unit_dict = {}
    for (y, m), group in df_sorted.groupby(['year', 'month']):
        c_df = group.set_index('csd_uid')[WATER_HABITAT_COLS].reindex(csd_list)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(c_df.fillna(0.0).values)
        norms = np.linalg.norm(scaled, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        water_unit_dict[(int(y), int(m))] = scaled / norms
        
    return water_unit_dict

def build_snapshots_time_safe_subset(df_sub, water_unit_dict, csd_list, csd_to_idx, pairs_100km, geo_pair, poultry_pair, feature_cols):
    geo_src, geo_dst = geo_pair
    poultry_src, poultry_dst = poultry_pair
    num_nodes = len(csd_list)
    
    geo_src_arr = np.array(geo_src, dtype=np.int64)
    geo_dst_arr = np.array(geo_dst, dtype=np.int64)
    p_src_arr = np.array(poultry_src, dtype=np.int64)
    p_dst_arr = np.array(poultry_dst, dtype=np.int64)
    
    if len(pairs_100km) > 0:
        u_pairs = np.array([p[0] for p in pairs_100km], dtype=np.int64)
        v_pairs = np.array([p[1] for p in pairs_100km], dtype=np.int64)
    else:
        u_pairs = np.array([], dtype=np.int64)
        v_pairs = np.array([], dtype=np.int64)
        
    snapshots = []
    ym_tuples = df_sub[['year', 'month']].drop_duplicates().sort_values(['year', 'month']).values
    tau_water = 0.70
    
    for y, m in ym_tuples:
        y, m = int(y), int(m)
        group = df_sub[(df_sub['year'] == y) & (df_sub['month'] == m)]
        x_npy = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)
        y_npy = np.full((num_nodes,), -1.0, dtype=np.float32)
        
        node_indices = group['csd_uid'].map(csd_to_idx).values
        x_npy[node_indices] = group[feature_cols].values
        y_npy[node_indices] = group['outbreak_next_month'].values.astype(np.float32)
        
        x_tensor = torch.tensor(x_npy, dtype=torch.float)
        y_tensor = torch.tensor(y_npy, dtype=torch.float)
        mask_tensor = (y_tensor >= 0.0)
        
        if len(pairs_100km) > 0:
            w_unit = water_unit_dict[(y, m)]
            sims = np.sum(w_unit[u_pairs] * w_unit[v_pairs], axis=1)
            valid = (sims >= tau_water)
            valid_u = u_pairs[valid]
            valid_v = v_pairs[valid]
            w_src_arr = np.concatenate([valid_u, valid_v])
            w_dst_arr = np.concatenate([valid_v, valid_u])
        else:
            w_src_arr = np.array([], dtype=np.int64)
            w_dst_arr = np.array([], dtype=np.int64)
            
        all_src = np.concatenate([geo_src_arr, w_src_arr, p_src_arr])
        all_dst = np.concatenate([geo_dst_arr, w_dst_arr, p_dst_arr])
        all_types = np.concatenate([
            np.zeros(len(geo_src_arr), dtype=np.int64),
            np.ones(len(w_src_arr), dtype=np.int64),
            np.full(len(p_src_arr), 2, dtype=np.int64)
        ])
        
        if len(all_src) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros((0,), dtype=torch.long)
        else:
            edge_index = torch.tensor(np.stack([all_src, all_dst], axis=0), dtype=torch.long)
            edge_type = torch.tensor(all_types, dtype=torch.long)
            
        snapshots.append(Data(x=x_tensor, y=y_tensor, mask=mask_tensor, edge_index=edge_index, edge_type=edge_type))
        
    return snapshots

def main():
    print("==========================================================================================")
    print("      PART 3: STRICT REGIONAL SPATIAL-TEMPORAL RELATIONAL GNN VALIDATION")
    print("==========================================================================================")

    df_model = pd.read_csv(MODEL_TABLE_PATH, dtype={'csd_uid': str})
    feature_cols = [c for c in df_model.columns if c not in EXCLUDE_EXPLICIT_GNN and pd.api.types.is_numeric_dtype(df_model[c])]
    feature_cols = sorted(feature_cols)
    assert len(feature_cols) == 50, f"Expected 50 features, got {len(feature_cols)}"

    # Load centroids from lightweight CSV
    df_centroids = pd.read_csv(CENTROIDS_CSV_PATH, dtype={'csd_uid': str})
    model_csds = set(df_model['csd_uid'].unique())
    centroids_df = df_centroids[df_centroids['csd_uid'].isin(model_csds)].set_index('csd_uid')

    df_model = df_model.sort_values(by=['year', 'month', 'csd_uid']).reset_index(drop=True)

    seeds = [11, 22, 33]
    max_epochs = 100
    patience = 10

    seed_records = []
    summary_records = []
    all_pred_dfs = []

    for block_name, prov_codes in REGIONAL_BLOCKS.items():
        print(f"\n==========================================================================================")
        print(f"  Processing Regional Block: {block_name} (Provinces: {prov_codes})")
        print(f"==========================================================================================")

        test_csd_set = set(df_model[df_model['province_code'].isin(prov_codes)]['csd_uid'].unique())
        train_csd_set = set(df_model[~df_model['province_code'].isin(prov_codes)]['csd_uid'].unique())

        train_csd_list = sorted(train_csd_set)
        test_csd_list = sorted(test_csd_set)

        num_train_csds = len(train_csd_list)
        num_test_csds = len(test_csd_list)

        print(f"Train/Val CSD count (outside {block_name}): {num_train_csds:,}")
        print(f"Test CSD count (inside {block_name}):      {num_test_csds:,}")

        # Precompute vectorised water features for train and test subsets
        train_water_dict = precompute_subset_water_unit_vecs(df_model, train_csd_list)
        test_water_dict = precompute_subset_water_unit_vecs(df_model, test_csd_list)

        train_csd_to_idx, train_coords, train_pairs_100km, train_geo_pair, train_poultry_pair = build_fixed_subset_geo_and_poultry_edges(
            train_csd_list, df_model, centroids_df
        )
        test_csd_to_idx, test_coords, test_pairs_100km, test_geo_pair, test_poultry_pair = build_fixed_subset_geo_and_poultry_edges(
            test_csd_list, df_model, centroids_df
        )

        df_train_tune = df_model[
            (~df_model['province_code'].isin(prov_codes)) &
            ((df_model['year'].isin([2022, 2023])) | ((df_model['year'] == 2024) & (df_model['month'] <= 9)))
        ].copy()

        df_val_tune = df_model[
            (~df_model['province_code'].isin(prov_codes)) &
            ((df_model['year'] == 2024) & (df_model['month'] >= 10))
        ].copy()

        df_full_train = df_model[
            (~df_model['province_code'].isin(prov_codes)) &
            (df_model['year'].isin([2022, 2023, 2024]))
        ].copy()

        df_test = df_model[
            (df_model['province_code'].isin(prov_codes)) &
            (df_model['year'] == 2025)
        ].copy()

        train_tune_pos = int(df_train_tune['outbreak_next_month'].sum())
        val_tune_pos = int(df_val_tune['outbreak_next_month'].sum())
        full_train_pos = int(df_full_train['outbreak_next_month'].sum())
        test_pos = int(df_test['outbreak_next_month'].sum())

        print(f"Train pos: {train_tune_pos} | Val pos: {val_tune_pos} | Full Train pos: {full_train_pos} | Test pos: {test_pos}")

        scaler_tune = StandardScaler()
        df_train_tune[feature_cols] = scaler_tune.fit_transform(df_train_tune[feature_cols].fillna(0.0))
        df_val_tune[feature_cols] = scaler_tune.transform(df_val_tune[feature_cols].fillna(0.0))

        scaler_full = StandardScaler()
        df_full_train[feature_cols] = scaler_full.fit_transform(df_full_train[feature_cols].fillna(0.0))
        df_test[feature_cols] = scaler_full.transform(df_test[feature_cols].fillna(0.0))

        pos_weight_tune = torch.tensor([(len(df_train_tune) - train_tune_pos) / train_tune_pos], dtype=torch.float)
        pos_weight_full = torch.tensor([(len(df_full_train) - full_train_pos) / full_train_pos], dtype=torch.float)

        train_tune_snaps = build_snapshots_time_safe_subset(df_train_tune, train_water_dict, train_csd_list, train_csd_to_idx, train_pairs_100km, train_geo_pair, train_poultry_pair, feature_cols)
        val_tune_snaps = build_snapshots_time_safe_subset(df_val_tune, train_water_dict, train_csd_list, train_csd_to_idx, train_pairs_100km, train_geo_pair, train_poultry_pair, feature_cols)
        full_train_snaps = build_snapshots_time_safe_subset(df_full_train, train_water_dict, train_csd_list, train_csd_to_idx, train_pairs_100km, train_geo_pair, train_poultry_pair, feature_cols)
        test_snaps = build_snapshots_time_safe_subset(df_test, test_water_dict, test_csd_list, test_csd_to_idx, test_pairs_100km, test_geo_pair, test_poultry_pair, feature_cols)

        val_aps = []
        best_epochs = []
        for s in seeds:
            set_seed(s)
            model = RelationalGNNModel(in_channels=len(feature_cols), hidden_channels=32, num_relations=3)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tune)

            best_val_ap = -1.0
            best_ep = 0
            no_improve = 0

            for epoch in range(1, max_epochs + 1):
                model.train()
                for data in train_tune_snaps:
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index, data.edge_type)
                    loss = criterion(out[data.mask], data.y[data.mask])
                    loss.backward()
                    optimizer.step()

                val_true, val_scores = get_snapshot_predictions_relational(model, val_tune_snaps)
                val_ap = average_precision_score(val_true, val_scores)

                if val_ap > best_val_ap:
                    best_val_ap = val_ap
                    best_ep = epoch
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    break

            val_aps.append(best_val_ap)
            best_epochs.append(best_ep)

        mean_val_ap = float(np.mean(val_aps))
        optimal_epochs = int(round(np.mean(best_epochs)))
        print(f"Optimal Epochs for {block_name}: {optimal_epochs} (Mean Val PR-AUC: {mean_val_ap:.4f})")

        block_seed_res = []
        block_preds_dict = {
            'regional_block': block_name,
            'csd_uid': df_test['csd_uid'].values,
            'csd_name': df_test['csd_name'].values,
            'province_code': df_test['province_code'].values,
            'year': df_test['year'].values,
            'month': df_test['month'].values,
            'actual': df_test['outbreak_next_month'].values
        }

        for s in seeds:
            set_seed(s)
            model = RelationalGNNModel(in_channels=len(feature_cols), hidden_channels=32, num_relations=3)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)

            for epoch in range(1, optimal_epochs + 1):
                model.train()
                for data in full_train_snaps:
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index, data.edge_type)
                    loss = criterion(out[data.mask], data.y[data.mask])
                    loss.backward()
                    optimizer.step()

            test_true, test_scores = get_snapshot_predictions_relational(model, test_snaps)
            ap = average_precision_score(test_true, test_scores)
            roc = roc_auc_score(test_true, test_scores)
            k_metrics = topk_metrics(test_true, test_scores, [50], test_pos)

            rec = {
                'regional_block': block_name,
                'province_codes': str(prov_codes),
                'seed': s,
                'train_rows': len(df_full_train),
                'train_positives': full_train_pos,
                'test_rows': len(df_test),
                'test_positives': test_pos,
                'test_prevalence': round(test_pos / len(df_test), 6),
                'val_pr_auc': mean_val_ap,
                'pr_auc': ap,
                'pr_auc_lift': round(ap / (test_pos / len(df_test)), 2) if test_pos > 0 else 0.0,
                'roc_auc': roc,
                'precision_50': k_metrics['precision_50'],
                'recall_50': k_metrics['recall_50']
            }
            seed_records.append(rec)
            block_seed_res.append(rec)
            block_preds_dict[f'rgcn_seed_{s}_prob'] = test_scores
            print(f"  Seed {s} | Test PR-AUC: {ap:.4f} | ROC-AUC: {roc:.4f} | Precision@50: {k_metrics['precision_50']:.4f}")

        ens_scores = np.mean([block_preds_dict[f'rgcn_seed_{s}_prob'] for s in seeds], axis=0)
        block_preds_dict['rgcn_ensemble_prob'] = ens_scores
        all_pred_dfs.append(pd.DataFrame(block_preds_dict))

        df_b_seeds = pd.DataFrame(block_seed_res)
        pr_aucs = df_b_seeds['pr_auc'].values
        roc_aucs = df_b_seeds['roc_auc'].values
        p50s = df_b_seeds['precision_50'].values
        r50s = df_b_seeds['recall_50'].values

        prev = test_pos / len(df_test)
        warning_msg = "WARNING: Test positive count (5) is very small (< 10); metrics may be unstable." if test_pos < 10 else ""

        summary_records.append({
            'regional_block': block_name,
            'province_codes': str(prov_codes),
            'train_rows': len(df_full_train),
            'train_positives': full_train_pos,
            'test_rows': len(df_test),
            'test_positives': test_pos,
            'test_prevalence': round(prev, 6),
            'pr_auc_mean': round(float(np.mean(pr_aucs)), 4),
            'pr_auc_std': round(float(np.std(pr_aucs)), 4),
            'pr_auc_lift_mean': round(float(np.mean(pr_aucs)) / prev, 2) if test_pos > 0 else 0.0,
            'roc_auc_mean': round(float(np.mean(roc_aucs)), 4),
            'roc_auc_std': round(float(np.std(roc_aucs)), 4),
            'precision_50_mean': round(float(np.mean(p50s)), 4),
            'recall_50_mean': round(float(np.mean(r50s)), 4),
            'small_sample_warning': warning_msg
        })

    df_seed_out = pd.DataFrame(seed_records)
    df_summary_out = pd.DataFrame(summary_records)
    df_preds_out = pd.concat(all_pred_dfs, ignore_index=True)

    os.makedirs(os.path.dirname(METRICS_SEED_OUT), exist_ok=True)
    df_seed_out.to_csv(METRICS_SEED_OUT, index=False)
    df_summary_out.to_csv(SUMMARY_OUT, index=False)
    df_preds_out.to_csv(PREDS_OUT, index=False)

    print("\n==========================================================================================")
    print("      STRICT REGIONAL SPATIAL-TEMPORAL EVALUATION SUMMARY (RELATIONAL GNN)")
    print("==========================================================================================")
    print(df_summary_out[['regional_block', 'test_positives', 'pr_auc_mean', 'pr_auc_std', 'roc_auc_mean', 'precision_50_mean', 'recall_50_mean', 'small_sample_warning']].to_string(index=False))

    if os.path.exists(LR_REGIONAL_METRICS_PATH) and os.path.exists(KNN3_REGIONAL_SUMMARY_PATH):
        df_lr_reg = pd.read_csv(LR_REGIONAL_METRICS_PATH)
        df_knn3_reg = pd.read_csv(KNN3_REGIONAL_SUMMARY_PATH)

        comp_rows = []
        for r_name in REGIONAL_BLOCKS.keys():
            rgcn_row = df_summary_out[df_summary_out['regional_block'] == r_name].iloc[0]
            lr_row = df_lr_reg[df_lr_reg['regional_block'] == r_name].iloc[0]
            knn3_row = df_knn3_reg[df_knn3_reg['regional_block'] == r_name].iloc[0]

            comp_rows.append({
                'regional_block': r_name,
                'test_positives': rgcn_row['test_positives'],
                'strict_lr_pr_auc': lr_row['pr_auc'],
                'knn3_graphsage_pr_auc': knn3_row['pr_auc_mean'],
                'relational_gnn_pr_auc': rgcn_row['pr_auc_mean'],
                'relational_gnn_pr_auc_std': rgcn_row['pr_auc_std'],
                'warning': rgcn_row['small_sample_warning']
            })

        df_comp = pd.DataFrame(comp_rows)
        print("\n==========================================================================================")
        print("          SIDE-BY-SIDE REGIONAL BENCHMARK COMPARISON (PR-AUC)")
        print("==========================================================================================")
        print(df_comp.to_string(index=False))
        print("==========================================================================================")

    print(f"\nSaved multiseed seed metrics to: {METRICS_SEED_OUT}")
    print(f"Saved regional summary to:        {SUMMARY_OUT}")
    print(f"Saved regional predictions to:    {PREDS_OUT}")

if __name__ == '__main__':
    main()
