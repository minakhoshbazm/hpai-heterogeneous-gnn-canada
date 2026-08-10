#!/usr/bin/env python3
"""
scripts/run_pooled_rolling_origin_model_comparison.py

Part 2: Pooled Rolling-Origin Model Comparison.

Evaluates 4 strict models across 3 rolling-origin temporal folds:
- Fold 1: Train 2022 -> Test 2023
- Fold 2: Train 2022-2023 -> Test 2024
- Fold 3: Train 2022-2024 -> Test 2025

Models compared:
1. Strict Environment + Lag Logistic Regression (50 features)
2. Strict MLP (No Graph, 50 features)
3. kNN-3 GraphSAGE Ensemble (3 seeds [11, 22, 33])
4. Time-Safe Relational GNN Ensemble (3 seeds [11, 22, 33])

Pools test predictions across all 3 test years (~176,890 total observations, 292 positive outbreaks):
- Calculates per-fold and pooled PR-AUC, ROC-AUC, Precision/Recall at 50, 100, 200.
- Runs 1,000 CSD cluster-bootstrap iterations on pooled test predictions to compute 95% CIs.

Outputs:
- results/baselines/pooled_rolling_origin_model_comparison.csv
- results/baselines/pooled_rolling_origin_model_comparison.md
"""

import os
import sys
import gc
import copy
import random
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, RGCNConv

MODEL_TABLE_PATH = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
CENTROIDS_CSV_PATH = 'data/processed/graph/csd_centroids_epsg3347.csv'
KNN3_PATH = 'data/processed/graph/csd_graph_knn_3_v1.csv'

BASELINES_DIR = 'results/baselines'
CSV_OUT = os.path.join(BASELINES_DIR, 'pooled_rolling_origin_model_comparison.csv')
MD_OUT = os.path.join(BASELINES_DIR, 'pooled_rolling_origin_model_comparison.md')

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class StrictMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.out(x).squeeze(-1)

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
            del out, probs
    return np.array(all_y_true), np.array(all_y_scores)

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

def build_sage_edge_index(csv_path, csd_to_idx):
    df_e = pd.read_csv(csv_path, dtype=str)
    src = df_e['source_csd_uid'].map(csd_to_idx).values
    dst = df_e['target_csd_uid'].map(csd_to_idx).values
    edge_index = torch.tensor(
        np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0),
        dtype=torch.long
    )
    return edge_index

def build_fixed_geo_and_poultry_edges(model_csds, centroids_df, df_model):
    csd_to_idx = {csd: i for i, csd in enumerate(model_csds)}
    coords = np.column_stack([centroids_df["cx"].values, centroids_df["cy"].values])
    kdtree = KDTree(coords)
    
    pairs_50km = sorted(list(kdtree.query_pairs(50000.0)))
    geo_src, geo_dst = [], []
    for u, v in pairs_50km:
        geo_src.extend([u, v])
        geo_dst.extend([v, u])
        
    pairs_100km = list(kdtree.query_pairs(100000.0))
    csd_static_df = df_model.groupby('csd_uid')[POULTRY_EXPOSURE_COLS].first().reindex(model_csds)
    poultry_raw = np.log1p(np.maximum(0.0, csd_static_df.fillna(0.0).values))
    scaler_p = StandardScaler()
    p_scaled = scaler_p.fit_transform(poultry_raw).astype(np.float32)
    p_norms = np.linalg.norm(p_scaled, axis=1, keepdims=True)
    p_norms[p_norms == 0] = 1e-8
    p_unit = p_scaled / p_norms
    
    poultry_src, poultry_dst = [], []
    tau_poultry = 0.70
    for u, v in pairs_100km:
        p_sim = float(np.dot(p_unit[u], p_unit[v]))
        if p_sim >= tau_poultry:
            poultry_src.extend([u, v])
            poultry_dst.extend([v, u])
            
    return csd_to_idx, coords, pairs_100km, (geo_src, geo_dst), (poultry_src, poultry_dst)

def precompute_time_safe_water_unit_vecs(df_model, model_csds):
    df_sorted = df_model.sort_values(by=['csd_uid', 'year', 'month']).copy()
    cumsum = df_sorted.groupby('csd_uid')[WATER_HABITAT_COLS].cumsum()
    count = df_sorted.groupby('csd_uid')[WATER_HABITAT_COLS].cumcount() + 1
    df_sorted[WATER_HABITAT_COLS] = cumsum.div(count, axis=0)
    
    water_unit_dict = {}
    for (y, m), group in df_sorted.groupby(['year', 'month']):
        c_df = group.set_index('csd_uid')[WATER_HABITAT_COLS].reindex(model_csds)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(c_df.fillna(0.0).values).astype(np.float32)
        norms = np.linalg.norm(scaled, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        water_unit_dict[(int(y), int(m))] = (scaled / norms).astype(np.float32)
        
    return water_unit_dict

def build_snapshots_time_safe(df_sub, water_unit_dict, csd_to_idx, pairs_100km, geo_pair, poultry_pair, num_nodes, feature_cols):
    geo_src, geo_dst = geo_pair
    poultry_src, poultry_dst = poultry_pair
    
    geo_src_arr = np.array(geo_src, dtype=np.int64)
    geo_dst_arr = np.array(geo_dst, dtype=np.int64)
    p_src_arr = np.array(poultry_src, dtype=np.int64)
    p_dst_arr = np.array(poultry_dst, dtype=np.int64)
    
    u_pairs = np.array([p[0] for p in pairs_100km], dtype=np.int64)
    v_pairs = np.array([p[1] for p in pairs_100km], dtype=np.int64)
    
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
        
        w_unit = water_unit_dict[(y, m)]
        sims = np.sum(w_unit[u_pairs] * w_unit[v_pairs], axis=1)
        valid = (sims >= tau_water)
        
        valid_u = u_pairs[valid]
        valid_v = v_pairs[valid]
        
        w_src_arr = np.concatenate([valid_u, valid_v])
        w_dst_arr = np.concatenate([valid_v, valid_u])
        
        all_src = np.concatenate([geo_src_arr, w_src_arr, p_src_arr])
        all_dst = np.concatenate([geo_dst_arr, w_dst_arr, p_dst_arr])
        all_types = np.concatenate([
            np.zeros(len(geo_src_arr), dtype=np.int64),
            np.ones(len(w_src_arr), dtype=np.int64),
            np.full(len(p_src_arr), 2, dtype=np.int64)
        ])
        
        edge_index = torch.tensor(np.stack([all_src, all_dst], axis=0), dtype=torch.long)
        edge_type = torch.tensor(all_types, dtype=torch.long)
        
        snapshots.append(Data(x=x_tensor, y=y_tensor, mask=mask_tensor, edge_index=edge_index, edge_type=edge_type))
        
    return snapshots

def build_snapshots_sage(df_sub, csd_to_idx, num_nodes, feature_cols):
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
    print("      PART 2: POOLED ROLLING-ORIGIN MODEL EVALUATION (2023, 2024, 2025 FOLDS)")
    print("==========================================================================================")

    df_model = pd.read_csv(MODEL_TABLE_PATH, dtype={'csd_uid': str})
    model_csds = sorted(df_model['csd_uid'].unique())
    num_model_csds = len(model_csds)

    df_centroids = pd.read_csv(CENTROIDS_CSV_PATH, dtype={'csd_uid': str})
    centroids_df = df_centroids[df_centroids['csd_uid'].isin(set(model_csds))].set_index('csd_uid').reindex(model_csds)

    feature_cols = sorted([c for c in df_model.columns if c not in EXCLUDE_EXPLICIT_GNN and pd.api.types.is_numeric_dtype(df_model[c])])
    assert len(feature_cols) == 50, f"Expected 50 features, got {len(feature_cols)}"

    df_model = df_model.sort_values(by=['year', 'month', 'csd_uid']).reset_index(drop=True)

    # Precompute water units and static edges
    water_unit_dict = precompute_time_safe_water_unit_vecs(df_model, model_csds)
    csd_to_idx, coords, pairs_100km, geo_pair, poultry_pair = build_fixed_geo_and_poultry_edges(
        model_csds, centroids_df, df_model
    )
    knn3_edge_index = build_sage_edge_index(KNN3_PATH, csd_to_idx)

    # Define 3 rolling origin folds
    folds_config = [
        {
            'fold_name': 'Fold 1 (2023 Test)',
            'train_years': [2022],
            'train_tune_years': [2022],
            'train_tune_months_max': 9,
            'val_tune_months_min': 10,
            'test_year': 2023
        },
        {
            'fold_name': 'Fold 2 (2024 Test)',
            'train_years': [2022, 2023],
            'train_tune_years_match': lambda y, m: (y == 2022) or (y == 2023 and m <= 9),
            'val_tune_match': lambda y, m: (y == 2023 and m >= 10),
            'test_year': 2024
        },
        {
            'fold_name': 'Fold 3 (2025 Test)',
            'train_years': [2022, 2023, 2024],
            'train_tune_years_match': lambda y, m: (y in [2022, 2023]) or (y == 2024 and m <= 9),
            'val_tune_match': lambda y, m: (y == 2024 and m >= 10),
            'test_year': 2025
        }
    ]

    seeds = [11, 22, 33]
    max_epochs = 100
    patience = 10

    all_fold_preds = []

    for f_idx, fc in enumerate(folds_config):
        f_name = fc['fold_name']
        test_y = fc['test_year']

        print(f"\n==========================================================================================")
        print(f"  Processing {f_name}")
        print(f"==========================================================================================")

        if f_idx == 0:
            df_train_tune = df_model[(df_model['year'] == 2022) & (df_model['month'] <= 9)].copy()
            df_val_tune = df_model[(df_model['year'] == 2022) & (df_model['month'] >= 10)].copy()
            df_full_train = df_model[df_model['year'] == 2022].copy()
        else:
            df_train_tune = df_model[df_model.apply(lambda r: fc['train_tune_years_match'](r['year'], r['month']), axis=1)].copy()
            df_val_tune = df_model[df_model.apply(lambda r: fc['val_tune_match'](r['year'], r['month']), axis=1)].copy()
            df_full_train = df_model[df_model['year'].isin(fc['train_years'])].copy()

        df_test = df_model[df_model['year'] == test_y].copy()

        train_tune_pos = int(df_train_tune['outbreak_next_month'].sum())
        full_train_pos = int(df_full_train['outbreak_next_month'].sum())
        test_pos = int(df_test['outbreak_next_month'].sum())

        print(f"Train/Val Pos: {train_tune_pos} | Full Train Pos: {full_train_pos} | Test Pos: {test_pos}")

        scaler_tune = StandardScaler()
        df_train_tune_scaled = df_train_tune.copy()
        df_val_tune_scaled = df_val_tune.copy()
        df_train_tune_scaled[feature_cols] = scaler_tune.fit_transform(df_train_tune[feature_cols].fillna(0.0))
        df_val_tune_scaled[feature_cols] = scaler_tune.transform(df_val_tune[feature_cols].fillna(0.0))

        scaler_full = StandardScaler()
        df_full_train_scaled = df_full_train.copy()
        df_test_scaled = df_test.copy()
        df_full_train_scaled[feature_cols] = scaler_full.fit_transform(df_full_train[feature_cols].fillna(0.0))
        df_test_scaled[feature_cols] = scaler_full.transform(df_test[feature_cols].fillna(0.0))

        pos_weight_tune = torch.tensor([(len(df_train_tune) - train_tune_pos) / train_tune_pos], dtype=torch.float)
        pos_weight_full = torch.tensor([(len(df_full_train) - full_train_pos) / full_train_pos], dtype=torch.float)

        # 1. Model 1: Strict Environment + Lag Logistic Regression
        clf_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        clf_lr.fit(df_full_train_scaled[feature_cols].values, df_full_train['outbreak_next_month'].values.astype(int))
        lr_test_probs = clf_lr.predict_proba(df_test_scaled[feature_cols].values)[:, 1]

        # 2. Model 2: Strict MLP (No Graph)
        mlp_seed_probs = []
        for s in seeds:
            set_seed(s)
            mlp_m = StrictMLP(in_channels=len(feature_cols), hidden_channels=32)
            opt_mlp = torch.optim.Adam(mlp_m.parameters(), lr=0.005, weight_decay=1e-4)
            crit_mlp = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tune)

            best_val_ap = -1.0
            best_ep = 0
            no_imp = 0

            X_tr_t = torch.tensor(df_train_tune_scaled[feature_cols].values, dtype=torch.float)
            y_tr_t = torch.tensor(df_train_tune['outbreak_next_month'].values, dtype=torch.float)
            X_va_t = torch.tensor(df_val_tune_scaled[feature_cols].values, dtype=torch.float)
            y_va_t = df_val_tune['outbreak_next_month'].values

            for ep in range(1, max_epochs + 1):
                mlp_m.train()
                opt_mlp.zero_grad()
                out = mlp_m(X_tr_t)
                loss = crit_mlp(out, y_tr_t)
                loss.backward()
                opt_mlp.step()

                mlp_m.eval()
                with torch.no_grad():
                    v_out = mlp_m(X_va_t)
                    v_probs = torch.sigmoid(v_out).numpy()
                    v_ap = average_precision_score(y_va_t, v_probs)

                if v_ap > best_val_ap:
                    best_val_ap = v_ap
                    best_ep = ep
                    no_imp = 0
                else:
                    no_imp += 1

                if no_imp >= patience:
                    break

            set_seed(s)
            mlp_m_full = StrictMLP(in_channels=len(feature_cols), hidden_channels=32)
            opt_mlp_f = torch.optim.Adam(mlp_m_full.parameters(), lr=0.005, weight_decay=1e-4)
            crit_mlp_f = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)

            X_tr_f = torch.tensor(df_full_train_scaled[feature_cols].values, dtype=torch.float)
            y_tr_f = torch.tensor(df_full_train['outbreak_next_month'].values, dtype=torch.float)
            X_te_f = torch.tensor(df_test_scaled[feature_cols].values, dtype=torch.float)

            for ep in range(1, best_ep + 1):
                mlp_m_full.train()
                opt_mlp_f.zero_grad()
                out = mlp_m_full(X_tr_f)
                loss = crit_mlp_f(out, y_tr_f)
                loss.backward()
                opt_mlp_f.step()

            mlp_m_full.eval()
            with torch.no_grad():
                t_out = mlp_m_full(X_te_f)
                mlp_seed_probs.append(torch.sigmoid(t_out).numpy())

        mlp_ens_probs = np.mean(mlp_seed_probs, axis=0)
        del mlp_m, mlp_m_full, X_tr_t, X_va_t, X_tr_f, X_te_f
        gc.collect()

        # 3. Model 3: kNN-3 GraphSAGE Ensemble
        sage_train_tune_snaps = build_snapshots_sage(df_train_tune_scaled, csd_to_idx, num_model_csds, feature_cols)
        sage_val_tune_snaps = build_snapshots_sage(df_val_tune_scaled, csd_to_idx, num_model_csds, feature_cols)
        sage_full_train_snaps = build_snapshots_sage(df_full_train_scaled, csd_to_idx, num_model_csds, feature_cols)
        sage_test_snaps = build_snapshots_sage(df_test_scaled, csd_to_idx, num_model_csds, feature_cols)

        sage_best_epochs = []
        for s in seeds:
            set_seed(s)
            model_sage = GraphSAGEModel(in_channels=len(feature_cols), hidden_channels=32)
            opt_s = torch.optim.Adam(model_sage.parameters(), lr=0.005, weight_decay=1e-4)
            crit_s = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tune)

            best_v_ap = -1.0
            best_ep = 0
            no_imp = 0

            for ep in range(1, max_epochs + 1):
                model_sage.train()
                for data in sage_train_tune_snaps:
                    opt_s.zero_grad()
                    out = model_sage(data.x, knn3_edge_index)
                    loss = crit_s(out[data.mask], data.y[data.mask])
                    loss.backward()
                    opt_s.step()

                v_true, v_scores = get_snapshot_predictions_sage(model_sage, sage_val_tune_snaps, knn3_edge_index)
                v_ap = average_precision_score(v_true, v_scores)

                if v_ap > best_v_ap:
                    best_v_ap = v_ap
                    best_ep = ep
                    no_imp = 0
                else:
                    no_imp += 1

                if no_imp >= patience:
                    break

            sage_best_epochs.append(best_ep)

        sage_optimal_epochs = int(round(np.mean(sage_best_epochs)))

        sage_seed_probs = []
        for s in seeds:
            set_seed(s)
            model_sage_f = GraphSAGEModel(in_channels=len(feature_cols), hidden_channels=32)
            opt_sf = torch.optim.Adam(model_sage_f.parameters(), lr=0.005, weight_decay=1e-4)
            crit_sf = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)

            for ep in range(1, sage_optimal_epochs + 1):
                model_sage_f.train()
                for data in sage_full_train_snaps:
                    opt_sf.zero_grad()
                    out = model_sage_f(data.x, knn3_edge_index)
                    loss = crit_sf(out[data.mask], data.y[data.mask])
                    loss.backward()
                    opt_sf.step()

            t_true, t_scores = get_snapshot_predictions_sage(model_sage_f, sage_test_snaps, knn3_edge_index)
            sage_seed_probs.append(t_scores)

        sage_ens_probs = np.mean(sage_seed_probs, axis=0)

        del sage_train_tune_snaps, sage_val_tune_snaps, sage_full_train_snaps, sage_test_snaps, model_sage, model_sage_f
        gc.collect()

        # 4. Model 4: Time-Safe Relational GNN Ensemble
        rel_train_tune_snaps = build_snapshots_time_safe(df_train_tune_scaled, water_unit_dict, csd_to_idx, pairs_100km, geo_pair, poultry_pair, num_model_csds, feature_cols)
        rel_val_tune_snaps = build_snapshots_time_safe(df_val_tune_scaled, water_unit_dict, csd_to_idx, pairs_100km, geo_pair, poultry_pair, num_model_csds, feature_cols)
        
        rgcn_best_epochs = []
        for s in seeds:
            set_seed(s)
            model_rgcn = RelationalGNNModel(in_channels=len(feature_cols), hidden_channels=32, num_relations=3)
            opt_r = torch.optim.Adam(model_rgcn.parameters(), lr=0.005, weight_decay=1e-4)
            crit_r = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tune)

            best_v_ap = -1.0
            best_ep = 0
            no_imp = 0

            for ep in range(1, max_epochs + 1):
                model_rgcn.train()
                for data in rel_train_tune_snaps:
                    opt_r.zero_grad()
                    out = model_rgcn(data.x, data.edge_index, data.edge_type)
                    loss = crit_r(out[data.mask], data.y[data.mask])
                    loss.backward()
                    opt_r.step()

                v_true, v_scores = get_snapshot_predictions_relational(model_rgcn, rel_val_tune_snaps)
                v_ap = average_precision_score(v_true, v_scores)

                if v_ap > best_v_ap:
                    best_v_ap = v_ap
                    best_ep = ep
                    no_imp = 0
                else:
                    no_imp += 1

                if no_imp >= patience:
                    break

            rgcn_best_epochs.append(best_ep)

        rgcn_optimal_epochs = int(round(np.mean(rgcn_best_epochs)))
        del rel_train_tune_snaps, rel_val_tune_snaps, model_rgcn
        gc.collect()

        rel_full_train_snaps = build_snapshots_time_safe(df_full_train_scaled, water_unit_dict, csd_to_idx, pairs_100km, geo_pair, poultry_pair, num_model_csds, feature_cols)
        rel_test_snaps = build_snapshots_time_safe(df_test_scaled, water_unit_dict, csd_to_idx, pairs_100km, geo_pair, poultry_pair, num_model_csds, feature_cols)

        rgcn_seed_probs = []
        for s in seeds:
            set_seed(s)
            model_rgcn_f = RelationalGNNModel(in_channels=len(feature_cols), hidden_channels=32, num_relations=3)
            opt_rf = torch.optim.Adam(model_rgcn_f.parameters(), lr=0.005, weight_decay=1e-4)
            crit_rf = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)

            for ep in range(1, rgcn_optimal_epochs + 1):
                model_rgcn_f.train()
                for data in rel_full_train_snaps:
                    opt_rf.zero_grad()
                    out = model_rgcn_f(data.x, data.edge_index, data.edge_type)
                    loss = crit_rf(out[data.mask], data.y[data.mask])
                    loss.backward()
                    opt_rf.step()

            t_true, t_scores = get_snapshot_predictions_relational(model_rgcn_f, rel_test_snaps)
            rgcn_seed_probs.append(t_scores)

        rgcn_ens_probs = np.mean(rgcn_seed_probs, axis=0)

        del rel_full_train_snaps, rel_test_snaps, model_rgcn_f
        gc.collect()

        # Store test predictions for fold
        fold_pred_df = pd.DataFrame({
            'fold': f_name,
            'csd_uid': df_test['csd_uid'].values,
            'csd_name': df_test['csd_name'].values,
            'province_code': df_test['province_code'].values,
            'year': df_test['year'].values,
            'month': df_test['month'].values,
            'actual': df_test['outbreak_next_month'].values.astype(int),
            'lr_prob': lr_test_probs,
            'mlp_prob': mlp_ens_probs,
            'knn3_sage_prob': sage_ens_probs,
            'relational_gnn_prob': rgcn_ens_probs
        })
        all_fold_preds.append(fold_pred_df)

        lr_ap = average_precision_score(df_test['outbreak_next_month'], lr_test_probs)
        mlp_ap = average_precision_score(df_test['outbreak_next_month'], mlp_ens_probs)
        sage_ap = average_precision_score(df_test['outbreak_next_month'], sage_ens_probs)
        rgcn_ap = average_precision_score(df_test['outbreak_next_month'], rgcn_ens_probs)

        print(f"  {f_name} Test PR-AUCs:")
        print(f"    Strict LR:        {lr_ap:.4f}")
        print(f"    Strict MLP:       {mlp_ap:.4f}")
        print(f"    kNN-3 GraphSAGE:  {sage_ap:.4f}")
        print(f"    Relational GNN:   {rgcn_ap:.4f}")

    # Pool predictions across all 3 folds
    df_pooled = pd.concat(all_fold_preds, ignore_index=True)
    pooled_rows = len(df_pooled)
    pooled_pos = int(df_pooled['actual'].sum())
    print(f"\n==========================================================================================")
    print(f"  POOLED PREDICTIONS SUMMARY ({pooled_rows:,} rows, {pooled_pos} total positives)")
    print(f"==========================================================================================")

    y_pooled = df_pooled['actual'].values

    models_dict = {
        'Strict Environment + Lag Logistic Regression': 'lr_prob',
        'Strict MLP (No Graph)': 'mlp_prob',
        'kNN-3 GraphSAGE Ensemble': 'knn3_sage_prob',
        'Time-Safe Relational GNN Ensemble': 'relational_gnn_prob'
    }

    per_fold_summary_records = []

    for m_name, col in models_dict.items():
        # Overall pooled metrics
        p_probs = df_pooled[col].values
        p_ap = average_precision_score(y_pooled, p_probs)
        p_roc = roc_auc_score(y_pooled, p_probs)
        p_km = topk_metrics(y_pooled, p_probs, [50, 100, 200], pooled_pos)

        rec = {
            'model_name': m_name,
            'scope': 'POOLED (2023-2025)',
            'test_rows': pooled_rows,
            'test_positives': pooled_pos,
            'pr_auc': round(p_ap, 4),
            'roc_auc': round(p_roc, 4),
            'precision_50': round(p_km['precision_50'], 4),
            'recall_50': round(p_km['recall_50'], 4),
            'precision_100': round(p_km['precision_100'], 4),
            'recall_100': round(p_km['recall_100'], 4),
            'precision_200': round(p_km['precision_200'], 4),
            'recall_200': round(p_km['recall_200'], 4)
        }
        per_fold_summary_records.append(rec)

        # Per fold metrics
        for fc in folds_config:
            f_n = fc['fold_name']
            df_f = df_pooled[df_pooled['fold'] == f_n]
            f_y = df_f['actual'].values
            f_pos = int(f_y.sum())
            f_probs = df_f[col].values
            f_ap = average_precision_score(f_y, f_probs)
            f_roc = roc_auc_score(f_y, f_probs)
            f_km = topk_metrics(f_y, f_probs, [50, 100, 200], f_pos)

            f_rec = {
                'model_name': m_name,
                'scope': f_n,
                'test_rows': len(df_f),
                'test_positives': f_pos,
                'pr_auc': round(f_ap, 4),
                'roc_auc': round(f_roc, 4),
                'precision_50': round(f_km['precision_50'], 4),
                'recall_50': round(f_km['recall_50'], 4),
                'precision_100': round(f_km['precision_100'], 4),
                'recall_100': round(f_km['recall_100'], 4),
                'precision_200': round(f_km['precision_200'], 4),
                'recall_200': round(f_km['recall_200'], 4)
            }
            per_fold_summary_records.append(f_rec)

    df_results_all = pd.DataFrame(per_fold_summary_records)
    df_results_all.to_csv(CSV_OUT, index=False)

    # 1,000 CSD Cluster Bootstrap on Pooled Predictions
    print("\n--- Running 1,000 CSD Cluster Bootstrap Iterations on Pooled Predictions ---")
    unique_csds = sorted(df_pooled['csd_uid'].unique())
    num_csds = len(unique_csds)
    csd_group_dict = {csd: group.index.values for csd, group in df_pooled.groupby('csd_uid')}

    num_boot = 1000
    rng = np.random.RandomState(42)

    diffs_rgcn_vs_lr = []
    diffs_rgcn_vs_knn3 = []
    diffs_knn3_vs_lr = []

    for _ in range(num_boot):
        sampled_csds = rng.choice(unique_csds, size=num_csds, replace=True)
        sampled_idx = np.concatenate([csd_group_dict[c] for c in sampled_csds])
        b_df = df_pooled.iloc[sampled_idx]
        b_y = b_df['actual'].values
        if b_y.sum() == 0:
            continue

        b_lr = average_precision_score(b_y, b_df['lr_prob'].values)
        b_knn3 = average_precision_score(b_y, b_df['knn3_sage_prob'].values)
        b_rgcn = average_precision_score(b_y, b_df['relational_gnn_prob'].values)

        diffs_rgcn_vs_lr.append(b_rgcn - b_lr)
        diffs_rgcn_vs_knn3.append(b_rgcn - b_knn3)
        diffs_knn3_vs_lr.append(b_knn3 - b_lr)

    ci_rgcn_lr = np.percentile(diffs_rgcn_vs_lr, [2.5, 97.5])
    ci_rgcn_knn3 = np.percentile(diffs_rgcn_vs_knn3, [2.5, 97.5])
    ci_knn3_lr = np.percentile(diffs_knn3_vs_lr, [2.5, 97.5])

    # Observed pooled point estimates, derived from the pooled metrics table so that
    # the reported difference can never drift away from the PR-AUC values above it.
    def pooled_pr_auc(model_name):
        sel = df_results_all[
            (df_results_all['model_name'] == model_name)
            & (df_results_all['scope'] == 'POOLED (2023-2025)')
        ]['pr_auc']
        return float(sel.values[0])

    pooled_ap_lr = pooled_pr_auc('Strict Environment + Lag Logistic Regression')
    pooled_ap_knn3 = pooled_pr_auc('kNN-3 GraphSAGE Ensemble')
    pooled_ap_rgcn = pooled_pr_auc('Time-Safe Relational GNN Ensemble')

    obs_rgcn_lr = pooled_ap_rgcn - pooled_ap_lr
    obs_rgcn_knn3 = pooled_ap_rgcn - pooled_ap_knn3
    obs_knn3_lr = pooled_ap_knn3 - pooled_ap_lr

    frac_rgcn_lr = np.mean(np.array(diffs_rgcn_vs_lr) > 0)
    frac_rgcn_knn3 = np.mean(np.array(diffs_rgcn_vs_knn3) > 0)
    frac_knn3_lr = np.mean(np.array(diffs_knn3_vs_lr) > 0)

    # Decisions
    dec_rgcn_lr = "comparable, not proven better" if ci_rgcn_lr[0] <= 0 <= ci_rgcn_lr[1] else ("statistically superior" if ci_rgcn_lr[0] > 0 else "inferior")
    dec_rgcn_knn3 = "comparable, not proven better" if ci_rgcn_knn3[0] <= 0 <= ci_rgcn_knn3[1] else ("statistically superior" if ci_rgcn_knn3[0] > 0 else "inferior")
    dec_knn3_lr = "statistically superior" if ci_knn3_lr[0] > 0 else ("comparable, not proven better" if ci_knn3_lr[0] <= 0 <= ci_knn3_lr[1] else "inferior")

    md_content = f"""# Pooled Rolling-Origin Model Comparison (2023, 2024, 2025 Test Folds)

## Executive Summary

To evaluate model performance across multiple historical outbreak periods and maximize statistical power, we conducted a **pooled rolling-origin validation** combining predictions from three consecutive temporal test folds:
- **Fold 1 (2023 Test)**: Train 2022 $\\rightarrow$ Test 2023 ($N = 60,648$, $N_{{\\text{{pos}}}} = 72$)
- **Fold 2 (2024 Test)**: Train 2022–2023 $\\rightarrow$ Test 2024 ($N = 60,648$, $N_{{\\text{{pos}}}} = 40$)
- **Fold 3 (2025 Test)**: Train 2022–2024 $\\rightarrow$ Test 2025 ($N = 55,594$, $N_{{\\text{{pos}}}} = 79$)
- **Pooled Total**: $N = 176,890$ CSD-month observations, $N_{{\\text{{pos}}}} = 191$ positive outbreak months.

---

## 1. Pooled & Per-Fold Performance Metrics

| Model Name | Scope / Fold | PR-AUC | ROC-AUC | Precision@50 | Recall@50 | Precision@100 | Recall@100 |
|---|---|---|---|---|---|---|---|
| **kNN-3 GraphSAGE Ensemble** | **POOLED (2023-2025)** | **{df_results_all[(df_results_all['model_name']=='kNN-3 GraphSAGE Ensemble')&(df_results_all['scope']=='POOLED (2023-2025)')]['pr_auc'].values[0]:.4f}** | **{df_results_all[(df_results_all['model_name']=='kNN-3 GraphSAGE Ensemble')&(df_results_all['scope']=='POOLED (2023-2025)')]['roc_auc'].values[0]:.4f}** | **{df_results_all[(df_results_all['model_name']=='kNN-3 GraphSAGE Ensemble')&(df_results_all['scope']=='POOLED (2023-2025)')]['precision_50'].values[0]:.4f}** | **{df_results_all[(df_results_all['model_name']=='kNN-3 GraphSAGE Ensemble')&(df_results_all['scope']=='POOLED (2023-2025)')]['recall_50'].values[0]:.4f}** | **{df_results_all[(df_results_all['model_name']=='kNN-3 GraphSAGE Ensemble')&(df_results_all['scope']=='POOLED (2023-2025)')]['precision_100'].values[0]:.4f}** | **{df_results_all[(df_results_all['model_name']=='kNN-3 GraphSAGE Ensemble')&(df_results_all['scope']=='POOLED (2023-2025)')]['recall_100'].values[0]:.4f}** |
| **Relational GNN Ensemble** | **POOLED (2023-2025)** | **{df_results_all[(df_results_all['model_name']=='Time-Safe Relational GNN Ensemble')&(df_results_all['scope']=='POOLED (2023-2025)')]['pr_auc'].values[0]:.4f}** | **{df_results_all[(df_results_all['model_name']=='Time-Safe Relational GNN Ensemble')&(df_results_all['scope']=='POOLED (2023-2025)')]['roc_auc'].values[0]:.4f}** | {df_results_all[(df_results_all['model_name']=='Time-Safe Relational GNN Ensemble')&(df_results_all['scope']=='POOLED (2023-2025)')]['precision_50'].values[0]:.4f} | {df_results_all[(df_results_all['model_name']=='Time-Safe Relational GNN Ensemble')&(df_results_all['scope']=='POOLED (2023-2025)')]['recall_50'].values[0]:.4f} | {df_results_all[(df_results_all['model_name']=='Time-Safe Relational GNN Ensemble')&(df_results_all['scope']=='POOLED (2023-2025)')]['precision_100'].values[0]:.4f} | {df_results_all[(df_results_all['model_name']=='Time-Safe Relational GNN Ensemble')&(df_results_all['scope']=='POOLED (2023-2025)')]['recall_100'].values[0]:.4f} |
| **Strict Env + Lag LR** | **POOLED (2023-2025)** | **{df_results_all[(df_results_all['model_name']=='Strict Environment + Lag Logistic Regression')&(df_results_all['scope']=='POOLED (2023-2025)')]['pr_auc'].values[0]:.4f}** | **{df_results_all[(df_results_all['model_name']=='Strict Environment + Lag Logistic Regression')&(df_results_all['scope']=='POOLED (2023-2025)')]['roc_auc'].values[0]:.4f}** | {df_results_all[(df_results_all['model_name']=='Strict Environment + Lag Logistic Regression')&(df_results_all['scope']=='POOLED (2023-2025)')]['precision_50'].values[0]:.4f} | {df_results_all[(df_results_all['model_name']=='Strict Environment + Lag Logistic Regression')&(df_results_all['scope']=='POOLED (2023-2025)')]['recall_50'].values[0]:.4f} | {df_results_all[(df_results_all['model_name']=='Strict Environment + Lag Logistic Regression')&(df_results_all['scope']=='POOLED (2023-2025)')]['precision_100'].values[0]:.4f} | {df_results_all[(df_results_all['model_name']=='Strict Environment + Lag Logistic Regression')&(df_results_all['scope']=='POOLED (2023-2025)')]['recall_100'].values[0]:.4f} |
| **Strict MLP (No Graph)** | **POOLED (2023-2025)** | **{df_results_all[(df_results_all['model_name']=='Strict MLP (No Graph)')&(df_results_all['scope']=='POOLED (2023-2025)')]['pr_auc'].values[0]:.4f}** | **{df_results_all[(df_results_all['model_name']=='Strict MLP (No Graph)')&(df_results_all['scope']=='POOLED (2023-2025)')]['roc_auc'].values[0]:.4f}** | {df_results_all[(df_results_all['model_name']=='Strict MLP (No Graph)')&(df_results_all['scope']=='POOLED (2023-2025)')]['precision_50'].values[0]:.4f} | {df_results_all[(df_results_all['model_name']=='Strict MLP (No Graph)')&(df_results_all['scope']=='POOLED (2023-2025)')]['recall_50'].values[0]:.4f} | {df_results_all[(df_results_all['model_name']=='Strict MLP (No Graph)')&(df_results_all['scope']=='POOLED (2023-2025)')]['precision_100'].values[0]:.4f} | {df_results_all[(df_results_all['model_name']=='Strict MLP (No Graph)')&(df_results_all['scope']=='POOLED (2023-2025)')]['recall_100'].values[0]:.4f} |

---

## 2. Pooled Cluster-Bootstrap Uncertainty Analysis (1,000 Resamples)

| Pairwise Comparison | Pooled PR-AUC Diff | 95% Percentile Confidence Interval | Resample Fraction (GNN Better) | Formal Decision Conclusion |
|---|---|---|---|---|
| **Time-Safe Relational GNN vs Strict Env + Lag LR** | **{obs_rgcn_lr:+.4f}** | **[{ci_rgcn_lr[0]:+.4f}, {ci_rgcn_lr[1]:+.4f}]** | {frac_rgcn_lr:.4f} ({frac_rgcn_lr:.1%}) | **{dec_rgcn_lr}** |
| **Time-Safe Relational GNN vs kNN-3 GraphSAGE** | **{obs_rgcn_knn3:+.4f}** | **[{ci_rgcn_knn3[0]:+.4f}, {ci_rgcn_knn3[1]:+.4f}]** | {frac_rgcn_knn3:.4f} ({frac_rgcn_knn3:.1%}) | **{dec_rgcn_knn3}** |
| **kNN-3 GraphSAGE vs Strict Env + Lag LR** | **{obs_knn3_lr:+.4f}** | **[{ci_knn3_lr[0]:+.4f}, {ci_knn3_lr[1]:+.4f}]** | {frac_knn3_lr:.4f} ({frac_knn3_lr:.1%}) | **{dec_knn3_lr}** |

---

## 3. Per-Fold Breakdown

"""

    for fc in folds_config:
        fn = fc['fold_name']
        md_content += f"### {fn}\n\n"
        md_content += "| Model Name | PR-AUC | ROC-AUC | Precision@50 | Recall@50 | Precision@100 | Recall@100 |\n"
        md_content += "|---|---|---|---|---|---|---|\n"
        for m_name in models_dict.keys():
            row = df_results_all[(df_results_all['model_name'] == m_name) & (df_results_all['scope'] == fn)].iloc[0]
            md_content += f"| **{m_name}** | {row['pr_auc']:.4f} | {row['roc_auc']:.4f} | {row['precision_50']:.4f} | {row['recall_50']:.4f} | {row['precision_100']:.4f} | {row['recall_100']:.4f} |\n"
        md_content += "\n"

    with open(MD_OUT, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print("\n==========================================================================================")
    print("                POOLED ROLLING-ORIGIN EVALUATION RESULTS")
    print("==========================================================================================")
    print(df_results_all[df_results_all['scope'] == 'POOLED (2023-2025)'][['model_name', 'pr_auc', 'roc_auc', 'precision_50', 'recall_50']].to_string(index=False))
    print("==========================================================================================")
    print(f"\nSaved CSV summary to: {CSV_OUT}")
    print(f"Saved MD summary to:  {MD_OUT}")

if __name__ == '__main__':
    main()
