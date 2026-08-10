#!/usr/bin/env python3
"""
scripts/train_spatiotemporal_directed_relational_gnn_transfer.py

Spatiotemporal Directed Relational GNN with Supervised Transfer Learning.
Pretrains on wild-bird HPAI next-month presence, then fine-tunes on poultry outbreak next-month presence.
"""

import os
import sys
import gc
import random
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_TABLE_PATH = 'data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv'
CENTROIDS_CSV_PATH = 'data/processed/graph/csd_centroids_epsg3347.csv'
KNN3_PATH = 'data/processed/graph/csd_graph_knn_3_v1.csv'
STRICT_LR_PREDS_PATH = 'results/baselines/environment_with_lags_v1_strict_test_predictions.csv'
KNN3_PREDS_PATH = 'results/baselines/graphsage_selected_spatial_graph_test_predictions.csv'

WB_RAW_CSV = 'data/raw/cfia/CFIA_outbreaks_wildbirds.csv'
WB_CROSSWALK = 'data/processed/geography/wildbird_cfia_ccs_crosswalk_reviewed.csv'

BASELINES_DIR = 'results/baselines'
MULTI_METRICS_OUT = os.path.join(BASELINES_DIR, 'temporal_gnn_transfer_multiseed_metrics.csv')
SUMMARY_OUT = os.path.join(BASELINES_DIR, 'temporal_gnn_transfer_summary.csv')
PREDS_OUT = os.path.join(BASELINES_DIR, 'temporal_gnn_transfer_test_predictions.csv')
BOOTSTRAP_CSV_OUT = os.path.join(BASELINES_DIR, 'temporal_gnn_transfer_vs_baselines_bootstrap_comparison.csv')
BOOTSTRAP_MD_OUT = os.path.join(BASELINES_DIR, 'temporal_gnn_transfer_vs_baselines_bootstrap_summary.md')

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

class SparseSpatiotemporalRGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, num_relations=4):
        super().__init__()
        self.num_relations = num_relations
        
        # Self-loops & neighbor aggregators for layer 1
        self.w_self1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.w_neigh1 = nn.ModuleList([nn.Linear(in_channels, hidden_channels, bias=False) for _ in range(num_relations)])
        self.rel_weights1 = nn.Parameter(torch.ones(num_relations) / float(num_relations))
        self.bias1 = nn.Parameter(torch.zeros(hidden_channels))

        # Self-loops & neighbor aggregators for layer 2
        self.w_self2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.w_neigh2 = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(num_relations)])
        self.rel_weights2 = nn.Parameter(torch.ones(num_relations) / float(num_relations))
        self.bias2 = nn.Parameter(torch.zeros(hidden_channels))

        # Skip connections
        self.raw_proj = nn.Linear(in_channels, hidden_channels, bias=False)
        self.skip_weight = nn.Parameter(torch.tensor(0.5))

        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, sparse_adj_list):
        # Layer 1
        h_self1 = self.w_self1(x)
        h_rel1 = []
        for r, A_r in enumerate(sparse_adj_list):
            msg = torch.sparse.mm(A_r, x)
            h_rel1.append(self.w_neigh1[r](msg))
        
        h1 = h_self1 + sum(self.rel_weights1[r] * h_rel1[r] for r in range(self.num_relations)) + self.bias1
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.2, training=self.training)

        # Layer 2 with Residual Connection (Skip connection from h1)
        h_self2 = self.w_self2(h1)
        h_rel2 = []
        for r, A_r in enumerate(sparse_adj_list):
            msg = torch.sparse.mm(A_r, h1)
            h_rel2.append(self.w_neigh2[r](msg))
            
        h2 = h_self2 + sum(self.rel_weights2[r] * h_rel2[r] for r in range(self.num_relations)) + self.bias2
        h2 = h2 + h1  # residual skip connection

        # Raw Feature Skip connection: projection of raw inputs to output representation
        h2 = h2 + self.skip_weight * self.raw_proj(x)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=0.2, training=self.training)

        return self.out(h2).squeeze(-1)

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

def precompute_time_safe_water_units(df_model, model_csds):
    sub_df = df_model[['csd_uid', 'year', 'month'] + WATER_HABITAT_COLS].copy()
    df_sorted = sub_df.sort_values(by=['csd_uid', 'year', 'month'])
    
    vals = df_sorted[WATER_HABITAT_COLS].values
    N, C = vals.shape
    num_csds = N // 47
    vals_reshaped = vals.reshape(num_csds, 47, C)
    cumsum = np.cumsum(vals_reshaped, axis=1)
    counts = np.arange(1, 48).reshape(1, 47, 1)
    cummean = cumsum / counts
    df_sorted[WATER_HABITAT_COLS] = cummean.reshape(N, C)

    water_unit_dict = {}
    for (y, m), group in df_sorted.groupby(['year', 'month']):
        c_df = group.set_index('csd_uid')[WATER_HABITAT_COLS].reindex(model_csds)
        scaled = c_df.fillna(0.0).values.astype(np.float32)
        mean = scaled.mean(axis=0)
        std = scaled.std(axis=0)
        std[std == 0] = 1e-8
        scaled = (scaled - mean) / std
        norms = np.linalg.norm(scaled, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        water_unit_dict[(int(y), int(m))] = scaled / norms

    return water_unit_dict

def load_knn3_spatial_pairs(csv_path, csd_to_idx):
    df_e = pd.read_csv(csv_path, dtype=str)
    src = df_e['source_csd_uid'].map(csd_to_idx).values
    dst = df_e['target_csd_uid'].map(csd_to_idx).values
    return src, dst

def build_sparse_adj_matrices(model_csds, centroids_df, df_model, water_unit_dict, ym_tuples, total_nodes):
    num_csds = len(model_csds)
    num_months = len(ym_tuples)
    csd_to_idx = {csd: i for i, csd in enumerate(model_csds)}

    # Candidate 50km spatial pairs
    coords = np.column_stack([centroids_df["cx"].values, centroids_df["cy"].values])
    kdtree = KDTree(coords)
    pairs_50km = sorted(list(kdtree.query_pairs(50000.0)))
    u_knn3, v_knn3 = load_knn3_spatial_pairs(KNN3_PATH, csd_to_idx)

    # Static poultry exposure similarity
    csd_static_df = df_model.groupby('csd_uid')[POULTRY_EXPOSURE_COLS].first().reindex(model_csds)
    poultry_raw = np.log1p(np.maximum(0.0, csd_static_df.fillna(0.0).values))
    scaler_p = StandardScaler()
    p_scaled = scaler_p.fit_transform(poultry_raw).astype(np.float32)
    p_norms = np.linalg.norm(p_scaled, axis=1, keepdims=True)
    p_norms[p_norms == 0] = 1e-8
    p_unit = p_scaled / p_norms

    tau_poultry = 0.80
    tau_water = 0.80

    poultry_pairs_50km = [p for p in pairs_50km if float(np.dot(p_unit[p[0]], p_unit[p[1]])) >= tau_poultry]
    u_poul = np.array([p[0] for p in poultry_pairs_50km], dtype=np.int64)
    v_poul = np.array([p[1] for p in poultry_pairs_50km], dtype=np.int64)

    u_50km = np.array([p[0] for p in pairs_50km], dtype=np.int64)
    v_50km = np.array([p[1] for p in pairs_50km], dtype=np.int64)

    rel_src = [[], [], [], []]
    rel_dst = [[], [], [], []]

    for t in range(num_months - 1):
        t_src_offset = t * num_csds
        t_dst_offset = (t + 1) * num_csds

        # Relation 0: temporal_self (same CSD t -> t+1)
        self_src = t_src_offset + np.arange(num_csds)
        self_dst = t_dst_offset + np.arange(num_csds)
        rel_src[0].append(self_src)
        rel_dst[0].append(self_dst)

        # Relation 1: temporal_spatial_geographic (kNN-3 t -> t+1)
        if len(u_knn3) > 0:
            k3_src = np.concatenate([t_src_offset + u_knn3, t_src_offset + v_knn3])
            k3_dst = np.concatenate([t_dst_offset + v_knn3, t_dst_offset + u_knn3])
            rel_src[1].append(k3_src)
            rel_dst[1].append(k3_dst)

        # Relation 2: temporal_spatial_water (50km sim >= 0.80)
        y_src, m_src = ym_tuples[t]
        w_unit = water_unit_dict[(int(y_src), int(m_src))]
        water_pairs_50km = [p for p in pairs_50km if float(np.dot(w_unit[p[0]], w_unit[p[1]])) >= tau_water]
        if len(water_pairs_50km) > 0:
            u_w = np.array([p[0] for p in water_pairs_50km], dtype=np.int64)
            v_w = np.array([p[1] for p in water_pairs_50km], dtype=np.int64)
            w_src = np.concatenate([t_src_offset + u_w, t_src_offset + v_w])
            w_dst = np.concatenate([t_dst_offset + v_w, t_dst_offset + u_w])
            rel_src[2].append(w_src)
            rel_dst[2].append(w_dst)

        # Relation 3: temporal_spatial_poultry (50km sim >= 0.80)
        if len(u_poul) > 0:
            p_src = np.concatenate([t_src_offset + u_poul, t_src_offset + v_poul])
            p_dst = np.concatenate([t_dst_offset + v_poul, t_dst_offset + u_poul])
            rel_src[3].append(p_src)
            rel_dst[3].append(p_dst)

    sparse_adj_list = []
    total_edges = 0

    for r in range(4):
        r_src = np.concatenate(rel_src[r])
        r_dst = np.concatenate(rel_dst[r])
        total_edges += len(r_src)

        # Calculate in-degree normalization
        indices = np.vstack([r_dst, r_src]) # row = dst, col = src for matrix mult A @ X
        ind_tensor = torch.from_numpy(indices).to(torch.long)
        
        deg = torch.zeros(total_nodes, dtype=torch.float)
        deg.scatter_add_(0, ind_tensor[0], torch.ones(ind_tensor.shape[1], dtype=torch.float))
        deg[deg == 0] = 1.0
        
        values = 1.0 / deg[ind_tensor[0]]
        
        A_r = torch.sparse_coo_tensor(ind_tensor, values, size=(total_nodes, total_nodes), dtype=torch.float).coalesce()
        sparse_adj_list.append(A_r)

    return csd_to_idx, sparse_adj_list, total_edges

def build_wildbird_target(wb_raw_path, crosswalk_path, df_model):
    wb = pd.read_csv(wb_raw_path, encoding="latin-1")
    xw = pd.read_csv(crosswalk_path, encoding="latin-1")

    xw_matched = xw[xw["final_status"] == "matched"][
        ["raw_ccs_name", "raw_province", "target_ccs_uid"]
    ].dropna()

    wb["date_parsed"] = pd.to_datetime(wb["Collection Date"], errors="coerce")
    wb = wb.dropna(subset=["date_parsed"])
    wb["year"]  = wb["date_parsed"].dt.year
    wb["month"] = wb["date_parsed"].dt.month

    wb["raw_ccs_name"] = wb["Census Consolidated Subdivision (CCS)"].str.strip().str.lower()
    wb["raw_province"] = wb["Province"].str.strip().str.lower()

    xw_matched = xw_matched.copy()
    xw_matched["raw_ccs_name"] = xw_matched["raw_ccs_name"].str.strip().str.lower()
    xw_matched["raw_province"] = xw_matched["raw_province"].str.strip().str.lower()

    wb_m = wb.merge(xw_matched, on=["raw_ccs_name", "raw_province"], how="inner")
    
    wb_agg = (
        wb_m.groupby(["target_ccs_uid", "year", "month"])
        .size()
        .reset_index(name="wb_count")
        .rename(columns={"target_ccs_uid": "ccs_uid"})
    )
    wb_agg["wildbird_hpai_present"] = 1.0
    wb_agg["ccs_uid"] = wb_agg["ccs_uid"].astype(str)

    all_ccs_months = df_model[["ccs_uid", "year", "month"]].drop_duplicates().copy()
    all_ccs_months["ccs_uid"] = all_ccs_months["ccs_uid"].astype(str)
    wb_presence = all_ccs_months.merge(
        wb_agg[["ccs_uid", "year", "month", "wildbird_hpai_present"]],
        on=["ccs_uid", "year", "month"], how="left"
    ).fillna({"wildbird_hpai_present": 0.0})

    wb_pres_sorted = wb_presence.sort_values(["ccs_uid", "year", "month"])

    def next_ym(y, m):
        if m == 12:
            return (y + 1, 1)
        return (y, m + 1)

    wb_next = wb_pres_sorted[["ccs_uid", "year", "month", "wildbird_hpai_present"]].copy()
    wb_next.columns = ["ccs_uid", "next_year", "next_month", "wildbird_hpai_present_next_month"]
    wb_next["ccs_uid"] = wb_next["ccs_uid"].astype(str)

    df_nxt = df_model[["csd_uid", "ccs_uid", "year", "month"]].copy()
    df_nxt["ccs_uid"] = df_nxt["ccs_uid"].astype(str)
    df_nxt[["next_year", "next_month"]] = pd.DataFrame(
        [next_ym(r.year, r.month) for r in df_nxt.itertuples()],
        columns=["next_year", "next_month"]
    )
    df_nxt = df_nxt.merge(
        wb_next, on=["ccs_uid", "next_year", "next_month"], how="left"
    ).fillna({"wildbird_hpai_present_next_month": 0.0})

    return df_nxt[["csd_uid", "year", "month", "wildbird_hpai_present_next_month"]]

def main():
    print("==========================================================================================")
    print("      TRUE SPATIOTEMPORAL DIRECTED RELATIONAL GNN (TRANSFER LEARNING)")
    print("==========================================================================================")

    df_model = pd.read_csv(MODEL_TABLE_PATH, dtype={'csd_uid': str})
    model_csds = sorted(df_model['csd_uid'].unique())
    num_csds = len(model_csds)

    df_centroids = pd.read_csv(CENTROIDS_CSV_PATH, dtype={'csd_uid': str})
    centroids_df = df_centroids[df_centroids['csd_uid'].isin(set(model_csds))].set_index('csd_uid').reindex(model_csds)

    feature_cols = sorted([c for c in df_model.columns if c not in EXCLUDE_EXPLICIT_GNN and pd.api.types.is_numeric_dtype(df_model[c])])

    # Sort dataset by (year, month, csd_uid)
    df_model = df_model.sort_values(by=['year', 'month', 'csd_uid']).reset_index(drop=True)
    ym_tuples = df_model[['year', 'month']].drop_duplicates().sort_values(['year', 'month']).values
    num_months = len(ym_tuples)
    total_nodes = num_csds * num_months

    print(f"Total CSDs: {num_csds:,} | Total Months: {num_months} | Total Nodes (CSD-months): {total_nodes:,}")

    # Build wildbird targets
    print("\nBuilding wild-bird next-month presence target...")
    df_wb = build_wildbird_target(WB_RAW_CSV, WB_CROSSWALK, df_model)
    df_model = df_model.merge(df_wb, on=['csd_uid', 'year', 'month'], how='left').fillna({'wildbird_hpai_present_next_month': 0.0})

    # Precompute time-safe water habitat features and sparse spatiotemporal adjacency matrices
    print("Precomputing time-safe water features & building sparse spatiotemporal adjacency matrices...")
    water_unit_dict = precompute_time_safe_water_units(df_model, model_csds)
    csd_to_idx, sparse_adj_list, total_edges = build_sparse_adj_matrices(
        model_csds, centroids_df, df_model, water_unit_dict, ym_tuples, total_nodes
    )
    print(f"Constructed spatiotemporal graph with {total_edges:,} total directed forward edges across 4 relations.")

    # Masks for train, val, test splits
    node_year = df_model['year'].values
    y_poultry_np = df_model['outbreak_next_month'].values.astype(np.float32)
    y_wildbird_np = df_model['wildbird_hpai_present_next_month'].values.astype(np.float32)

    train_tune_mask_np = np.isin(node_year, [2022, 2023])
    val_tune_mask_np = (node_year == 2024)
    full_train_mask_np = np.isin(node_year, [2022, 2023, 2024])
    test_mask_np = (node_year == 2025)

    train_tune_pos_wb = int(y_wildbird_np[train_tune_mask_np].sum())
    train_tune_pos_pl = int(y_poultry_np[train_tune_mask_np].sum())
    val_tune_pos_pl = int(y_poultry_np[val_tune_mask_np].sum())
    full_train_pos_wb = int(y_wildbird_np[full_train_mask_np].sum())
    full_train_pos_pl = int(y_poultry_np[full_train_mask_np].sum())
    test_pos_pl = int(y_poultry_np[test_mask_np].sum())

    print(f"\nSplit Summary (Nodes & Positives):")
    print(f"  Train Tune Split: {train_tune_mask_np.sum():,} nodes | WB Positives: {train_tune_pos_wb} | Poultry Positives: {train_tune_pos_pl}")
    print(f"  Val Tune Split:   {val_tune_mask_np.sum():,} nodes | Poultry Positives: {val_tune_pos_pl}")
    print(f"  Test Split:       {test_mask_np.sum():,} nodes | Poultry Positives: {test_pos_pl}")

    # Fit Scaler strictly on 2022-2023 for tuning
    scaler_tune = StandardScaler()
    X_tune_raw = df_model[feature_cols].values.astype(np.float32)
    scaler_tune.fit(X_tune_raw[train_tune_mask_np])
    X_tune_scaled = scaler_tune.transform(X_tune_raw)

    # Fit Scaler strictly on 2022-2024 for full retraining
    scaler_full = StandardScaler()
    scaler_full.fit(X_tune_raw[full_train_mask_np])
    X_full_scaled = scaler_full.transform(X_tune_raw)

    pos_weight_wb_tune = torch.tensor([(train_tune_mask_np.sum() - train_tune_pos_wb) / train_tune_pos_wb], dtype=torch.float)
    pos_weight_pl_tune = torch.tensor([(train_tune_mask_np.sum() - train_tune_pos_pl) / train_tune_pos_pl], dtype=torch.float)
    pos_weight_wb_full = torch.tensor([(full_train_mask_np.sum() - full_train_pos_wb) / full_train_pos_wb], dtype=torch.float)
    pos_weight_pl_full = torch.tensor([(full_train_mask_np.sum() - full_train_pos_pl) / full_train_pos_pl], dtype=torch.float)

    train_tune_mask = torch.tensor(train_tune_mask_np, dtype=torch.bool)
    val_tune_mask = torch.tensor(val_tune_mask_np, dtype=torch.bool)
    full_train_mask = torch.tensor(full_train_mask_np, dtype=torch.bool)
    test_mask = torch.tensor(test_mask_np, dtype=torch.bool)

    y_poultry_tensor = torch.tensor(y_poultry_np, dtype=torch.float)
    y_wildbird_tensor = torch.tensor(y_wildbird_np, dtype=torch.float)

    seeds = [11, 22, 33]
    max_epochs = 100
    patience = 10

    # Phase 1: Tuning on 2024 Validation
    print("\n==========================================================================================")
    print("  PHASE 1: Tuning & Early Stopping (2024 Validation Split)")
    print("==========================================================================================")

    x_tune_tensor = torch.tensor(X_tune_scaled, dtype=torch.float)
    best_pretrain_epochs = []
    best_finetune_epochs = []

    for s in seeds:
        set_seed(s)
        model = SparseSpatiotemporalRGCN(in_channels=len(feature_cols), hidden_channels=32, num_relations=4)
        
        # 1. Pretrain on Wildbird
        optimizer_pre = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion_wb = nn.BCEWithLogitsLoss(pos_weight=pos_weight_wb_tune)

        best_val_wb_ap = -1.0
        best_wb_ep = 0
        no_imp_wb = 0

        for ep in range(1, max_epochs + 1):
            model.train()
            optimizer_pre.zero_grad()
            out = model(x_tune_tensor, sparse_adj_list)
            loss = criterion_wb(out[train_tune_mask], y_wildbird_tensor[train_tune_mask])
            loss.backward()
            optimizer_pre.step()

            model.eval()
            with torch.no_grad():
                v_out = model(x_tune_tensor, sparse_adj_list)
                v_probs = torch.sigmoid(v_out[val_tune_mask]).cpu().numpy()
                v_ap = average_precision_score(y_wildbird_np[val_tune_mask_np], v_probs)

            if v_ap > best_val_wb_ap:
                best_val_wb_ap = v_ap
                best_wb_ep = ep
                no_imp_wb = 0
            else:
                no_imp_wb += 1

            del out, loss, v_out, v_probs
            gc.collect()

            if no_imp_wb >= patience:
                break

        best_pretrain_epochs.append(best_wb_ep)
        print(f"  Seed {s:2d} | [Pretrain WildBird] Best Epoch: {best_wb_ep} | Val PR-AUC: {best_val_wb_ap:.4f}")

        # Re-train to optimal pretrain epoch to get the pretrained encoder weights
        set_seed(s)
        model_pre = SparseSpatiotemporalRGCN(in_channels=len(feature_cols), hidden_channels=32, num_relations=4)
        opt_pre = torch.optim.Adam(model_pre.parameters(), lr=0.005, weight_decay=1e-4)
        for ep in range(1, best_wb_ep + 1):
            model_pre.train()
            opt_pre.zero_grad()
            out = model_pre(x_tune_tensor, sparse_adj_list)
            loss = criterion_wb(out[train_tune_mask], y_wildbird_tensor[train_tune_mask])
            loss.backward()
            opt_pre.step()
            del out, loss

        # Now construct fine-tune model: copy encoder weights, re-init classification head
        model_ft = SparseSpatiotemporalRGCN(in_channels=len(feature_cols), hidden_channels=32, num_relations=4)
        model_ft.load_state_dict(model_pre.state_dict())
        nn.init.xavier_uniform_(model_ft.out.weight)
        nn.init.zeros_(model_ft.out.bias)

        # 2. Fine-tune on Poultry
        optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.001, weight_decay=1e-4)
        criterion_pl = nn.BCEWithLogitsLoss(pos_weight=pos_weight_pl_tune)

        best_val_pl_ap = -1.0
        best_pl_ep = 0
        no_imp_pl = 0

        for ep in range(1, max_epochs + 1):
            model_ft.train()
            optimizer_ft.zero_grad()
            out = model_ft(x_tune_tensor, sparse_adj_list)
            loss = criterion_pl(out[train_tune_mask], y_poultry_tensor[train_tune_mask])
            loss.backward()
            optimizer_ft.step()

            model_ft.eval()
            with torch.no_grad():
                v_out = model_ft(x_tune_tensor, sparse_adj_list)
                v_probs = torch.sigmoid(v_out[val_tune_mask]).cpu().numpy()
                v_ap = average_precision_score(y_poultry_np[val_tune_mask_np], v_probs)

            if v_ap > best_val_pl_ap:
                best_val_pl_ap = v_ap
                best_pl_ep = ep
                no_imp_pl = 0
            else:
                no_imp_pl += 1

            del out, loss, v_out, v_probs
            gc.collect()

            if no_imp_pl >= patience:
                break

        best_finetune_epochs.append(best_pl_ep)
        print(f"  Seed {s:2d} | [Fine-tune Poultry] Best Epoch: {best_pl_ep} | Val PR-AUC: {best_val_pl_ap:.4f}")
        del model_pre, model_ft, optimizer_pre, optimizer_ft

    optimal_pre_epochs = int(round(np.mean(best_pretrain_epochs)))
    optimal_ft_epochs = int(round(np.mean(best_finetune_epochs)))
    print(f"Optimal Epochs: Pretrain={optimal_pre_epochs} | Fine-tune={optimal_ft_epochs}")

    # Phase 2: Full Retraining & Testing on 2025
    print("\n==========================================================================================")
    print("  PHASE 2: Retraining on All 2022-2024 Data & Testing on 2025")
    print("==========================================================================================")

    x_full_tensor = torch.tensor(X_full_scaled, dtype=torch.float)
    seed_records = []
    seed_test_probs = []

    for s in seeds:
        set_seed(s)
        model = SparseSpatiotemporalRGCN(in_channels=len(feature_cols), hidden_channels=32, num_relations=4)
        
        # Pretrain on wildbird on full training set
        opt_pre = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        crit_wb = nn.BCEWithLogitsLoss(pos_weight=pos_weight_wb_full)
        for ep in range(1, optimal_pre_epochs + 1):
            model.train()
            opt_pre.zero_grad()
            out = model(x_full_tensor, sparse_adj_list)
            loss = crit_wb(out[full_train_mask], y_wildbird_tensor[full_train_mask])
            loss.backward()
            opt_pre.step()
            del out, loss

        # Copy state and re-initialize head
        model_final = SparseSpatiotemporalRGCN(in_channels=len(feature_cols), hidden_channels=32, num_relations=4)
        model_final.load_state_dict(model.state_dict())
        nn.init.xavier_uniform_(model_final.out.weight)
        nn.init.zeros_(model_final.out.bias)

        # Fine-tune on poultry on full training set
        opt_ft = torch.optim.Adam(model_final.parameters(), lr=0.001, weight_decay=1e-4)
        crit_pl = nn.BCEWithLogitsLoss(pos_weight=pos_weight_pl_full)

        for ep in range(1, optimal_ft_epochs + 1):
            model_final.train()
            opt_ft.zero_grad()
            out = model_final(x_full_tensor, sparse_adj_list)
            loss = crit_pl(out[full_train_mask], y_poultry_tensor[full_train_mask])
            loss.backward()
            opt_ft.step()
            del out, loss

        # Evaluate on Test
        model_final.eval()
        with torch.no_grad():
            t_out = model_final(x_full_tensor, sparse_adj_list)
            t_probs = torch.sigmoid(t_out[test_mask]).cpu().numpy()

        y_test = y_poultry_np[test_mask_np]
        ap = average_precision_score(y_test, t_probs)
        roc = roc_auc_score(y_test, t_probs)
        km = topk_metrics(y_test, t_probs, [50, 100, 200], test_pos_pl)

        seed_test_probs.append(t_probs)
        seed_records.append({
            'model_name': 'Spatiotemporal Directed RGCN Transfer',
            'seed': s,
            'pr_auc': ap,
            'roc_auc': roc,
            'precision_50': km['precision_50'],
            'recall_50': km['recall_50'],
            'precision_100': km['precision_100'],
            'recall_100': km['recall_100'],
            'precision_200': km['precision_200'],
            'recall_200': km['recall_200']
        })

        print(f"  Seed {s:2d} | Test PR-AUC: {ap:.4f} | ROC-AUC: {roc:.4f} | Precision@50: {km['precision_50']:.4f}")
        del model, model_final, opt_pre, opt_ft, crit_wb, crit_pl
        gc.collect()

    df_seeds = pd.DataFrame(seed_records)
    ens_probs = np.mean(seed_test_probs, axis=0)
    ens_ap = average_precision_score(y_test, ens_probs)
    ens_roc = roc_auc_score(y_test, ens_probs)
    ens_km = topk_metrics(y_test, ens_probs, [50, 100, 200], test_pos_pl)

    summary_records = [
        {
            'model_variant': 'Spatiotemporal Directed RGCN Transfer (Multiseed Mean)',
            'pr_auc': round(df_seeds['pr_auc'].mean(), 6),
            'pr_auc_std': round(df_seeds['pr_auc'].std(), 6),
            'roc_auc': round(df_seeds['roc_auc'].mean(), 6),
            'precision_50': round(df_seeds['precision_50'].mean(), 4),
            'recall_50': round(df_seeds['recall_50'].mean(), 4),
            'precision_100': round(df_seeds['precision_100'].mean(), 4),
            'recall_100': round(df_seeds['recall_100'].mean(), 4),
            'precision_200': round(df_seeds['precision_200'].mean(), 4),
            'recall_200': round(df_seeds['recall_200'].mean(), 4)
        },
        {
            'model_variant': 'Spatiotemporal Directed RGCN Transfer (3-Seed Ensemble)',
            'pr_auc': round(ens_ap, 6),
            'pr_auc_std': 0.0,
            'roc_auc': round(ens_roc, 6),
            'precision_50': round(ens_km['precision_50'], 4),
            'recall_50': round(ens_km['recall_50'], 4),
            'precision_100': round(ens_km['precision_100'], 4),
            'recall_100': round(ens_km['recall_100'], 4),
            'precision_200': round(ens_km['precision_200'], 4),
            'recall_200': round(ens_km['recall_200'], 4)
        }
    ]

    df_summary = pd.DataFrame(summary_records)

    test_nodes_df = df_model[test_mask_np].copy()
    preds_df = pd.DataFrame({
        'csd_uid': test_nodes_df['csd_uid'].values,
        'csd_name': test_nodes_df['csd_name'].values,
        'province_code': test_nodes_df['province_code'].values,
        'year': test_nodes_df['year'].values,
        'month': test_nodes_df['month'].values,
        'actual': y_test.astype(int),
        'temporal_gnn_transfer_prob': ens_probs
    })

    os.makedirs(BASELINES_DIR, exist_ok=True)
    df_seeds.to_csv(MULTI_METRICS_OUT, index=False)
    df_summary.to_csv(SUMMARY_OUT, index=False)
    preds_df.to_csv(PREDS_OUT, index=False)

    print("\n==========================================================================================")
    print("      SPATIOTEMPORAL DIRECTED RELATIONAL GNN TRANSFER RESULTS (2025 TEST SPLIT)")
    print("==========================================================================================")
    print(df_summary.to_string(index=False))
    print("==========================================================================================")

    # Bootstrap Evaluation
    print("\n--- Running 1,000 CSD Cluster-Bootstrap Uncertainty Analysis vs Baselines ---")
    if not os.path.exists(STRICT_LR_PREDS_PATH) or not os.path.exists(KNN3_PREDS_PATH):
        print(f"Warning: Baseline prediction files missing. Skipping bootstrap comparison.")
        return

    df_lr = pd.read_csv(STRICT_LR_PREDS_PATH, dtype={'csd_uid': str})
    df_knn3 = pd.read_csv(KNN3_PREDS_PATH, dtype={'csd_uid': str})

    if 'knn3_graphsage_prob' not in df_knn3.columns:
        knn3_prob_cols = [c for c in df_knn3.columns if 'prob' in c]
        df_knn3['knn3_graphsage_prob'] = df_knn3[knn3_prob_cols].mean(axis=1)

    m1 = pd.merge(preds_df, df_lr[['csd_uid', 'year', 'month', 'strict_env_lag_logistic_prob']], on=['csd_uid', 'year', 'month'], how='inner')
    m2 = pd.merge(m1, df_knn3[['csd_uid', 'year', 'month', 'knn3_graphsage_prob']], on=['csd_uid', 'year', 'month'], how='inner')

    y_true_boot = m2['actual'].values
    p_tgnn = m2['temporal_gnn_transfer_prob'].values
    p_lr = m2['strict_env_lag_logistic_prob'].values
    p_knn3 = m2['knn3_graphsage_prob'].values

    csd_ids = m2['csd_uid'].values
    unique_csds = np.unique(csd_ids)
    n_csds = len(unique_csds)

    tgnn_ap_obs = average_precision_score(y_true_boot, p_tgnn)
    lr_ap_obs = average_precision_score(y_true_boot, p_lr)
    knn3_ap_obs = average_precision_score(y_true_boot, p_knn3)

    diff_lr_boot = []
    diff_knn3_boot = []

    np.random.seed(42)
    for _ in range(1000):
        boot_csds = np.random.choice(unique_csds, size=n_csds, replace=True)
        boot_indices = []
        for csd in boot_csds:
            boot_indices.extend(np.where(csd_ids == csd)[0])
        boot_indices = np.array(boot_indices)

        y_b = y_true_boot[boot_indices]
        if y_b.sum() == 0:
            continue

        ap_t = average_precision_score(y_b, p_tgnn[boot_indices])
        ap_l = average_precision_score(y_b, p_lr[boot_indices])
        ap_k = average_precision_score(y_b, p_knn3[boot_indices])

        diff_lr_boot.append(ap_t - ap_l)
        diff_knn3_boot.append(ap_t - ap_k)

    ci_lr = np.percentile(diff_lr_boot, [2.5, 97.5])
    ci_knn3 = np.percentile(diff_knn3_boot, [2.5, 97.5])

    frac_lr = np.mean(np.array(diff_lr_boot) > 0)
    frac_knn3 = np.mean(np.array(diff_knn3_boot) > 0)

    dec_lr = "statistically better" if ci_lr[0] > 0 else "statistically worse" if ci_lr[1] < 0 else "comparable, not proven better"
    dec_knn3 = "statistically better" if ci_knn3[0] > 0 else "statistically worse" if ci_knn3[1] < 0 else "comparable, not proven better"

    boot_records = [
        {
            'comparison_baseline': 'Strict Environment + Lag Logistic Regression',
            'temporal_gnn_pr_auc': round(tgnn_ap_obs, 4),
            'baseline_pr_auc': round(lr_ap_obs, 4),
            'observed_pr_auc_diff': round(tgnn_ap_obs - lr_ap_obs, 4),
            'ci_95_lower': round(ci_lr[0], 4),
            'ci_95_upper': round(ci_lr[1], 4),
            'resample_fraction_tgnn_better': round(frac_lr, 4),
            'decision_conclusion': dec_lr
        },
        {
            'comparison_baseline': 'kNN-3 GraphSAGE Ensemble',
            'temporal_gnn_pr_auc': round(tgnn_ap_obs, 4),
            'baseline_pr_auc': round(knn3_ap_obs, 4),
            'observed_pr_auc_diff': round(tgnn_ap_obs - knn3_ap_obs, 4),
            'ci_95_lower': round(ci_knn3[0], 4),
            'ci_95_upper': round(ci_knn3[1], 4),
            'resample_fraction_tgnn_better': round(frac_knn3, 4),
            'decision_conclusion': dec_knn3
        }
    ]

    df_boot = pd.DataFrame(boot_records)
    df_boot.to_csv(BOOTSTRAP_CSV_OUT, index=False)

    md_content = rf"""# True Spatiotemporal Directed Relational GNN Transfer Bootstrap Comparison

## Pairwise Bootstrap Uncertainty Results (1,000 Resamples on 2025 Test Set)

| Comparison Baseline | Temporal GNN Transfer PR-AUC | Baseline PR-AUC | Observed Diff | 95% Percentile Confidence Interval | Resample Fraction (TGNN Better) | Decision Conclusion |
|---|---|---|---|---|---|---|
| **Strict Env + Lag LR** | **{tgnn_ap_obs:.4f}** | {lr_ap_obs:.4f} | **{tgnn_ap_obs - lr_ap_obs:+.4f}** | **[{ci_lr[0]:+.4f}, {ci_lr[1]:+.4f}]** | {frac_lr:.4f} ({frac_lr:.1%}) | **{dec_lr}** |
| **kNN-3 GraphSAGE Ensemble** | **{tgnn_ap_obs:.4f}** | {knn3_ap_obs:.4f} | **{tgnn_ap_obs - knn3_ap_obs:+.4f}** | **[{ci_knn3[0]:+.4f}, {ci_knn3[1]:+.4f}]** | {frac_knn3:.4f} ({frac_knn3:.1%}) | **{dec_knn3}** |

---

## Technical Notes
- **Graph Design**: 237,538 CSD-month nodes ($5,054 \times 47$). Directed forward temporal edges ($t \rightarrow t+1$) across 4 relation types.
- **Transfer Learning**: Pretrained on wild-bird HPAI next-month targets prior to fine-tuning on poultry.
"""

    with open(BOOTSTRAP_MD_OUT, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print("\n==========================================================================================")
    print("                BOOTSTRAP UNCERTAINTY COMPARISON RESULTS")
    print("==========================================================================================")
    print(df_boot.to_string(index=False))
    print("==========================================================================================")

if __name__ == '__main__':
    main()
