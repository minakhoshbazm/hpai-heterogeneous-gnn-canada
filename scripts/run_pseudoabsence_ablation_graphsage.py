#!/usr/bin/env python3
"""
scripts/run_pseudoabsence_ablation_graphsage.py

Negative-Sampling Ablation for the standalone GraphSAGE model.
Tests 6 configurations of negative sampling during training:
1. all_national (baseline)
2. all_poultry
3. ratio_50
4. ratio_25
5. ratio_10
6. ratio_3

Evaluates on the locked 2025 full-prevalence test set.
"""

import os
import random
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import SAGEConv

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

INPUT_MODEL_TABLE = "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
KNN3_GRAPH = "data/processed/graph/csd_graph_knn_3_v1.csv"
OUT_CSV = "results/pseudoabsence_ablation/graphsage_ablation_results.csv"

SEEDS = [11, 22, 33, 44, 55]
EPOCHS = 30

POULTRY_EXPOSURE_COLS = [
    'ccs_broilers_roasters_cornish_farms_reporting',
    'ccs_broilers_roasters_cornish_production_kg',
    'ccs_turkey_farms_reporting', 'ccs_turkey_production_kg',
    'glw2020_chicken_density_mean', 'glw2020_chicken_density_max',
    'glw2015_duck_density_mean', 'glw2015_duck_density_max'
]

WATER_HABITAT_COLS = [
    'dw_water_mean', 'dw_water_p95',
    'dw_flooded_vegetation_mean', 'dw_seasonal_pond_proxy'
]

EXCLUDE_EXPLICIT = {
    "csd_uid", "csd_name", "province_code", "province_name", "ccs_uid",
    "year", "month", "outbreak_next_month",
    "dw_fill_method", "climate_extraction_method",
    "glw2020_chicken_extraction_method", "glw2015_duck_extraction_method",
    "road_data_coverage", "outbreak_current_month",
}

CONFIGS = {
    "all_national": None,  # no sampling, use all negs
    "all_poultry": "all",  # use all poultry negs
    "ratio_50": 50,
    "ratio_25": 25,
    "ratio_10": 10,
    "ratio_3": 3
}

def get_62_features(df):
    cols = [c for c in df.columns if c not in EXCLUDE_EXPLICIT]
    return sorted(cols)

class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.out(x).squeeze(-1)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_monthly_rare_event_metrics(df_res):
    precisions_10 = []
    recalls_50 = []
    for (y, m), group in df_res.groupby(["year", "month"]):
        if group["true_label"].sum() > 0:
            g_sorted = group.sort_values("pred_prob", ascending=False)
            top10 = g_sorted.head(10)
            top50 = g_sorted.head(50)
            p10 = top10["true_label"].sum() / 10.0
            r50 = top50["true_label"].sum() / group["true_label"].sum()
            precisions_10.append(p10)
            recalls_50.append(r50)
    return np.mean(precisions_10) if precisions_10 else 0.0, np.mean(recalls_50) if recalls_50 else 0.0

def build_monthly_snapshots(df: pd.DataFrame, feature_cols: list, edges_raw: pd.DataFrame, scaler: StandardScaler):
    snapshots = []
    ym_pairs = df[["year", "month"]].drop_duplicates().sort_values(by=["year", "month"]).values
    
    # Precompute neighbor dict for fast kNN lookup
    global_neighbor_map = {}
    for _, row in edges_raw.iterrows():
        global_neighbor_map.setdefault(row['source_csd_uid'], set()).add(row['target_csd_uid'])
        global_neighbor_map.setdefault(row['target_csd_uid'], set()).add(row['source_csd_uid'])

    for y, m in ym_pairs:
        df_t = df[(df["year"] == y) & (df["month"] == m)].copy().reset_index(drop=True)
        
        X_t_raw = df_t[feature_cols].fillna(0.0).values
        X_t = scaler.transform(X_t_raw).astype(np.float32)
        y_t = df_t["outbreak_next_month"].values.astype(np.float32)

        uid_map = {uid: i for i, uid in enumerate(df_t["csd_uid"].values)}
        src_mapped = edges_raw["source_csd_uid"].map(uid_map).dropna().astype(int).values
        tgt_mapped = edges_raw["target_csd_uid"].map(uid_map).dropna().astype(int).values

        edge_index = torch.tensor(np.stack([src_mapped, tgt_mapped], axis=0), dtype=torch.long)
        x_tensor = torch.from_numpy(X_t)
        y_tensor = torch.from_numpy(y_t)
        
        poultry_mask = (df_t[POULTRY_EXPOSURE_COLS].sum(axis=1) > 0).values
        water_features = df_t[WATER_HABITAT_COLS].fillna(0.0).values
        
        # Build local neighbor map for this month's indices
        local_neighbors = {}
        for csd_uid, idx in uid_map.items():
            neighbors = global_neighbor_map.get(csd_uid, set())
            idx_neighbors = [uid_map[n] for n in neighbors if n in uid_map]
            local_neighbors[idx] = idx_neighbors

        data = Data(
            x=x_tensor, 
            edge_index=edge_index, 
            y=y_tensor, 
            year=int(y), 
            month=int(m), 
            csd_uids=df_t["csd_uid"].values,
            poultry_mask=torch.from_numpy(poultry_mask),
            water_features=torch.from_numpy(water_features.astype(np.float32)),
            local_neighbors=[local_neighbors.get(i, []) for i in range(len(df_t))]
        )
        snapshots.append(data)
        
    return snapshots

def sample_negatives(batch, config_val):
    """
    Returns boolean tensor mask of length num_nodes indicating which negatives to include,
    and a weight tensor of length num_nodes indicating the IPW weight for those negatives.
    """
    num_nodes = batch.y.size(0)
    neg_mask = (batch.y == 0)
    pos_mask = (batch.y == 1)
    
    weights = torch.ones(num_nodes, dtype=torch.float32)
    final_loss_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    if config_val is None:
        # all_national
        final_loss_mask[neg_mask] = True
        return final_loss_mask, weights
        
    # Poultry restricted
    poultry_neg_mask = neg_mask & batch.poultry_mask
    poultry_neg_idx = poultry_neg_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    pos_idx = pos_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    
    if config_val == "all" or len(pos_idx) == 0 or len(poultry_neg_idx) == 0:
        # include all poultry negatives
        final_loss_mask[poultry_neg_idx] = True
        return final_loss_mask, weights
        
    # Sample based on ratio
    ratio = config_val
    n_target = min(ratio * len(pos_idx), len(poultry_neg_idx))
    
    if n_target <= 0:
        return final_loss_mask, weights
        
    n_spatial = n_target // 3
    n_env = n_target // 3
    
    sampled = set()
    
    # 1. Spatial kNN-3 Neighbors of Positives
    spatial_candidates = set()
    # PyG DataLoader collates custom lists by wrapping them in an outer list
    node_neighbors = batch.local_neighbors[0] if len(batch.local_neighbors) == 1 and isinstance(batch.local_neighbors[0], list) else batch.local_neighbors
    for p in pos_idx:
        for n_idx in node_neighbors[p]:
            if n_idx in poultry_neg_idx:
                spatial_candidates.add(n_idx)
    
    spatial_candidates -= sampled
    if len(spatial_candidates) > 0:
        n_spatial_actual = min(n_spatial, len(spatial_candidates))
        sampled.update(np.random.choice(list(spatial_candidates), n_spatial_actual, replace=False))
        
    # 2. Environmental Matching
    env_candidates = set()
    if len(pos_idx) > 0 and len(poultry_neg_idx) > 0:
        pos_water = batch.water_features[pos_idx].cpu().numpy()
        neg_water = batch.water_features[poultry_neg_idx].cpu().numpy()
        
        # We want highest cosine similarity. cosine distance = 1 - sim.
        dist = cdist(neg_water, pos_water, metric='cosine')
        min_dist = dist.min(axis=1) # min distance -> max similarity
        
        n_env_target = min(n_env, len(poultry_neg_idx))
        top_env_idx = np.argsort(min_dist)[:n_env_target*2]
        for idx in top_env_idx:
            env_candidates.add(poultry_neg_idx[idx])
            
    env_candidates -= sampled
    if len(env_candidates) > 0:
        n_env_actual = min(n_env, len(env_candidates))
        sampled.update(np.random.choice(list(env_candidates), n_env_actual, replace=False))
        
    # 3. Random
    remaining = n_target - len(sampled)
    random_pool = list(set(poultry_neg_idx) - sampled)
    if remaining > 0 and len(random_pool) > 0:
        n_random_actual = min(remaining, len(random_pool))
        sampled.update(np.random.choice(random_pool, n_random_actual, replace=False))
        
    sampled_idx = np.array(list(sampled))
    final_loss_mask[sampled_idx] = True
    
    # IPW Calculation
    # We sampled N negatives out of N_total poultry negatives.
    # IPW weight for the sampled negatives = N_total / N
    ipw = float(len(poultry_neg_idx)) / float(len(sampled_idx))
    weights[sampled_idx] = ipw
    
    return final_loss_mask, weights

def main():
    print("=== GraphSAGE Negative Sampling Ablation ===")
    
    df_all = pd.read_csv(INPUT_MODEL_TABLE, dtype={"csd_uid": str})
    
    df_train = df_all[df_all["year"] <= 2024].copy()
    df_test = df_all[df_all["year"] == 2025].copy()
    
    print(f"Train size: {len(df_train)}")
    print(f"Test size: {len(df_test)}")
    
    feature_cols = get_62_features(df_all)
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].fillna(0.0).values)
    
    edges_raw = pd.read_csv(KNN3_GRAPH, dtype=str)
    
    train_snaps = build_monthly_snapshots(df_train, feature_cols, edges_raw, scaler)
    test_snaps = build_monthly_snapshots(df_test, feature_cols, edges_raw, scaler)
    
    train_loader = PyGDataLoader(train_snaps, batch_size=1, shuffle=True)
    test_loader = PyGDataLoader(test_snaps, batch_size=1, shuffle=False)
    
    results_summary = []
    os.makedirs("results/pseudoabsence_ablation", exist_ok=True)
    
    for config_name, config_val in CONFIGS.items():
        print(f"\n--- Evaluating Config: {config_name} ---")
        
        seed_predictions = []
        for seed in SEEDS:
            seed_everything(seed)
            model = GraphSAGEModel(in_channels=62, hidden=64)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            
            # Focal Loss parameters
            alpha = 0.25
            gamma = 2.0
            
            for epoch in range(1, EPOCHS + 1):
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index)
                    
                    pos_mask = (batch.y == 1)
                    neg_loss_mask, ipw_weights = sample_negatives(batch, config_val)
                    
                    p = torch.sigmoid(out)
                    
                    bce_pos = F.binary_cross_entropy_with_logits(out[pos_mask], batch.y[pos_mask], reduction='none') if pos_mask.sum() > 0 else torch.tensor([0.0])
                    bce_neg = F.binary_cross_entropy_with_logits(out[neg_loss_mask], batch.y[neg_loss_mask], reduction='none') if neg_loss_mask.sum() > 0 else torch.tensor([0.0])
                    
                    pt_pos = p[pos_mask]
                    pt_neg = 1 - p[neg_loss_mask]
                    
                    focal_pos = alpha * (1 - pt_pos)**gamma * bce_pos
                    focal_neg = (1 - alpha) * (1 - pt_neg)**gamma * bce_neg
                    
                    # Apply IPW weights to negative focal loss
                    focal_neg = focal_neg * ipw_weights[neg_loss_mask].to(focal_neg.device)
                    
                    loss = focal_pos.sum() + focal_neg.sum()
                    loss = loss / (pos_mask.sum() + neg_loss_mask.sum() + 1e-6)
                    
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            all_preds, all_y, all_years, all_months, all_csds = [], [], [], [], []
            with torch.no_grad():
                for batch in test_loader:
                    out = model(batch.x, batch.edge_index)
                    preds = torch.sigmoid(out).cpu().numpy()
                    all_preds.extend(preds)
                    all_y.extend(batch.y.cpu().numpy())
                    
                    num_nodes = len(preds)
                    y_val = int(batch.year[0].item()) if hasattr(batch.year, '__len__') else int(batch.year)
                    m_val = int(batch.month[0].item()) if hasattr(batch.month, '__len__') else int(batch.month)
                    
                    all_years.extend([y_val] * num_nodes)
                    all_months.extend([m_val] * num_nodes)
                    
                    flat_csds = []
                    for item in batch.csd_uids:
                        if isinstance(item, (list, np.ndarray, tuple)):
                            flat_csds.extend(item)
                        else:
                            flat_csds.append(item)
                            
                    if len(flat_csds) > num_nodes:
                        flat_csds = flat_csds[:num_nodes]
                    elif len(flat_csds) < num_nodes:
                        flat_csds.extend(["unknown"] * (num_nodes - len(flat_csds)))
                        
                    all_csds.extend(flat_csds)
            
            df_res = pd.DataFrame({
                "year": all_years,
                "month": all_months,
                "csd_uid": all_csds,
                "true_label": all_y,
                f"pred_prob_{seed}": all_preds
            })
            seed_predictions.append(df_res)
            
        df_ens = seed_predictions[0].copy()
        pred_cols = []
        for s_idx, df_s in enumerate(seed_predictions):
            col_name = f"pred_prob_{SEEDS[s_idx]}"
            df_ens[col_name] = df_s[col_name]
            pred_cols.append(col_name)
            
        df_ens["pred_prob"] = df_ens[pred_cols].mean(axis=1)
        
        ap = average_precision_score(df_ens["true_label"], df_ens["pred_prob"])
        roc_auc = roc_auc_score(df_ens["true_label"], df_ens["pred_prob"])
        p10, r50 = compute_monthly_rare_event_metrics(df_ens)
        
        print(f"  {config_name} -> AP: {ap:.4f} | ROC-AUC: {roc_auc:.4f} | R@50: {r50:.4f}")
        
        results_summary.append({
            "Configuration": config_name,
            "AP (PR-AUC)": ap,
            "ROC-AUC": roc_auc,
            "Recall@50": r50,
            "Precision@10": p10
        })
        
    df_sum = pd.DataFrame(results_summary)
    print("\n=== Final Ablation Table ===")
    print(df_sum.to_markdown(index=False))
    df_sum.to_csv(OUT_CSV, index=False)
    print(f"Results saved to {OUT_CSV}")

if __name__ == '__main__':
    main()
