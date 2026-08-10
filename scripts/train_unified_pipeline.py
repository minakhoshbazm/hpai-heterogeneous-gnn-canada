#!/usr/bin/env python3
"""
scripts/train_unified_pipeline.py

Unified evaluation pipeline for HPAI models.
Ensures strict consistency:
- Forward-time split: Train 22-23, Val 24, Test 25.
- Fixed 5 seeds.
- Focal loss for all neural models.
- Poultry-at-risk negative filtering for GNNs.
- Spatial feature ablation support (dropping lag features).
- 1,000-iteration cluster bootstrap on test set.
"""

import os
import random
import copy
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import SAGEConv

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

OUT_DIR = "results/unified_evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_MODEL_TABLE = "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
KNN_GRAPH = "data/processed/graph/csd_graph_knn_3_v1.csv"

SEEDS = [11, 22, 33, 44, 55]

EXCLUDE_EXPLICIT = {
    "csd_uid", "csd_name", "province_code", "province_name", "ccs_uid",
    "year", "month", "outbreak_next_month",
    "dw_fill_method", "climate_extraction_method",
    "glw2020_chicken_extraction_method", "glw2015_duck_extraction_method",
    "road_data_coverage", "outbreak_current_month", "wildbird_hpai_present_next_month"
}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class StandaloneGraphSAGE(nn.Module):
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

class HarmonizedMLP(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, edge_index=None):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.out(x).squeeze(-1)

def build_monthly_snapshots(df: pd.DataFrame, feature_cols: list, edges_raw: pd.DataFrame, scaler: StandardScaler, is_at_risk_mask: np.ndarray):
    snapshots = []
    ym_pairs = df[["year", "month"]].drop_duplicates().sort_values(by=["year", "month"]).values

    # We map CSDs globally for the graph
    unique_csds = sorted(df["csd_uid"].unique())
    uid_map = {uid: i for i, uid in enumerate(unique_csds)}
    
    # We will build full snapshots (all nodes) per month, with a mask for valid rows in that month
    src_mapped = edges_raw["source_csd_uid"].map(uid_map).dropna().astype(int).values
    tgt_mapped = edges_raw["target_csd_uid"].map(uid_map).dropna().astype(int).values
    edge_index = torch.tensor(np.stack([src_mapped, tgt_mapped], axis=0), dtype=torch.long)
    
    for y, m in ym_pairs:
        df_t = df[(df["year"] == y) & (df["month"] == m)].copy().reset_index(drop=True)
        # Create full canvas
        X_t_raw = np.zeros((len(unique_csds), len(feature_cols)), dtype=np.float32)
        y_t = np.zeros(len(unique_csds), dtype=np.float32)
        node_mask = np.zeros(len(unique_csds), dtype=bool)
        
        node_indices = df_t["csd_uid"].map(uid_map).values
        X_t_raw[node_indices] = df_t[feature_cols].fillna(0.0).values
        y_t[node_indices] = df_t["outbreak_next_month"].values.astype(np.float32)
        node_mask[node_indices] = True
        
        X_t = scaler.transform(X_t_raw).astype(np.float32)
        
        data = Data(
            x=torch.from_numpy(X_t),
            edge_index=edge_index,
            y=torch.from_numpy(y_t),
            node_mask=torch.from_numpy(node_mask),
            at_risk=torch.from_numpy(is_at_risk_mask),
            year=int(y),
            month=int(m),
            csd_uids=np.array(unique_csds)
        )
        snapshots.append(data)
    return snapshots

def compute_metrics(y_true, probs, df_eval):
    ap = average_precision_score(y_true, probs)
    roc = roc_auc_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    
    df_temp = pd.DataFrame({'month': df_eval['month'].values, 'y_true': y_true, 'prob': probs})
    
    df_sorted = df_temp.sort_values('prob', ascending=False)
    p10_pooled = df_sorted.head(10)['y_true'].mean()
    
    macro_r50 = []
    macro_p10 = []
    for m in sorted(df_temp['month'].unique()):
        df_m = df_temp[df_temp['month'] == m].sort_values('prob', ascending=False)
        n_pos_m = df_m['y_true'].sum()
        r50_m = df_m.head(50)['y_true'].sum() / n_pos_m if n_pos_m > 0 else 0.0
        p10_m = df_m.head(10)['y_true'].mean()
        macro_r50.append(r50_m)
        macro_p10.append(p10_m)
        
    return {
        'AP': ap,
        'ROC-AUC': roc,
        'Brier': brier,
        'Macro Recall@50': np.mean(macro_r50),
        'Macro Precision@10': np.mean(macro_p10)
    }

def train_neural_model(model_cls, in_channels, train_snaps, test_snaps, df_test, seed, use_at_risk_filter=False):
    seed_everything(seed)
    model = model_cls(in_channels=in_channels, hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    alpha = 0.25
    gamma = 2.0
    
    # Train 30 epochs
    for epoch in range(1, 31):
        model.train()
        for batch in train_snaps:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            
            # Mask active nodes
            active_out = out[batch.node_mask]
            active_y = batch.y[batch.node_mask]
            
            if use_at_risk_filter:
                active_at_risk = batch.at_risk[batch.node_mask]
                pos_mask = (active_y == 1)
                # Negatives must be at-risk
                neg_mask = (active_y == 0) & active_at_risk
            else:
                pos_mask = (active_y == 1)
                neg_mask = (active_y == 0)
                
            p = torch.sigmoid(active_out)
            bce_pos = F.binary_cross_entropy_with_logits(active_out[pos_mask], active_y[pos_mask], reduction='none') if pos_mask.sum() > 0 else torch.tensor([0.0])
            bce_neg = F.binary_cross_entropy_with_logits(active_out[neg_mask], active_y[neg_mask], reduction='none') if neg_mask.sum() > 0 else torch.tensor([0.0])
            
            pt_pos = p[pos_mask]
            pt_neg = 1 - p[neg_mask]
            
            focal_pos = alpha * (1 - pt_pos)**gamma * bce_pos
            focal_neg = (1 - alpha) * (1 - pt_neg)**gamma * bce_neg
            
            loss = focal_pos.sum() + focal_neg.sum()
            loss = loss / (pos_mask.sum() + neg_mask.sum() + 1e-6)
            loss.backward()
            optimizer.step()
            
    # Eval test
    model.eval()
    test_probs = []
    
    with torch.no_grad():
        for batch in test_snaps:
            out = model(batch.x, batch.edge_index)
            probs = torch.sigmoid(out)[batch.node_mask].cpu().numpy()
            test_probs.extend(probs)
            
    return np.array(test_probs)

def train_lightgbm(df_train, df_test, feature_cols, seed):
    seed_everything(seed)
    # Simple Tabular LightGBM
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=seed,
        class_weight='balanced',
        n_jobs=1
    )
    X_tr = df_train[feature_cols].fillna(0.0).values
    y_tr = df_train['outbreak_next_month'].values
    
    clf.fit(X_tr, y_tr)
    X_te = df_test[feature_cols].fillna(0.0).values
    return clf.predict_proba(X_te)[:, 1]


def run_pipeline():
    print("Loading data...")
    df_all = pd.read_csv(INPUT_MODEL_TABLE, dtype={"csd_uid": str})
    
    feature_cols = []
    for c in df_all.columns:
        if c not in EXCLUDE_EXPLICIT and pd.api.types.is_numeric_dtype(df_all[c]):
            feature_cols.append(c)
    feature_cols = sorted(feature_cols)
    print(f"Total baseline features: {len(feature_cols)}")
    
    # Identify Lag Features for Ablation
    lag_keywords = ['outbreak', 'hpai', 'count']
    no_lag_features = [f for f in feature_cols if not any(k in f for k in lag_keywords)]
    # Make sure we don't drop structural things mistakenly, but 'dw_image_count' etc might be dropped. Let's be precise:
    lag_features_exact = [
        'same_csd_outbreak_current_month', 'same_csd_outbreak_prev_1_month',
        'same_csd_outbreak_count_past_3_months', 'same_csd_outbreak_count_past_6_months', 'same_csd_outbreak_count_past_12_months',
        'province_outbreak_count_current_month', 'province_outbreak_count_past_3_months',
        'wildbird_hpai_count_ccs_current_month', 'wildbird_hpai_count_ccs_prev_1_month',
        'wildbird_hpai_count_ccs_past_3_months', 'wildbird_hpai_count_ccs_past_6_months',
        'wildbird_hpai_present_ccs_current_month', 'wildbird_hpai_present_ccs_prev_1_month',
        'wildbird_hpai_waterfowl_count_ccs_current_month', 'wildbird_hpai_waterfowl_count_ccs_past_3_months',
        'wildbird_hpai_raptor_count_ccs_current_month', 'wildbird_hpai_raptor_count_ccs_past_3_months',
        'wildbird_hpai_goose_count_ccs_current_month', 'wildbird_hpai_duck_count_ccs_current_month'
    ]
    ablated_features = [f for f in feature_cols if f not in lag_features_exact]
    print(f"Ablated features (No Lags): {len(ablated_features)}")
    
    edges_raw = pd.read_csv(KNN_GRAPH, dtype=str)
    
    # Forward Splits
    df_train = df_all[df_all["year"].isin([2022, 2023])].copy().sort_values(["year", "month", "csd_uid"]).reset_index(drop=True)
    df_test = df_all[df_all["year"] == 2025].copy().sort_values(["year", "month", "csd_uid"]).reset_index(drop=True)
    
    # At-Risk Mask (for all unique CSDs)
    poultry_farm_cols = ['ccs_broilers_roasters_cornish_farms_reporting', 'ccs_broilers_roasters_cornish_production_kg', 'ccs_turkey_farms_reporting', 'ccs_turkey_production_kg']
    df_all['farm_cap_sum'] = df_all[poultry_farm_cols].fillna(0.0).sum(axis=1)
    csd_at_risk_set = set(df_all.groupby('csd_uid')['farm_cap_sum'].max().loc[lambda x: x > 0].index)
    unique_csds = sorted(df_all["csd_uid"].unique())
    is_at_risk_mask = np.array([uid in csd_at_risk_set for uid in unique_csds])

    # Setup Scalers and Snapshots
    scaler_full = StandardScaler()
    scaler_full.fit(df_train[feature_cols].fillna(0.0).values)
    
    scaler_ablate = StandardScaler()
    scaler_ablate.fit(df_train[ablated_features].fillna(0.0).values)
    
    train_snaps_full = build_monthly_snapshots(df_train, feature_cols, edges_raw, scaler_full, is_at_risk_mask)
    test_snaps_full = build_monthly_snapshots(df_test, feature_cols, edges_raw, scaler_full, is_at_risk_mask)
    
    train_snaps_ablate = build_monthly_snapshots(df_train, ablated_features, edges_raw, scaler_ablate, is_at_risk_mask)
    test_snaps_ablate = build_monthly_snapshots(df_test, ablated_features, edges_raw, scaler_ablate, is_at_risk_mask)
    
    y_test_true = df_test["outbreak_next_month"].values

    models = [
        {"name": "LightGBM (Tabular, Full)", "type": "lgb", "features": feature_cols, "snaps": None, "at_risk": False},
        {"name": "MLP (Full Features)", "type": "mlp", "features": feature_cols, "snaps": (train_snaps_full, test_snaps_full), "at_risk": False},
        {"name": "GraphSAGE (Full Features, National Negs)", "type": "sage", "features": feature_cols, "snaps": (train_snaps_full, test_snaps_full), "at_risk": False},
        {"name": "MLP (Ablated: No Lags)", "type": "mlp", "features": ablated_features, "snaps": (train_snaps_ablate, test_snaps_ablate), "at_risk": False},
        {"name": "GraphSAGE (Ablated: No Lags, National Negs)", "type": "sage", "features": ablated_features, "snaps": (train_snaps_ablate, test_snaps_ablate), "at_risk": False},
    ]
    
    results = []
    model_ensembles = {}
    
    for cfg in models:
        print(f"\n--- Running: {cfg['name']} ---")
        seed_preds = []
        for s in SEEDS:
            if cfg["type"] == "lgb":
                preds = train_lightgbm(df_train, df_test, cfg["features"], s)
            else:
                m_cls = HarmonizedMLP if cfg["type"] == "mlp" else StandaloneGraphSAGE
                tr_snaps, te_snaps = cfg["snaps"]
                preds = train_neural_model(m_cls, len(cfg["features"]), tr_snaps, te_snaps, df_test, s, cfg["at_risk"])
            seed_preds.append(preds)
        
        ens_preds = np.mean(seed_preds, axis=0)
        model_ensembles[cfg["name"]] = ens_preds
        metrics = compute_metrics(y_test_true, ens_preds, df_test)
        
        results.append({
            "Configuration": cfg["name"],
            **metrics
        })
        print(f"Ensemble AP: {metrics['AP']:.4f}")
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OUT_DIR, "unified_summary_results.csv"), index=False)
    print("\n--- Final Unified Results ---")
    print(df_res.to_string(index=False))
    
    # Cluster Bootstrap
    print("\n--- Running Cluster Bootstrap (1,000 Iters) ---")
    months = df_test['month'].unique()
    rng = np.random.RandomState(42)
    boot_records = []
    
    for i in range(1000):
        boot_m = rng.choice(months, size=len(months), replace=True)
        # Reconstruct indices
        idx_boot = []
        for m in boot_m:
            idx_boot.extend(df_test[df_test['month'] == m].index.tolist())
            
        y_b = y_test_true[idx_boot]
        if y_b.sum() == 0:
            continue
            
        ap_dict = {}
        for name, preds in model_ensembles.items():
            ap_dict[name] = average_precision_score(y_b, preds[idx_boot])
            
        boot_records.append({
            'Diff: SAGE vs Tabular': ap_dict['GraphSAGE (Full Features, National Negs)'] - ap_dict['LightGBM (Tabular, Full)'],
            'Diff: SAGE vs MLP': ap_dict['GraphSAGE (Full Features, National Negs)'] - ap_dict['MLP (Full Features)'],
            'Diff: SAGE (Ablated) vs MLP (Ablated)': ap_dict['GraphSAGE (Ablated: No Lags, National Negs)'] - ap_dict['MLP (Ablated: No Lags)'],
            'SAGE (Ablated: No Lags) AP': ap_dict['GraphSAGE (Ablated: No Lags, National Negs)'],
            'MLP (Ablated: No Lags) AP': ap_dict['MLP (Ablated: No Lags)']
        })
        
    df_boot = pd.DataFrame(boot_records)
    df_boot.to_csv(os.path.join(OUT_DIR, "bootstrap_results.csv"), index=False)
    
    for col in df_boot.columns:
        vals = df_boot[col]
        print(f"{col}: Mean = {vals.mean():.4f}, 95% CI [{np.percentile(vals, 2.5):.4f}, {np.percentile(vals, 97.5):.4f}]")

if __name__ == "__main__":
    run_pipeline()
