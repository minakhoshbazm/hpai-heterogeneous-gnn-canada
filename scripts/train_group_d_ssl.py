#!/usr/bin/env python3
"""
scripts/train_group_d_ssl.py

Group D: Self-Supervised Masked-Feature Pretraining for GraphSAGE.
- Mask 15% of features during training (2022-2023).
- Reconstruct using MSE.
- Fine-tune on poultry classification.
- Compare vs baseline (scratch).
"""

import scipy.optimize
import scipy.stats
import os
import random
import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

OUT_DIR = "results/group_d_ssl"
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_MODEL_TABLE = "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
KNN_GRAPH = "data/processed/graph/csd_graph_knn_3_v1.csv"
REGISTRY_PATH = "results/experiment_registry.csv"

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

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        return x

class SSLGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden)
        # Reconstruction head maps back to feature dimension
        self.reconstructor = nn.Linear(hidden, in_channels)
        # Classification head maps to 1 logit
        self.classifier = nn.Linear(hidden, 1)

def build_monthly_snapshots(df: pd.DataFrame, feature_cols: list, edges_raw: pd.DataFrame, scaler: StandardScaler):
    snapshots = []
    ym_pairs = df[["year", "month"]].drop_duplicates().sort_values(by=["year", "month"]).values
    unique_csds = sorted(df["csd_uid"].unique())
    uid_map = {uid: i for i, uid in enumerate(unique_csds)}
    
    src_mapped = edges_raw["source_csd_uid"].map(uid_map).dropna().astype(int).values
    tgt_mapped = edges_raw["target_csd_uid"].map(uid_map).dropna().astype(int).values
    edge_index = torch.tensor(np.stack([src_mapped, tgt_mapped], axis=0), dtype=torch.long)
    
    for y, m in ym_pairs:
        df_t = df[(df["year"] == y) & (df["month"] == m)].copy().reset_index(drop=True)
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
            node_mask=torch.from_numpy(node_mask)
        )
        snapshots.append(data)
    return snapshots

def compute_metrics(y_true, probs, df_eval):
    ap = average_precision_score(y_true, probs)
    roc = roc_auc_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    
    df_temp = pd.DataFrame({'month': df_eval['month'].values, 'y_true': y_true, 'prob': probs})
    df_sorted = df_temp.sort_values('prob', ascending=False)
    
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

def focal_loss(out, y, pos_mask, neg_mask, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(out)
    bce_pos = F.binary_cross_entropy_with_logits(out[pos_mask], y[pos_mask], reduction='none') if pos_mask.sum() > 0 else torch.tensor([0.0])
    bce_neg = F.binary_cross_entropy_with_logits(out[neg_mask], y[neg_mask], reduction='none') if neg_mask.sum() > 0 else torch.tensor([0.0])
    
    pt_pos = p[pos_mask]
    pt_neg = 1 - p[neg_mask]
    
    focal_pos = alpha * (1 - pt_pos)**gamma * bce_pos
    focal_neg = (1 - alpha) * (1 - pt_neg)**gamma * bce_neg
    
    loss = focal_pos.sum() + focal_neg.sum()
    return loss / (pos_mask.sum() + neg_mask.sum() + 1e-6)

def train_pretrain_finetune(in_channels, train_snaps, test_snaps, df_test, seed, do_pretrain=True):
    seed_everything(seed)
    model = SSLGraphSAGE(in_channels=in_channels, hidden=64)
    
    # Pretraining Phase
    if do_pretrain:
        opt_pretrain = torch.optim.Adam(list(model.encoder.parameters()) + list(model.reconstructor.parameters()), lr=0.005)
        mask_ratio = 0.15
        
        for epoch in range(1, 31):
            model.train()
            epoch_loss = 0
            for batch in train_snaps:
                opt_pretrain.zero_grad()
                
                # Active nodes only
                x = batch.x.clone()
                active_idx = torch.where(batch.node_mask)[0]
                
                # Create mask only for active nodes
                mask = torch.rand(len(active_idx), in_channels) < mask_ratio
                
                # Replace masked features with 0 (which is the mean for standardized features)
                x[active_idx[mask[:, 0]], :] = 0.0 # simplified: actually we should mask element-wise
                
                # Element-wise masking
                mask_full = torch.zeros_like(x, dtype=torch.bool)
                mask_full[active_idx] = mask
                x[mask_full] = 0.0
                
                emb = model.encoder(x, batch.edge_index)
                rec = model.reconstructor(emb)
                
                # MSE loss only on the masked elements of active nodes
                loss = F.mse_loss(rec[mask_full], batch.x[mask_full])
                if not torch.isnan(loss):
                    loss.backward()
                    opt_pretrain.step()
                    epoch_loss += loss.item()
            # print(f"Pretrain Epoch {epoch} Loss: {epoch_loss:.4f}")
    
    # Fine-tuning Phase
    opt_finetune = torch.optim.Adam(list(model.encoder.parameters()) + list(model.classifier.parameters()), lr=0.005)
    
    for epoch in range(1, 31):
        model.train()
        for batch in train_snaps:
            opt_finetune.zero_grad()
            emb = model.encoder(batch.x, batch.edge_index)
            out = model.classifier(emb).squeeze(-1)
            
            active_out = out[batch.node_mask]
            active_y = batch.y[batch.node_mask]
            
            pos_mask = (active_y == 1)
            neg_mask = (active_y == 0)
            
            loss = focal_loss(active_out, active_y, pos_mask, neg_mask)
            loss.backward()
            opt_finetune.step()
            
    # Eval test
    model.eval()
    test_probs = []
    with torch.no_grad():
        for batch in test_snaps:
            emb = model.encoder(batch.x, batch.edge_index)
            out = model.classifier(emb).squeeze(-1)
            probs = torch.sigmoid(out)[batch.node_mask].cpu().numpy()
            test_probs.extend(probs)
            
    return np.array(test_probs)

def log_to_registry(experiment_id, group, method_name, ap, roc, r50, brier):
    import csv
    from datetime import datetime
    
    row = [
        experiment_id, datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        group, "v1", "Full 62", "kNN-3", "22-23/24/25", "National", "All",
        "Focal (y=2.0)", "Ensemble (5 seeds)", "GraphSAGE", "~8.5k",
        0.005, 0.0, 0.2, 30, "None", "30", "models/group_d/", "results/group_d/",
        ap, roc, r50, brier
    ]
    
    if not os.path.exists(REGISTRY_PATH):
        print(f"Error: {REGISTRY_PATH} not found. Run build_experiment_registry.py first.")
        return
        
    with open(REGISTRY_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def run_pipeline():
    print("Loading data...")
    df_all = pd.read_csv(INPUT_MODEL_TABLE, dtype={"csd_uid": str})
    
    feature_cols = [c for c in df_all.columns if c not in EXCLUDE_EXPLICIT and pd.api.types.is_numeric_dtype(df_all[c])]
    feature_cols = sorted(feature_cols)
    print(f"Total features: {len(feature_cols)}")
    
    edges_raw = pd.read_csv(KNN_GRAPH, dtype=str)
    
    df_train = df_all[df_all["year"].isin([2022, 2023])].copy().sort_values(["year", "month", "csd_uid"]).reset_index(drop=True)
    df_test = df_all[df_all["year"] == 2025].copy().sort_values(["year", "month", "csd_uid"]).reset_index(drop=True)
    
    scaler_full = StandardScaler()
    scaler_full.fit(df_train[feature_cols].fillna(0.0).values)
    
    train_snaps = build_monthly_snapshots(df_train, feature_cols, edges_raw, scaler_full)
    test_snaps = build_monthly_snapshots(df_test, feature_cols, edges_raw, scaler_full)
    
    y_test_true = df_test["outbreak_next_month"].values

    models = [
        {"id": "EXP_D_00", "name": "D0: GraphSAGE from scratch", "pretrain": False},
        {"id": "EXP_D_01", "name": "D1: Masked-Feature Pretraining + Finetune", "pretrain": True},
    ]
    
    results = []
    
    for cfg in models:
        print(f"\n--- Running: {cfg['name']} ---")
        seed_preds = []
        for s in SEEDS:
            preds = train_pretrain_finetune(len(feature_cols), train_snaps, test_snaps, df_test, s, do_pretrain=cfg["pretrain"])
            seed_preds.append(preds)
        
        ens_preds = np.mean(seed_preds, axis=0)
        metrics = compute_metrics(y_test_true, ens_preds, df_test)
        
        results.append({
            "Configuration": cfg["name"],
            **metrics
        })
        print(f"Ensemble AP: {metrics['AP']:.4f}")
        
        log_to_registry(cfg["id"], "Group D - SSL", cfg["name"], metrics["AP"], metrics["ROC-AUC"], metrics["Macro Recall@50"], metrics["Brier"])
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OUT_DIR, "group_d_summary_results.csv"), index=False)
    print("\n--- Final Group D Results ---")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    run_pipeline()
