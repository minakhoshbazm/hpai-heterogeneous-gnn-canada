#!/usr/bin/env python3
"""
scripts/evaluate_road_graphsage.py

Evaluates the GraphSAGE model on three candidate graphs:
1. Current kNN-3 geographic graph
2. Road-connectivity graph
3. Hybrid geographic + road-connectivity graph

Strict Temporal Evaluation: Train 2022-2024, Test 2025.
Reports AP (PR-AUC), Macro Recall@50, Macro P@10, and ROC-AUC.
"""

import os
import random
import copy
import numpy as np
import pandas as pd

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

GRAPHS = {
    "1. kNN-3 Geographic": "data/processed/graph/csd_graph_knn_3_v1.csv",
    "2. Road-Connectivity": "data/processed/graph/csd_graph_road_corridor_v1.csv",
    "3. Hybrid Geographic + Road": "data/processed/graph/csd_graph_hybrid_knn3_road_v1.csv"
}

SEEDS = [11, 22, 33, 44, 55]

EXCLUDE_EXPLICIT = {
    "csd_uid", "csd_name", "province_code", "province_name", "ccs_uid",
    "year", "month", "outbreak_next_month",
    "dw_fill_method", "climate_extraction_method",
    "glw2020_chicken_extraction_method", "glw2015_duck_extraction_method",
    "road_data_coverage", "outbreak_current_month",
}

def get_62_features(df):
    cols = []
    for c in df.columns:
        if c in EXCLUDE_EXPLICIT:
            continue
        cols.append(c)
    cols = sorted(cols)
    assert len(cols) == 62, f"Expected 62 features, got {len(cols)}\nFound cols: {cols}"
    return cols

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

def build_monthly_snapshots(df: pd.DataFrame, feature_cols: list, edges_raw: pd.DataFrame, scaler: StandardScaler):
    snapshots = []
    ym_pairs = df[["year", "month"]].drop_duplicates().sort_values(by=["year", "month"]).values

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

        data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor, year=int(y), month=int(m), csd_uids=df_t["csd_uid"].values)
        snapshots.append(data)
        
    return snapshots

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

def main():
    print("=== GraphSAGE Graph Construction Evaluation ===")
    
    df_all = pd.read_csv(INPUT_MODEL_TABLE, dtype={"csd_uid": str})
    
    # 2025 Test Split
    df_train = df_all[df_all["year"] <= 2024].copy()
    df_test = df_all[df_all["year"] == 2025].copy()
    
    print(f"Train size: {len(df_train)} (Outbreaks: {df_train['outbreak_next_month'].sum()})")
    print(f"Test size: {len(df_test)} (Outbreaks: {df_test['outbreak_next_month'].sum()})")
    
    feature_cols = get_62_features(df_all)
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].fillna(0.0).values)
    
    results_summary = []
    
    for graph_name, graph_path in GRAPHS.items():
        print(f"\nEvaluating: {graph_name}")
        edges_raw = pd.read_csv(graph_path, dtype=str)
        print(f"  Edges loaded: {len(edges_raw)}")
        
        train_snaps = build_monthly_snapshots(df_train, feature_cols, edges_raw, scaler)
        test_snaps = build_monthly_snapshots(df_test, feature_cols, edges_raw, scaler)
        
        train_loader = PyGDataLoader(train_snaps, batch_size=1, shuffle=True)
        test_loader = PyGDataLoader(test_snaps, batch_size=1, shuffle=False)
        
        seed_predictions = []
        
        for seed in SEEDS:
            seed_everything(seed)
            model = GraphSAGEModel(in_channels=62, hidden=64)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            
            # Focal Loss
            alpha = 0.25
            gamma = 2.0
            
            best_val_loss = float('inf')
            best_state = None
            
            # Since no val set is strictly defined, train for fixed 30 epochs (standard in this project)
            for epoch in range(1, 31):
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index)
                    
                    pos_mask = (batch.y == 1)
                    neg_mask = (batch.y == 0)
                    
                    p = torch.sigmoid(out)
                    bce_pos = F.binary_cross_entropy_with_logits(out[pos_mask], batch.y[pos_mask], reduction='none') if pos_mask.sum() > 0 else torch.tensor([0.0])
                    bce_neg = F.binary_cross_entropy_with_logits(out[neg_mask], batch.y[neg_mask], reduction='none') if neg_mask.sum() > 0 else torch.tensor([0.0])
                    
                    pt_pos = p[pos_mask]
                    pt_neg = 1 - p[neg_mask]
                    
                    focal_pos = alpha * (1 - pt_pos)**gamma * bce_pos
                    focal_neg = (1 - alpha) * (1 - pt_neg)**gamma * bce_neg
                    
                    loss = focal_pos.sum() + focal_neg.sum()
                    loss = loss / (pos_mask.sum() + neg_mask.sum() + 1e-6)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate on Test
            model.eval()
            all_preds = []
            all_y = []
            all_years = []
            all_months = []
            all_csds = []
            
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
                            
                    # Sometimes PyG wraps things weirdly. Ensure exact length.
                    if len(flat_csds) > num_nodes:
                        flat_csds = flat_csds[:num_nodes]
                    elif len(flat_csds) < num_nodes:
                        flat_csds.extend(["unknown"] * (num_nodes - len(flat_csds)))
                        
                    all_csds.extend(flat_csds)
                    
            print(f"Lengths: y={len(all_y)}, preds={len(all_preds)}, years={len(all_years)}, months={len(all_months)}, csds={len(all_csds)}")
            
            df_res = pd.DataFrame({
                "year": all_years,
                "month": all_months,
                "csd_uid": all_csds,
                "true_label": all_y,
                f"pred_prob_{seed}": all_preds
            })
            seed_predictions.append(df_res)
            
        # Ensemble predictions
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
        
        print(f"  Ensemble AP (PR-AUC): {ap:.4f}")
        print(f"  Ensemble ROC-AUC: {roc_auc:.4f}")
        print(f"  Macro Recall@50: {r50:.4f}")
        print(f"  Macro Precision@10: {p10:.4f}")
        
        results_summary.append({
            "Graph": graph_name,
            "AP (PR-AUC)": ap,
            "ROC-AUC": roc_auc,
            "Recall@50": r50,
            "Precision@10": p10
        })
        
    print("\n=== Final Summary Table ===")
    df_sum = pd.DataFrame(results_summary)
    print(df_sum.to_markdown(index=False))

if __name__ == '__main__':
    main()
