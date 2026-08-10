#!/usr/bin/env python3
"""
scripts/train_wildbird_monthly_snapshot_graphsage_knn3.py

Trains a true CSD-month snapshot GraphSAGE model on wild-bird-enhanced features.

Critical Architecture Fix:
- Builds a separate spatial graph Data snapshot for every (year, month).
- Each snapshot contains nodes for CSDs in that month with their ACTUAL time-varying 62 features.
- Spatial edges (kNN-3) connect CSDs in the SAME month only (no cross-time edges, no static feature overwriting).
- GraphSAGE encoder weights are shared across all monthly snapshots.
- Model processes monthly snapshots dynamically, eliminating temporal collapse.

Splits:
- Development Training: 2022-01 through 2024-09
- Development Validation: 2024-10 through 2024-12 (determines optimal early-stopping epoch per seed)
- Final Training: 2022-01 through 2024-12 (StandardScaler fit on full 2022-2024)
- Held-out Evaluation: 2025 CSD-month rows (evaluated exactly once)

Outputs (new, never overwritten):
- results/baselines/wildbird_monthly_snapshot_graphsage_knn3_multiseed_metrics.csv
- results/baselines/wildbird_monthly_snapshot_graphsage_knn3_test_predictions.csv
- results/baselines/wildbird_monthly_snapshot_graphsage_knn3_summary.md
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

# Environment configuration
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# Paths
INPUT_MODEL_TABLE = "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
GRAPH_EDGE_CSV = "data/processed/graph/csd_graph_knn_3_v1.csv"

OUT_METRICS = "results/baselines/wildbird_monthly_snapshot_graphsage_knn3_multiseed_metrics.csv"
OUT_PRED = "results/baselines/wildbird_monthly_snapshot_graphsage_knn3_test_predictions.csv"
OUT_MD = "results/baselines/wildbird_monthly_snapshot_graphsage_knn3_summary.md"

SEEDS = [11, 22, 33]

EXCLUDE_EXPLICIT = {
    "csd_uid", "csd_name", "province_code", "province_name", "ccs_uid",
    "year", "month", "outbreak_next_month",
    "dw_fill_method", "climate_extraction_method",
    "glw2020_chicken_extraction_method", "glw2015_duck_extraction_method",
    "road_data_coverage", "outbreak_current_month",
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def precision_recall_at_k(y_true, y_prob, k):
    df_eval = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df_eval = df_eval.sort_values(by="y_prob", ascending=False)
    top_k = df_eval.head(k)
    prec = top_k["y_true"].mean()
    total_pos = y_true.sum()
    rec = top_k["y_true"].sum() / total_pos if total_pos > 0 else 0.0
    return prec, rec

def evaluate_metrics(y_true, y_prob):
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    p50, r50 = precision_recall_at_k(y_true, y_prob, 50)
    p100, r100 = precision_recall_at_k(y_true, y_prob, 100)
    p200, r200 = precision_recall_at_k(y_true, y_prob, 200)
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision_at_50": p50,
        "recall_at_50": r50,
        "precision_at_100": p100,
        "recall_at_100": r100,
        "precision_at_200": p200,
        "recall_at_200": r200,
    }

class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden=32):
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

def load_spatial_edges(edge_csv: str):
    return pd.read_csv(edge_csv)

def build_monthly_snapshots(df: pd.DataFrame, feature_cols: list, edges_raw: pd.DataFrame, scaler: StandardScaler):
    """
    Builds a list of PyG Data snapshots, one per (year, month).
    Each snapshot has its own node order, scaled feature matrix X_t, y_t, and mapped edge_index_t.
    """
    snapshots = []
    # Sort distinct year-month pairs chronologically
    ym_pairs = df[["year", "month"]].drop_duplicates().sort_values(by=["year", "month"]).values

    for y, m in ym_pairs:
        df_t = df[(df["year"] == y) & (df["month"] == m)].copy().reset_index(drop=True)
        n_t = len(df_t)

        # Scale node features for this month
        X_t_raw = df_t[feature_cols].fillna(0.0).values
        X_t = scaler.transform(X_t_raw).astype(np.float32)

        y_t = df_t["outbreak_next_month"].values.astype(np.float32)

        # Remap csd_uid to 0..n_t-1 index within this month's snapshot
        uid_map = {uid: i for i, uid in enumerate(df_t["csd_uid"].values)}

        # Filter and remap edges
        src_mapped = edges_raw["source_csd_uid"].map(uid_map).dropna().astype(int).values
        tgt_mapped = edges_raw["target_csd_uid"].map(uid_map).dropna().astype(int).values

        edge_index = torch.tensor(np.stack([src_mapped, tgt_mapped], axis=0), dtype=torch.long)
        x_tensor = torch.from_numpy(X_t)
        y_tensor = torch.from_numpy(y_t)

        data = Data(
            x=x_tensor,
            edge_index=edge_index,
            y=y_tensor,
            year=int(y),
            month=int(m),
            csd_uids=df_t["csd_uid"].values,
            csd_names=df_t["csd_name"].values if "csd_name" in df_t else np.array([""] * n_t),
            province_codes=df_t["province_code"].values if "province_code" in df_t else np.array([0] * n_t)
        )
        snapshots.append(data)

    return snapshots

def evaluate_snapshots(model, snapshot_loader):
    model.eval()
    all_probs = []
    all_ys = []
    with torch.no_grad():
        for batch in snapshot_loader:
            logits = model(batch.x, batch.edge_index)
            probs = torch.sigmoid(logits).cpu().numpy()
            ys = batch.y.cpu().numpy()
            all_probs.extend(probs)
            all_ys.extend(ys)
    return np.array(all_ys), np.array(all_probs)

def main():
    print("=== Training True Monthly-Snapshot Wild-Bird GraphSAGE (kNN-3) ===")

    # Load data
    df = pd.read_csv(INPUT_MODEL_TABLE)
    print(f"Loaded modeling table: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # Select feature columns
    feature_cols = []
    for c in df.columns:
        if c in EXCLUDE_EXPLICIT:
            continue
        if any(sub in c.lower() for sub in ["ebird", "gbif", "neighbor", "pressure"]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)
    feature_cols = sorted(feature_cols)

    print(f"Selected {len(feature_cols)} features")
    assert len(feature_cols) == 62, f"Expected 62 features, got {len(feature_cols)}"

    edges_raw = load_spatial_edges(GRAPH_EDGE_CSV)
    print(f"Loaded raw edge list: {len(edges_raw):,} edges")

    # Splits
    df_dev_train = df[(df["year"] <= 2023) | ((df["year"] == 2024) & (df["month"] <= 9))].copy()
    df_dev_val = df[(df["year"] == 2024) & (df["month"] >= 10)].copy()
    df_final_train = df[df["year"] <= 2024].copy()
    df_test = df[df["year"] == 2025].copy()

    print(f"Dev Train: {len(df_dev_train):,} rows, Dev Val: {len(df_dev_val):,} rows")
    print(f"Final Train: {len(df_final_train):,} rows, Final Test: {len(df_test):,} rows")

    # -------------------------------------------------------------------
    # Step 1: Development Phase (Determine Optimal Epoch per Seed)
    # -------------------------------------------------------------------
    print("\n-------------------------------------------------------------------")
    print("STEP 1: DEVELOPMENT PHASE (Determining Early Stopping Epochs)")
    print("-------------------------------------------------------------------")

    scaler_dev = StandardScaler()
    scaler_dev.fit(df_dev_train[feature_cols].fillna(0.0).values)

    dev_train_snaps = build_monthly_snapshots(df_dev_train, feature_cols, edges_raw, scaler_dev)
    dev_val_snaps = build_monthly_snapshots(df_dev_val, feature_cols, edges_raw, scaler_dev)

    loader_dev_train = PyGDataLoader(dev_train_snaps, batch_size=4, shuffle=True)
    loader_dev_val = PyGDataLoader(dev_val_snaps, batch_size=len(dev_val_snaps), shuffle=False)

    dev_pos = sum([s.y.sum().item() for s in dev_train_snaps])
    dev_total = sum([s.y.shape[0] for s in dev_train_snaps])
    pos_weight_dev = torch.tensor([(dev_total - dev_pos) / dev_pos], dtype=torch.float)

    optimal_epochs = {}
    patience = 20

    for seed in SEEDS:
        set_seed(seed)
        model = GraphSAGEModel(in_channels=62, hidden=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_dev)

        best_val_ap = -1.0
        best_epoch = 0
        no_improve = 0

        for epoch in range(1, 201):
            model.train()
            for batch in loader_dev_train:
                optimizer.zero_grad()
                logits = model(batch.x, batch.edge_index)
                loss = criterion(logits, batch.y)
                loss.backward()
                optimizer.step()

            # Validation PR-AUC
            y_val, p_val = evaluate_snapshots(model, loader_dev_val)
            val_ap = average_precision_score(y_val, p_val)

            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        optimal_epochs[seed] = best_epoch
        print(f"Seed {seed}: Best Dev Val PR-AUC = {best_val_ap:.4f} at Epoch {best_epoch}")

    # -------------------------------------------------------------------
    # Step 2: Final Phase (Retrain on Full 2022-2024 & Evaluate on 2025)
    # -------------------------------------------------------------------
    print("\n-------------------------------------------------------------------")
    print("STEP 2: FINAL PHASE (Retraining on 2022-2024 & Evaluating 2025)")
    print("-------------------------------------------------------------------")

    scaler_final = StandardScaler()
    scaler_final.fit(df_final_train[feature_cols].fillna(0.0).values)

    final_train_snaps = build_monthly_snapshots(df_final_train, feature_cols, edges_raw, scaler_final)
    test_snaps = build_monthly_snapshots(df_test, feature_cols, edges_raw, scaler_final)

    loader_final_train = PyGDataLoader(final_train_snaps, batch_size=4, shuffle=True)
    loader_test = PyGDataLoader(test_snaps, batch_size=len(test_snaps), shuffle=False)

    final_pos = sum([s.y.sum().item() for s in final_train_snaps])
    final_total = sum([s.y.shape[0] for s in final_train_snaps])
    pos_weight_final = torch.tensor([(final_total - final_pos) / final_pos], dtype=torch.float)

    seed_results = []
    test_probs_seeds = []

    for seed in SEEDS:
        target_epochs = optimal_epochs[seed]
        print(f"\n--- Retraining Seed {seed} for {target_epochs} Epochs ---")
        set_seed(seed)
        model = GraphSAGEModel(in_channels=62, hidden=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_final)

        for epoch in range(1, target_epochs + 1):
            model.train()
            for batch in loader_final_train:
                optimizer.zero_grad()
                logits = model(batch.x, batch.edge_index)
                loss = criterion(logits, batch.y)
                loss.backward()
                optimizer.step()

        # Evaluate on 2025 test snapshots
        y_test, test_probs = evaluate_snapshots(model, loader_test)

        prob_stats = {
            "min": float(test_probs.min()),
            "max": float(test_probs.max()),
            "mean": float(test_probs.mean()),
            "std": float(test_probs.std()),
            "unique": int(np.unique(test_probs).shape[0])
        }
        print(f"Seed {seed} Prediction Stats:", prob_stats)
        assert prob_stats["std"] > 1e-6, f"Seed {seed}: Probability collapsed!"

        metrics = evaluate_metrics(y_test, test_probs)
        metrics.update({
            "seed": seed,
            "finetune_epoch": target_epochs,
            "val_pr_auc": np.nan,  # Dev validation score reported separately
            "model": f"Monthly Snapshot GraphSAGE Seed {seed}",
            **prob_stats
        })
        seed_results.append(metrics)
        test_probs_seeds.append(test_probs)

    # -------------------------------------------------------------------
    # Step 3: Multi-Seed Ensemble & Output Generation
    # -------------------------------------------------------------------
    ensemble_probs = np.mean(np.stack(test_probs_seeds, axis=0), axis=0)
    ensemble_metrics = evaluate_metrics(y_test, ensemble_probs)
    ensemble_metrics.update({
        "model": "Wild-Bird Monthly-Snapshot kNN-3 GraphSAGE (Ensemble)",
        "seed": "ensemble",
        "finetune_epoch": int(np.mean(list(optimal_epochs.values()))),
        "val_pr_auc": np.nan,
        "min": float(ensemble_probs.min()),
        "max": float(ensemble_probs.max()),
        "mean": float(ensemble_probs.mean()),
        "std": float(ensemble_probs.std()),
        "unique": int(np.unique(ensemble_probs).shape[0])
    })

    # Save Multi-Seed Metrics CSV
    df_metrics = pd.DataFrame(seed_results)
    df_metrics = pd.concat([df_metrics, pd.DataFrame([ensemble_metrics])], ignore_index=True)
    os.makedirs(os.path.dirname(OUT_METRICS), exist_ok=True)
    df_metrics.to_csv(OUT_METRICS, index=False)
    print(f"\nSaved metrics to {OUT_METRICS}")

    # Build Test Predictions CSV in exact chronological / row order of df_test
    # Extract prediction rows by combining test snapshot metadata
    pred_records = []
    curr_offset = 0
    for snap in test_snaps:
        n_snap = snap.x.shape[0]
        snap_probs_seeds = [t_probs[curr_offset:curr_offset+n_snap] for t_probs in test_probs_seeds]
        snap_ens_probs = ensemble_probs[curr_offset:curr_offset+n_snap]

        for i in range(n_snap):
            rec = {
                "csd_uid": snap.csd_uids[i],
                "csd_name": snap.csd_names[i],
                "province_code": snap.province_codes[i],
                "year": snap.year,
                "month": snap.month,
                "actual": snap.y[i].item(),
            }
            for idx_s, s_val in enumerate(SEEDS):
                rec[f"wildbird_snapshot_gs_seed_{s_val}_prob"] = snap_probs_seeds[idx_s][i]
            rec["wildbird_snapshot_gs_ensemble_prob"] = snap_ens_probs[i]
            pred_records.append(rec)
        curr_offset += n_snap

    pred_df = pd.DataFrame(pred_records)
    # Ensure exact alignment with df_test order
    pred_df = df_test[["csd_uid", "year", "month"]].merge(pred_df, on=["csd_uid", "year", "month"], how="left")
    pred_df.to_csv(OUT_PRED, index=False)
    print(f"Saved test predictions to {OUT_PRED}")

    # Markdown Summary
    md_content = f"""# Wild-Bird Monthly-Snapshot kNN-3 GraphSAGE Summary

**Architecture**: Monthly Spatio-Temporal Snapshot GraphSAGE (Dynamic Node Features per Month, Shared Spatial Edge Topology, No Temporal Collapse)  
**Training Period**: 2022-01 through 2024-12  
**Test Period**: 2025 (Held-out single evaluation)  
**Feature Count**: 62  

## Multi-Seed Metrics

| Seed | Selected Epoch | Test PR-AUC | Test ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
|---|---|---|---|---|---|---|---|---|---|
"""
    for r in seed_results:
        md_content += f"| {r['seed']} | {r['finetune_epoch']} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['precision_at_50']:.4f} | {r['recall_at_50']:.4f} | {r['precision_at_100']:.4f} | {r['recall_at_100']:.4f} | {r['precision_at_200']:.4f} | {r['recall_at_200']:.4f} |\n"

    md_content += f"| **Ensemble** | {ensemble_metrics['finetune_epoch']} | **{ensemble_metrics['pr_auc']:.4f}** | **{ensemble_metrics['roc_auc']:.4f}** | **{ensemble_metrics['precision_at_50']:.4f}** | **{ensemble_metrics['recall_at_50']:.4f}** | **{ensemble_metrics['precision_at_100']:.4f}** | **{ensemble_metrics['recall_at_100']:.4f}** | **{ensemble_metrics['precision_at_200']:.4f}** | **{ensemble_metrics['recall_at_200']:.4f}** |\n\n"

    md_content += f"""## Prediction Stats (Ensemble)
- Min Prob: {ensemble_metrics['min']:.6f}
- Max Prob: {ensemble_metrics['max']:.6f}
- Mean Prob: {ensemble_metrics['mean']:.6f}
- Std Dev: {ensemble_metrics['std']:.6f}
- Unique Predictions: {ensemble_metrics['unique']}
"""

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Saved summary markdown to {OUT_MD}")

if __name__ == '__main__':
    main()
