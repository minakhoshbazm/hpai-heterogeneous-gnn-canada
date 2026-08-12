#!/usr/bin/env python3
"""
Group I: Graph Construction analysis for GraphSAGE.
Compares different spatial graph topologies:
- I0: kNN-3 geographic centroids graph (baseline)
- I1: Border adjacency graph (csd_graph_border_only_v1.csv)
- I3: Road corridor graph (csd_graph_road_corridor_v1.csv)
- I4: Hybrid geo + road graph (csd_graph_hybrid_knn3_road_v1.csv)
Separately compares full national graph vs poultry-relevant CSD graph.
Reports removed CSDs and positives for poultry filtering.
"""

from __future__ import annotations
import os
import gc
import json
import random
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
INPUT_MODEL_TABLE = ROOT / "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
OUT_DIR = ROOT / "results/group_i_graphs"

# Import helper functions from Group B runner
import sys
sys.path.append(str(ROOT / "scripts"))
from run_group_b_negative_sampling import (
    GraphSAGEModel, build_snapshots, compute_metrics,
    SEEDS, HIDDEN_DIM, DROPOUT, LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, EXCLUDE_EXPLICIT, POULTRY_EXPOSURE_COLS
)

def load_graph_edge_index(csv_path: str, all_csds: list[str]) -> torch.Tensor:
    edges = pd.read_csv(csv_path, dtype=str)
    csd_to_idx = {csd: idx for idx, csd in enumerate(all_csds)}
    src = edges["source_csd_uid"].map(csd_to_idx)
    dst = edges["target_csd_uid"].map(csd_to_idx)
    valid = src.notna() & dst.notna()
    src_arr = src[valid].astype(int).to_numpy()
    dst_arr = dst[valid].astype(int).to_numpy()
    edge_index = np.stack(
        [np.concatenate([src_arr, dst_arr]), np.concatenate([dst_arr, src_arr])],
        axis=0,
    )
    return torch.tensor(edge_index, dtype=torch.long)

def run_graph_experiment(
    graph_path: str,
    train_bundle, val_bundle, test_bundle,
    csd_to_idx, cols
) -> dict:
    
    val_pred_df = val_bundle.rows.copy()
    test_pred_df = test_bundle.rows.copy()
    ensemble_val_cols = []
    ensemble_test_cols = []
    
    seed_val_aps = []
    seed_test_aps = []
    
    train_labels = np.concatenate([snap.y[snap.mask].numpy() for snap in train_bundle.snapshots])
    train_pos = float(train_labels.sum())
    train_neg = float(len(train_labels) - train_pos)
    pos_weight = torch.tensor([train_neg / train_pos], dtype=torch.float32)

    all_csds = sorted(csd_to_idx.keys())
    edge_index = load_graph_edge_index(graph_path, all_csds)

    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = GraphSAGEModel(in_channels=len(cols))
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_state = None
        best_val_ap = -1.0
        epochs_without_improvement = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for snap in train_bundle.snapshots:
                optimizer.zero_grad()
                logits = model(snap.x, edge_index)
                loss = criterion(logits[snap.mask], snap.y[snap.mask])
                loss.backward()
                optimizer.step()

            # Validation step
            model.eval()
            val_probs = []
            with torch.no_grad():
                for snap in val_bundle.snapshots:
                    logits = model(snap.x, edge_index)
                    probs = torch.sigmoid(logits)[snap.mask].cpu().numpy()
                    val_probs.extend(probs)
            
            val_eval = val_bundle.rows.copy()
            val_eval["prob"] = np.array(val_probs)
            val_metrics = compute_metrics(val_eval)
            
            if val_metrics["pr_auc"] > best_val_ap:
                best_val_ap = val_metrics["pr_auc"]
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= PATIENCE:
                break

        model.load_state_dict(best_state)
        
        # Predict test
        model.eval()
        val_probs = []
        test_probs = []
        with torch.no_grad():
            for snap in val_bundle.snapshots:
                logits = model(snap.x, edge_index)
                val_probs.extend(torch.sigmoid(logits)[snap.mask].cpu().numpy())
            for snap in test_bundle.snapshots:
                logits = model(snap.x, edge_index)
                test_probs.extend(torch.sigmoid(logits)[snap.mask].cpu().numpy())

        val_col = f"seed_{seed}_prob"
        test_col = f"seed_{seed}_prob"
        val_pred_df[val_col] = val_probs
        test_pred_df[test_col] = test_probs
        ensemble_val_cols.append(val_col)
        ensemble_test_cols.append(test_col)

        test_eval = test_bundle.rows.copy()
        test_eval["prob"] = test_probs
        t_met = compute_metrics(test_eval)
        
        seed_val_aps.append(best_val_ap)
        seed_test_aps.append(t_met["pr_auc"])

    val_pred_df["ensemble_prob"] = val_pred_df[ensemble_val_cols].mean(axis=1)
    test_pred_df["ensemble_prob"] = test_pred_df[ensemble_test_cols].mean(axis=1)

    val_eval = val_bundle.rows.copy()
    val_eval["prob"] = val_pred_df["ensemble_prob"].values
    test_eval = test_bundle.rows.copy()
    test_eval["prob"] = test_pred_df["ensemble_prob"].values

    ens_val_metrics = compute_metrics(val_eval)
    ens_test_metrics = compute_metrics(test_eval)

    mean_val_ap = float(np.mean(seed_val_aps))
    mean_test_ap = float(np.mean(seed_test_aps))
    std_test_ap = float(np.std(seed_test_aps))

    summary = {
        "Mean Val AP": mean_val_ap,
        "Ensemble Val AP": ens_val_metrics["pr_auc"],
        "Mean Test AP": mean_test_ap,
        "Std Test AP": std_test_ap,
        "Ensemble Test AP": ens_test_metrics["pr_auc"],
        "ROC-AUC": ens_test_metrics["roc_auc"],
        "Recall@50": ens_test_metrics["macro_recall_at_50"],
        "Precision@10": ens_test_metrics["macro_precision_at_10"],
        "Brier": ens_test_metrics["brier_score"]
    }
    return summary

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_MODEL_TABLE, dtype={"csd_uid": str})
    cols = sorted([c for c in df.columns if c not in EXCLUDE_EXPLICIT and pd.api.types.is_numeric_dtype(df[c])])
    
    # Pre-calculate poultry-at-risk CSDs
    poultry_farm_cols = [c for c in POULTRY_EXPOSURE_COLS if 'farms_reporting' in c or 'production_kg' in c]
    df['poultry_farm_sum'] = df[poultry_farm_cols].fillna(0.0).sum(axis=1)
    poultry_relevant_csds = set(df.groupby('csd_uid')['poultry_farm_sum'].max().loc[lambda x: x > 0].index)

    df = df.sort_values(["year", "month", "csd_uid"]).reset_index(drop=True)

    graphs = {
        "I0_knn3_centroid": str(ROOT / "data/processed/graph/csd_graph_knn_3_v1.csv"),
        "I1_border_adjacency": str(ROOT / "data/processed/graph/csd_graph_border_only_v1.csv"),
        "I3_road_corridor": str(ROOT / "data/processed/graph/csd_graph_road_corridor_v1.csv"),
        "I4_hybrid_geo_road": str(ROOT / "data/processed/graph/csd_graph_hybrid_knn3_road_v1.csv")
    }

    results = []

    # Part 1: National graph comparisons
    print("=== Part 1: Comparing Graph Topologies (National) ===")
    df_train = df[df["year"].isin([2022, 2023])].copy()
    df_val = df[df["year"] == 2024].copy()
    df_test = df[df["year"] == 2025].copy()

    scaler = StandardScaler()
    df_train.loc[:, cols] = scaler.fit_transform(df_train[cols].fillna(0.0))
    df_val.loc[:, cols] = scaler.transform(df_val[cols].fillna(0.0))
    df_test.loc[:, cols] = scaler.transform(df_test[cols].fillna(0.0))

    all_csds = sorted(df["csd_uid"].unique())
    csd_to_idx = {csd: idx for idx, csd in enumerate(all_csds)}
    
    # We will build standard snapshot sets over standard edge indexes
    # (The function build_snapshots internally loads a dummy edge index which we ignore and overwrite in run_graph_experiment)
    dummy_edge_index = torch.zeros((2, 0), dtype=torch.long)
    train_bundle = build_snapshots(df_train, cols, csd_to_idx, dummy_edge_index)
    val_bundle = build_snapshots(df_val, cols, csd_to_idx, dummy_edge_index)
    test_bundle = build_snapshots(df_test, cols, csd_to_idx, dummy_edge_index)

    for name, path in graphs.items():
        print(f"\n--- Running Graph Topology: {name} ---")
        g_sum = run_graph_experiment(path, train_bundle, val_bundle, test_bundle, csd_to_idx, cols)
        g_sum["Topology"] = name
        g_sum["Population"] = "National"
        results.append(g_sum)

    # Part 2: Poultry-Relevant CSD Graph (filtering evaluation population)
    print("\n=== Part 2: Poultry-Relevant CSD Graph ===")
    df_p = df[df["csd_uid"].isin(poultry_relevant_csds)].copy()
    df_p_train = df_p[df_p["year"].isin([2022, 2023])].copy()
    df_p_val = df_p[df_p["year"] == 2024].copy()
    df_p_test = df_p[df_p["year"] == 2025].copy()

    scaler_p = StandardScaler()
    df_p_train.loc[:, cols] = scaler_p.fit_transform(df_p_train[cols].fillna(0.0))
    df_p_val.loc[:, cols] = scaler_p.transform(df_p_val[cols].fillna(0.0))
    df_p_test.loc[:, cols] = scaler_p.transform(df_p_test[cols].fillna(0.0))

    p_csds = sorted(df_p["csd_uid"].unique())
    p_csd_to_idx = {csd: idx for idx, csd in enumerate(p_csds)}

    train_bundle_p = build_snapshots(df_p_train, cols, p_csd_to_idx, dummy_edge_index)
    val_bundle_p = build_snapshots(df_p_val, cols, p_csd_to_idx, dummy_edge_index)
    test_bundle_p = build_snapshots(df_p_test, cols, p_csd_to_idx, dummy_edge_index)

    print("\n--- Running Poultry-Relevant Graph: I0_knn3_centroid ---")
    g_sum_p = run_graph_experiment(graphs["I0_knn3_centroid"], train_bundle_p, val_bundle_p, test_bundle_p, p_csd_to_idx, cols)
    g_sum_p["Topology"] = "I0_knn3_centroid"
    g_sum_p["Population"] = "Poultry-Relevant"
    results.append(g_sum_p)

    # Calculate audit counts for filtering
    removed_csds = len(all_csds) - len(p_csds)
    pos_train_all = int((df_train["outbreak_next_month"] > 0).sum())
    pos_train_p = int((df_p_train["outbreak_next_month"] > 0).sum())
    removed_pos_train = pos_train_all - pos_train_p

    pos_val_all = int((df_val["outbreak_next_month"] > 0).sum())
    pos_val_p = int((df_p_val["outbreak_next_month"] > 0).sum())
    removed_pos_val = pos_val_all - pos_val_p

    pos_test_all = int((df_test["outbreak_next_month"] > 0).sum())
    pos_test_p = int((df_p_test["outbreak_next_month"] > 0).sum())
    removed_pos_test = pos_test_all - pos_test_p

    audit_stats = {
        "Total National CSDs": len(all_csds),
        "Poultry-Relevant CSDs": len(p_csds),
        "Removed CSDs": removed_csds,
        "Removed CSD %": removed_csds / len(all_csds),
        "National Train Positives": pos_train_all,
        "Poultry-Relevant Train Positives": pos_train_p,
        "Removed Train Positives": removed_pos_train,
        "National Val Positives": pos_val_all,
        "Poultry-Relevant Val Positives": pos_val_p,
        "Removed Val Positives": removed_pos_val,
        "National Test Positives": pos_test_all,
        "Poultry-Relevant Test Positives": pos_test_p,
        "Removed Test Positives": removed_pos_test
    }
    
    with open(OUT_DIR / "poultry_filtering_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_stats, f, indent=2)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_DIR / "graph_comparison_summary.csv", index=False)
    
    print("\n=== Group I Graph Construction Summary ===")
    print(df_res.to_markdown(index=False))
    print("\n=== Poultry Filtering Audit ===")
    print(json.dumps(audit_stats, indent=2))

if __name__ == "__main__":
    main()
