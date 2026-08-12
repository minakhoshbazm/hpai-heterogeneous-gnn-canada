#!/usr/bin/env python3
"""
Group H: Feature Ablation for GraphSAGE and MLP.
Compares GraphSAGE vs MLP under different feature sets:
- H0: All features
- H1: Without CSD outbreak lags
- H2: Without Province outbreak lags
- H3: Without all outbreak lags
- H4: MLP under the same sets
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
GRAPH_EDGE_CSV = ROOT / "data/processed/graph/csd_graph_knn_3_v1.csv"
OUT_DIR = ROOT / "results/group_h_ablation"

# Import helper functions from Group B runner
import sys
sys.path.append(str(ROOT / "scripts"))
from run_group_b_negative_sampling import (
    GraphSAGEModel, build_snapshots, load_edge_index, compute_metrics,
    SEEDS, HIDDEN_DIM, DROPOUT, LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, EXCLUDE_EXPLICIT
)

class MLPModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        return self.lin3(x).squeeze(-1)

def run_ablation_experiment(
    model_type: str,
    feature_subset: list[str],
    train_bundle, val_bundle, test_bundle,
    csd_to_idx
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

    # Feature mapping index
    all_features = train_bundle.snapshots[0].x.shape[1]
    # We will slice columns based on indices
    feat_indices = [train_bundle.feature_names.index(f) for f in feature_subset]

    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if model_type == "GraphSAGE":
            model = GraphSAGEModel(in_channels=len(feature_subset))
        else:
            model = MLPModel(in_channels=len(feature_subset))
            
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_state = None
        best_val_ap = -1.0
        epochs_without_improvement = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for snap in train_bundle.snapshots:
                optimizer.zero_grad()
                x_sub = snap.x[:, feat_indices]
                logits = model(x_sub, snap.edge_index)
                loss = criterion(logits[snap.mask], snap.y[snap.mask])
                loss.backward()
                optimizer.step()

            # Validation step
            model.eval()
            val_probs = []
            with torch.no_grad():
                for snap in val_bundle.snapshots:
                    x_sub = snap.x[:, feat_indices]
                    logits = model(x_sub, snap.edge_index)
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
                x_sub = snap.x[:, feat_indices]
                logits = model(x_sub, snap.edge_index)
                val_probs.extend(torch.sigmoid(logits)[snap.mask].cpu().numpy())
            for snap in test_bundle.snapshots:
                x_sub = snap.x[:, feat_indices]
                logits = model(x_sub, snap.edge_index)
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
    df = df.sort_values(["year", "month", "csd_uid"]).reset_index(drop=True)
    
    df_train = df[df["year"].isin([2022, 2023])].copy()
    df_val = df[df["year"] == 2024].copy()
    df_test = df[df["year"] == 2025].copy()

    scaler = StandardScaler()
    df_train.loc[:, cols] = scaler.fit_transform(df_train[cols].fillna(0.0))
    df_val.loc[:, cols] = scaler.transform(df_val[cols].fillna(0.0))
    df_test.loc[:, cols] = scaler.transform(df_test[cols].fillna(0.0))

    all_csds = sorted(df["csd_uid"].unique())
    csd_to_idx = {csd: idx for idx, csd in enumerate(all_csds)}
    edge_index, _ = load_edge_index(all_csds)

    train_bundle = build_snapshots(df_train, cols, csd_to_idx, edge_index)
    val_bundle = build_snapshots(df_val, cols, csd_to_idx, edge_index)
    test_bundle = build_snapshots(df_test, cols, csd_to_idx, edge_index)
    
    # Expose feature list to snaps
    train_bundle.feature_names = cols
    val_bundle.feature_names = cols
    test_bundle.feature_names = cols

    # Define feature groups
    csd_lag_cols = [c for c in cols if c.startswith("same_csd_outbreak")]
    prov_lag_cols = [c for c in cols if c.startswith("province_outbreak")]
    all_lag_cols = csd_lag_cols + prov_lag_cols

    subsets = {
        "H0_all_features": cols,
        "H1_no_csd_lags": [c for c in cols if c not in csd_lag_cols],
        "H2_no_prov_lags": [c for c in cols if c not in prov_lag_cols],
        "H3_no_lags": [c for c in cols if c not in all_lag_cols],
    }

    results = []

    for name, subset in subsets.items():
        print(f"\n--- Running GraphSAGE under {name} (features={len(subset)}) ---")
        gs_sum = run_ablation_experiment("GraphSAGE", subset, train_bundle, val_bundle, test_bundle, csd_to_idx)
        gs_sum["Configuration"] = f"GraphSAGE_{name}"
        results.append(gs_sum)

        print(f"\n--- Running MLP under {name} (features={len(subset)}) ---")
        mlp_sum = run_ablation_experiment("MLP", subset, train_bundle, val_bundle, test_bundle, csd_to_idx)
        mlp_sum["Configuration"] = f"MLP_{name}"
        results.append(mlp_sum)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_DIR / "ablation_summary_results.csv", index=False)
    print("\n=== Group H Ablation Results ===")
    print(df_res.to_markdown(index=False))

if __name__ == "__main__":
    main()
