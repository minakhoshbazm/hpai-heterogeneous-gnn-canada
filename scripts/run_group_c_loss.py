#!/usr/bin/env python3
"""
Group C: Loss Function comparison for GraphSAGE.
Reads the top two negative sampling configurations from Group B's summary based on validation performance,
and compares:
- C1: Standard BCE loss (unweighted)
- C2: Class-weighted BCE loss (the baseline)
- C3: Focal Loss (gamma=2, tuning alpha on validation data)
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
GROUP_B_SUMMARY = ROOT / "results/group_b_negative_sampling/negative_sampling_summary.csv"
OUT_DIR = ROOT / "results/group_c_loss"

# Import helper functions from Group B runner
import sys
sys.path.append(str(ROOT / "scripts"))
from run_group_b_negative_sampling import (
    GraphSAGEModel, build_snapshots, load_edge_index, sample_negatives_month, compute_metrics,
    SEEDS, HIDDEN_DIM, DROPOUT, LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, EXCLUDE_EXPLICIT
)

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        p = torch.sigmoid(logits)
        pt = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_factor * ((1.0 - pt) ** self.gamma)
        
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = focal_weight * bce
        if weights is not None:
            loss = loss * weights
        return loss.mean()

def run_experiment(
    ratio: int | None,
    loss_type: str,
    alpha: float | None,
    gamma: float | None,
    train_bundle, val_bundle, test_bundle,
    neighbor_map, csd_to_idx, cols
) -> tuple[dict, np.ndarray, np.ndarray]:
    
    val_pred_df = val_bundle.rows.copy()
    test_pred_df = test_bundle.rows.copy()
    ensemble_val_cols = []
    ensemble_test_cols = []
    
    seed_val_aps = []
    seed_test_aps = []
    seed_metrics = []
    
    train_labels = np.concatenate([snap.y[snap.mask].numpy() for snap in train_bundle.snapshots])
    train_pos = float(train_labels.sum())
    train_neg = float(len(train_labels) - train_pos)
    pos_weight = torch.tensor([train_neg / train_pos], dtype=torch.float32)

    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed + 200)

        model = GraphSAGEModel(in_channels=len(cols))
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_state = None
        best_val_ap = -1.0
        epochs_without_improvement = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for snap in train_bundle.snapshots:
                optimizer.zero_grad()
                logits = model(snap.x, snap.edge_index)
                
                pos_mask = (snap.y == 1) & snap.mask
                neg_candidates = (snap.y == 0) & snap.mask & snap.poultry_at_risk
                
                pos_idx = torch.where(pos_mask)[0].cpu().numpy()
                neg_idx = torch.where(neg_candidates)[0].cpu().numpy()
                
                if ratio is None or len(pos_idx) == 0:
                    loss_neg_idx = neg_idx
                    ipw = 1.0
                else:
                    loss_neg_idx, ipw = sample_negatives_month(
                        pos_idx, neg_idx, ratio, neighbor_map, snap.water_vecs.cpu().numpy(), rng
                    )
                
                loss_mask = np.zeros(len(snap.y), dtype=bool)
                loss_mask[pos_idx] = True
                loss_mask[loss_neg_idx] = True
                
                weights = torch.ones(len(snap.y), dtype=torch.float32)
                weights[loss_neg_idx] = ipw
                
                loss_m = loss_mask & snap.mask.cpu().numpy()
                if loss_m.sum() == 0:
                    continue
                    
                loss_logits = logits[loss_m]
                loss_targets = snap.y[loss_m]
                loss_weights = weights[loss_m]

                if loss_type == "BCE":
                    bce = F.binary_cross_entropy_with_logits(loss_logits, loss_targets, reduction='none')
                    loss = (bce * loss_weights).mean()
                elif loss_type == "weighted_BCE":
                    bce = F.binary_cross_entropy_with_logits(loss_logits, loss_targets, pos_weight=pos_weight, reduction='none')
                    loss = (bce * loss_weights).mean()
                elif loss_type == "Focal":
                    fl = FocalLoss(alpha=alpha, gamma=gamma)
                    loss = fl(loss_logits, loss_targets, loss_weights)
                
                loss.backward()
                optimizer.step()

            # Validation AP
            model.eval()
            val_probs = []
            with torch.no_grad():
                for snap in val_bundle.snapshots:
                    logits = model(snap.x, snap.edge_index)
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
                logits = model(snap.x, snap.edge_index)
                val_probs.extend(torch.sigmoid(logits)[snap.mask].cpu().numpy())
            for snap in test_bundle.snapshots:
                logits = model(snap.x, snap.edge_index)
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
    return summary, val_pred_df["ensemble_prob"].values, test_pred_df["ensemble_prob"].values

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Determine best configurations from Group B
    if not GROUP_B_SUMMARY.exists():
        # Fallback to predefined best configurations if Group B registry has not run
        print("Warning: Group B summary not found, using default best configs (all_poultry and ratio_25)")
        best_configs = [("B1_all_poultry", None), ("B4_ratio_25", 25)]
    else:
        df_b = pd.read_csv(GROUP_B_SUMMARY)
        # Find best two sampling models based on validation AP (if stored) or Ensemble/Mean AP as fallback
        # Let's see: `negative_sampling_summary.csv` has "Configuration" and metrics.
        # Since Group B uses Ensemble AP, we can sort by Ensemble AP.
        df_b = df_b.sort_values("Ensemble AP", ascending=False)
        best_names = df_b["Configuration"].head(2).tolist()
        print(f"Top 2 sampling configurations from Group B: {best_names}")
        
        ratio_map = {
            "B1_all_poultry": None,
            "B2_ratio_100": 100,
            "B3_ratio_50": 50,
            "B4_ratio_25": 25,
            "B5_ratio_10": 10,
            "B6_ratio_3": 3,
        }
        best_configs = []
        for name in best_names:
            best_configs.append((name, ratio_map.get(name, None)))

    # Load data
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
    edge_index, neighbor_map = load_edge_index(all_csds)

    train_bundle = build_snapshots(df_train, cols, csd_to_idx, edge_index)
    val_bundle = build_snapshots(df_val, cols, csd_to_idx, edge_index)
    test_bundle = build_snapshots(df_test, cols, csd_to_idx, edge_index)

    results = []

    # Run loss function experiments
    for config_name, ratio in best_configs:
        print(f"\n=========================================")
        print(f"  RUNNING LOSS ABLATION FOR: {config_name}")
        print(f"=========================================")
        
        # C1: BCE
        print("\n--- Loss: C1 (Standard unweighted BCE) ---")
        c1_sum, _, _ = run_experiment(ratio, "BCE", None, None, train_bundle, val_bundle, test_bundle, neighbor_map, csd_to_idx, cols)
        c1_sum["Sampling"] = config_name
        c1_sum["Loss"] = "C1_BCE"
        results.append(c1_sum)

        # C2: Class-weighted BCE
        print("\n--- Loss: C2 (Class-weighted BCE) ---")
        c2_sum, _, _ = run_experiment(ratio, "weighted_BCE", None, None, train_bundle, val_bundle, test_bundle, neighbor_map, csd_to_idx, cols)
        c2_sum["Sampling"] = config_name
        c2_sum["Loss"] = "C2_weighted_BCE"
        results.append(c2_sum)

        # C3: Focal Loss tuning alpha on validation AP
        print("\n--- Loss: C3 (Focal Loss Tuning alpha) ---")
        focal_candidates = []
        for alpha in [0.25, 0.50, 0.75, 0.85]:
            print(f"    Tuning Focal Loss with alpha={alpha}...")
            fl_sum, val_preds, test_preds = run_experiment(ratio, "Focal", alpha, 2.0, train_bundle, val_bundle, test_bundle, neighbor_map, csd_to_idx, cols)
            focal_candidates.append((alpha, fl_sum, val_preds, test_preds))
            
        # Select best alpha based on Ensemble Val AP
        focal_candidates = sorted(focal_candidates, key=lambda x: x[1]["Ensemble Val AP"], reverse=True)
        best_alpha, best_fl_sum, _, _ = focal_candidates[0]
        print(f"    Best alpha selected: {best_alpha} with Ensemble Val AP: {best_fl_sum['Ensemble Val AP']:.4f}")
        
        best_fl_sum["Sampling"] = config_name
        best_fl_sum["Loss"] = f"C3_Focal_alpha_{best_alpha}"
        results.append(best_fl_sum)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_DIR / "loss_comparison_summary.csv", index=False)
    print("\n=== Group C Loss Function Results ===")
    print(df_res.to_markdown(index=False))

if __name__ == "__main__":
    main()
