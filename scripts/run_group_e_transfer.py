#!/usr/bin/env python3
"""
Group E: Wild-Bird Supervised Transfer Learning for GraphSAGE.
- Stage 1: Train GraphSAGE to predict next-month wild-bird HPAI presence target using 2022-2023 training data.
- Stage 2: Transfer learned encoder weights, replace the classification head, and fine-tune on poultry.
- Compare:
  - E0: Poultry GraphSAGE from scratch (baseline)
  - E1: Wild-bird pretrain -> Poultry fine-tune
  - E2: Wild-bird pretrain -> Poultry fine-tune + best negative sampling
  - E3: Wild-bird pretrain -> Poultry fine-tune + best loss function
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
WB_RAW_CSV = ROOT / "data/raw/cfia/CFIA_outbreaks_wildbirds.csv"
WB_CROSSWALK = ROOT / "data/processed/geography/wildbird_cfia_ccs_crosswalk_reviewed.csv"
GROUP_B_SUMMARY = ROOT / "results/group_b_negative_sampling/negative_sampling_summary.csv"
GROUP_C_SUMMARY = ROOT / "results/group_c_loss/loss_comparison_summary.csv"
OUT_DIR = ROOT / "results/group_e_transfer"

# Import helper functions from Group B runner
import sys
sys.path.append(str(ROOT / "scripts"))
from run_group_b_negative_sampling import (
    GraphSAGEModel, build_snapshots, load_edge_index, sample_negatives_month, compute_metrics,
    SEEDS, HIDDEN_DIM, DROPOUT, LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, EXCLUDE_EXPLICIT
)
from run_group_c_loss import FocalLoss
from run_poultry_at_risk_negative_ablation import build_wildbird_target

def run_transfer_experiment(
    config_name: str,
    ratio: int | None,
    use_pretrain: bool,
    loss_type: str,
    alpha: float | None,
    gamma: float | None,
    train_bundle, val_bundle, test_bundle,
    neighbor_map, csd_to_idx, cols, y_wb_train
) -> tuple[dict, np.ndarray, np.ndarray]:
    
    val_pred_df = val_bundle.rows.copy()
    test_pred_df = test_bundle.rows.copy()
    ensemble_val_cols = []
    ensemble_test_cols = []
    
    seed_val_aps = []
    seed_test_aps = []
    
    # Pre-calculate positive weights
    train_labels = np.concatenate([snap.y[snap.mask].numpy() for snap in train_bundle.snapshots])
    train_pos = float(train_labels.sum())
    train_neg = float(len(train_labels) - train_pos)
    pos_weight_poultry = torch.tensor([train_neg / train_pos], dtype=torch.float32)

    # Wild-bird positive weight
    wb_pos = float(y_wb_train.sum())
    wb_neg = float(len(y_wb_train) - wb_pos)
    pos_weight_wb = torch.tensor([wb_neg / wb_pos], dtype=torch.float32) if wb_pos > 0 else torch.tensor([1.0])

    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed + 400)

        # Stage 1: Pretrain on Wild Bird
        pretrained_state = None
        if use_pretrain:
            wb_model = GraphSAGEModel(in_channels=len(cols))
            wb_optimizer = torch.optim.Adam(wb_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            wb_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_wb)
            
            # Simple pretraining for 15 epochs on wild-bird labels
            for epoch in range(1, 16):
                wb_model.train()
                wb_idx_offset = 0
                for snap in train_bundle.snapshots:
                    wb_optimizer.zero_grad()
                    logits = wb_model(snap.x, snap.edge_index)
                    active_mask = snap.mask.cpu().numpy()
                    
                    n_active = active_mask.sum()
                    y_wb_snap = torch.from_numpy(y_wb_train[wb_idx_offset:wb_idx_offset + n_active]).float()
                    wb_idx_offset += n_active
                    
                    loss = wb_criterion(logits[snap.mask], y_wb_snap)
                    loss.backward()
                    wb_optimizer.step()
            pretrained_state = copy.deepcopy(wb_model.state_dict())

        # Stage 2: Fine-tune on Poultry
        model = GraphSAGEModel(in_channels=len(cols))
        if use_pretrain and pretrained_state is not None:
            # Load conv layers and reinitialize prediction head
            model.load_state_dict(pretrained_state)
            nn.init.xavier_uniform_(model.lin.weight)
            nn.init.zeros_(model.lin.bias)

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

                if loss_type == "weighted_BCE":
                    bce = F.binary_cross_entropy_with_logits(loss_logits, loss_targets, pos_weight=pos_weight_poultry, reduction='none')
                    loss = (bce * loss_weights).mean()
                elif loss_type == "Focal":
                    fl = FocalLoss(alpha=alpha, gamma=gamma)
                    loss = fl(loss_logits, loss_targets, loss_weights)
                
                loss.backward()
                optimizer.step()

            # Validation step
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

    # 1. Determine best configurations from Group B and C
    best_ratio = None
    if GROUP_B_SUMMARY.exists():
        df_b = pd.read_csv(GROUP_B_SUMMARY)
        df_b = df_b.sort_values("Ensemble AP", ascending=False)
        best_b_name = df_b["Configuration"].iloc[0]
        ratio_map = {
            "B1_all_poultry": None,
            "B2_ratio_100": 100,
            "B3_ratio_50": 50,
            "B4_ratio_25": 25,
            "B5_ratio_10": 10,
            "B6_ratio_3": 3,
        }
        best_ratio = ratio_map.get(best_b_name, None)
        print(f"Best Group B ratio selected: {best_b_name} (ratio={best_ratio})")
    else:
        print("Warning: Group B summary not found, using default ratio = 25")
        best_ratio = 25

    best_loss = "weighted_BCE"
    best_alpha = None
    if GROUP_C_SUMMARY.exists():
        df_c = pd.read_csv(GROUP_C_SUMMARY)
        df_c = df_c.sort_values("Ensemble Test AP", ascending=False)
        best_c_loss = df_c["Loss"].iloc[0]
        print(f"Best Group C loss selected: {best_c_loss}")
        if "Focal" in best_c_loss:
            best_loss = "Focal"
            best_alpha = float(best_c_loss.split("alpha_")[1])
        else:
            best_loss = "weighted_BCE"
    else:
        print("Warning: Group C summary not found, using default loss = weighted_BCE")

    # Load data
    df = pd.read_csv(INPUT_MODEL_TABLE, dtype={"csd_uid": str})
    cols = sorted([c for c in df.columns if c not in EXCLUDE_EXPLICIT and pd.api.types.is_numeric_dtype(df[c])])
    df = df.sort_values(["year", "month", "csd_uid"]).reset_index(drop=True)

    # Build wild bird target dynamically
    print("Building wild-bird next-month presence target...")
    df_wb = build_wildbird_target(str(WB_RAW_CSV), str(WB_CROSSWALK), df)
    df = df.merge(df_wb, on=['csd_uid', 'year', 'month'], how='left').fillna({'wildbird_hpai_present_next_month': 0.0})

    df_train = df[df["year"].isin([2022, 2023])].copy()
    df_val = df[df["year"] == 2024].copy()
    df_test = df[df["year"] == 2025].copy()

    y_wb_train = df_train["wildbird_hpai_present_next_month"].values

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

    # E0: Poultry GraphSAGE from scratch
    print("\n--- Running E0: Scratch ---")
    e0_sum, _, _ = run_transfer_experiment("E0_scratch", None, False, "weighted_BCE", None, None, train_bundle, val_bundle, test_bundle, neighbor_map, csd_to_idx, cols, y_wb_train)
    e0_sum["Configuration"] = "E0_scratch"
    results.append(e0_sum)

    # E1: Wild-bird pretrain -> Poultry fine-tune (national negatives, weighted BCE)
    print("\n--- Running E1: Wild-bird pretrain -> Poultry fine-tune ---")
    e1_sum, _, _ = run_transfer_experiment("E1_transfer", None, True, "weighted_BCE", None, None, train_bundle, val_bundle, test_bundle, neighbor_map, csd_to_idx, cols, y_wb_train)
    e1_sum["Configuration"] = "E1_transfer"
    results.append(e1_sum)

    # E2: E1 + best negative sampling
    print(f"\n--- Running E2: E1 + negative sampling (ratio={best_ratio}) ---")
    e2_sum, _, _ = run_transfer_experiment(f"E2_transfer_sampling", best_ratio, True, "weighted_BCE", None, None, train_bundle, val_bundle, test_bundle, neighbor_map, csd_to_idx, cols, y_wb_train)
    e2_sum["Configuration"] = "E2_transfer_sampling"
    results.append(e2_sum)

    # E3: E1 + best loss function
    print(f"\n--- Running E3: E1 + loss function (loss={best_loss}, alpha={best_alpha}) ---")
    e3_sum, _, _ = run_transfer_experiment(f"E3_transfer_loss", None, True, best_loss, best_alpha, 2.0, train_bundle, val_bundle, test_bundle, neighbor_map, csd_to_idx, cols, y_wb_train)
    e3_sum["Configuration"] = "E3_transfer_loss"
    results.append(e3_sum)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_DIR / "transfer_summary_results.csv", index=False)
    print("\n=== Group E Transfer Results ===")
    print(df_res.to_markdown(index=False))

if __name__ == "__main__":
    main()
