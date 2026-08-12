#!/usr/bin/env python3
"""
Group D: Self-Supervised Masked-Feature Pretraining for GraphSAGE.
- Pretraining: Masks 15% of features during training (2022-2023) and reconstructs them using MSE.
- Fine-tuning: Initialize encoder with SSL weights, fine-tune using class-weighted BCE loss on poultry labels.
- Compares:
  - D0: GraphSAGE from scratch (baseline architecture)
  - D1: SSL pretraining -> Poultry fine-tuning
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
OUT_DIR = ROOT / "results/group_d_ssl"

# Import helper functions from Group B runner
import sys
sys.path.append(str(ROOT / "scripts"))
from run_group_b_negative_sampling import (
    GraphSAGEModel, build_snapshots, load_edge_index, compute_metrics,
    SEEDS, HIDDEN_DIM, DROPOUT, LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, EXCLUDE_EXPLICIT
)

class SSLGraphSAGEModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        # Reconstruction head maps back to feature dimension
        self.reconstructor = nn.Linear(hidden_channels, in_channels)
        # Classification head maps to 1 logit
        self.lin = nn.Linear(hidden_channels, 1)

    def forward_encoder(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        return x

    def forward_recon(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.forward_encoder(x, edge_index)
        return self.reconstructor(h)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.forward_encoder(x, edge_index)
        return self.lin(h).squeeze(-1)

def run_ssl_experiment(
    do_pretrain: bool,
    train_bundle, val_bundle, test_bundle,
    edge_index, cols
) -> tuple[dict, np.ndarray, np.ndarray]:
    
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

    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed + 300)

        model = SSLGraphSAGEModel(in_channels=len(cols))

        # Phase 1: Self-Supervised Pretraining
        if do_pretrain:
            # Optimize encoder + reconstructor
            optimizer_pre = torch.optim.Adam(
                list(model.conv1.parameters()) + list(model.conv2.parameters()) + list(model.reconstructor.parameters()),
                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            mask_ratio = 0.15
            
            # Run pretraining on training snapshots
            for epoch in range(1, 31):
                model.train()
                for snap in train_bundle.snapshots:
                    optimizer_pre.zero_grad()
                    active_idx = torch.where(snap.mask)[0]
                    if len(active_idx) == 0:
                        continue
                    
                    x_masked = snap.x.clone()
                    # Element-wise masking on active nodes
                    mask = torch.rand(len(active_idx), len(cols)) < mask_ratio
                    mask_full = torch.zeros_like(snap.x, dtype=torch.bool)
                    mask_full[active_idx] = mask
                    x_masked[mask_full] = 0.0 # replace with mean
                    
                    recon = model.forward_recon(x_masked, snap.edge_index)
                    loss = F.mse_loss(recon[mask_full], snap.x[mask_full])
                    if not torch.isnan(loss):
                        loss.backward()
                        optimizer_pre.step()

        # Phase 2: Supervised Fine-Tuning
        optimizer_ft = torch.optim.Adam(
            list(model.conv1.parameters()) + list(model.conv2.parameters()) + list(model.lin.parameters()),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_state = None
        best_val_ap = -1.0
        epochs_without_improvement = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for snap in train_bundle.snapshots:
                optimizer_ft.zero_grad()
                logits = model(snap.x, snap.edge_index)
                loss = criterion(logits[snap.mask], snap.y[snap.mask])
                loss.backward()
                optimizer_ft.step()

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
        
        # Predict
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

    results = []

    # D0: Scratch
    print("\n--- Running D0: Scratch ---")
    d0_sum, _, _ = run_ssl_experiment(False, train_bundle, val_bundle, test_bundle, edge_index, cols)
    d0_sum["Configuration"] = "D0_scratch"
    results.append(d0_sum)

    # D1: SSL Pretrained
    print("\n--- Running D1: SSL Pretrained ---")
    d1_sum, _, _ = run_ssl_experiment(True, train_bundle, val_bundle, test_bundle, edge_index, cols)
    d1_sum["Configuration"] = "D1_ssl_pretrained"
    results.append(d1_sum)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_DIR / "ssl_summary_results.csv", index=False)
    print("\n=== Group D SSL Results ===")
    print(df_res.to_markdown(index=False))

if __name__ == "__main__":
    main()
