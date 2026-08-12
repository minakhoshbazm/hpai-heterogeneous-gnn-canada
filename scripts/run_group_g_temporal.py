#!/usr/bin/env python3
"""
Group G: Temporal Modeling for GraphSAGE.
- G0: Standalone GraphSAGE (snapshot baseline)
- G1: GraphSAGE + GRU over previous 3 months
- G2: GraphSAGE + GRU over previous 6 months
- G3: GraphSAGE + Temporal Attention over previous 3-6 months (we will use 6 months)
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
OUT_DIR = ROOT / "results/group_g_temporal"

# Import helper functions from Group B runner
import sys
sys.path.append(str(ROOT / "scripts"))
from run_group_b_negative_sampling import (
    build_snapshots, load_edge_index, compute_metrics,
    SEEDS, HIDDEN_DIM, DROPOUT, LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, EXCLUDE_EXPLICIT
)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (V, seq_len, hidden_dim)
        u = torch.tanh(self.proj(x)) # (V, seq_len, hidden_dim)
        scores = self.w(u).squeeze(-1) # (V, seq_len)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1) # (V, seq_len, 1)
        return (x * weights).sum(dim=1) # (V, hidden_dim)

class SpatiotemporalModel(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = HIDDEN_DIM, temporal_type: str = "none", history_len: int = 0) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.temporal_type = temporal_type
        self.history_len = history_len
        
        if temporal_type == "gru":
            self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        elif temporal_type == "attention":
            self.attn = TemporalAttention(hidden_dim)
            
        self.lin = nn.Linear(hidden_dim, 1)

    def forward_gnn(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        return x

    def forward_temporal(self, seq_x: torch.Tensor) -> torch.Tensor:
        # seq_x shape: (V, seq_len, hidden_dim)
        if self.temporal_type == "gru":
            out, _ = self.gru(seq_x)
            h = out[:, -1, :] # final time step
        elif self.temporal_type == "attention":
            h = self.attn(seq_x)
        else:
            h = seq_x[:, -1, :]
        return self.lin(h).squeeze(-1)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_temporal_experiment(
    temporal_type: str,
    history_len: int,
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

    # We need parameter count from a dummy instantiation
    dummy_model = SpatiotemporalModel(len(cols), temporal_type=temporal_type, history_len=history_len)
    param_count = count_parameters(dummy_model)
    print(f"  Architecture: {temporal_type} (history={history_len}) | Parameter Count: {param_count:,}")

    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = SpatiotemporalModel(len(cols), temporal_type=temporal_type, history_len=history_len)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_state = None
        best_val_ap = -1.0
        epochs_without_improvement = 0

        # Helper to construct history tensor for a given snapshot index in a bundle
        def get_history_tensor(embs_list, idx, h_len):
            # embs_list is a list of tensors of shape (V, hidden_dim)
            seq = []
            for k in range(idx - h_len, idx + 1):
                if k < 0:
                    seq.append(torch.zeros_like(embs_list[0]))
                else:
                    seq.append(embs_list[k])
            return torch.stack(seq, dim=1) # (V, h_len + 1, hidden_dim)

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            # 1. Forward GNN for all snapshots first (spatially independent by month)
            train_embs = [model.forward_gnn(snap.x, snap.edge_index) for snap in train_bundle.snapshots]
            
            # 2. Backprop over each month's temporal loss
            optimizer.zero_grad()
            epoch_loss = 0
            for idx, snap in enumerate(train_bundle.snapshots):
                seq_x = get_history_tensor(train_embs, idx, history_len)
                logits = model.forward_temporal(seq_x)
                loss = criterion(logits[snap.mask], snap.y[snap.mask])
                loss.backward(retain_graph=True) # since train_embs are shared
                epoch_loss += loss.item()
            optimizer.step()

            # Validation step
            model.eval()
            with torch.no_grad():
                # We need to construct val embeddings. Note that to prevent leakage,
                # we concatenate train + val embeddings, and query history from them.
                train_embs_eval = [model.forward_gnn(snap.x, snap.edge_index) for snap in train_bundle.snapshots]
                val_embs = [model.forward_gnn(snap.x, snap.edge_index) for snap in val_bundle.snapshots]
                all_embs = train_embs_eval + val_embs
                
                val_probs = []
                for idx, snap in enumerate(val_bundle.snapshots):
                    global_idx = len(train_bundle.snapshots) + idx
                    seq_x = get_history_tensor(all_embs, global_idx, history_len)
                    logits = model.forward_temporal(seq_x)
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
        with torch.no_grad():
            train_embs_eval = [model.forward_gnn(snap.x, snap.edge_index) for snap in train_bundle.snapshots]
            val_embs_eval = [model.forward_gnn(snap.x, snap.edge_index) for snap in val_bundle.snapshots]
            test_embs = [model.forward_gnn(snap.x, snap.edge_index) for snap in test_bundle.snapshots]
            
            # Predict val
            val_probs = []
            all_val_embs = train_embs_eval + val_embs_eval
            for idx, snap in enumerate(val_bundle.snapshots):
                global_idx = len(train_bundle.snapshots) + idx
                seq_x = get_history_tensor(all_val_embs, global_idx, history_len)
                logits = model.forward_temporal(seq_x)
                val_probs.extend(torch.sigmoid(logits)[snap.mask].cpu().numpy())
                
            # Predict test
            test_probs = []
            all_test_embs = train_embs_eval + val_embs_eval + test_embs
            for idx, snap in enumerate(test_bundle.snapshots):
                global_idx = len(train_bundle.snapshots) + len(val_bundle.snapshots) + idx
                seq_x = get_history_tensor(all_test_embs, global_idx, history_len)
                logits = model.forward_temporal(seq_x)
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
        "Brier": ens_test_metrics["brier_score"],
        "Params": param_count
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

    # G0: snapshot GraphSAGE
    print("\n--- Running G0: Snapshot GraphSAGE ---")
    g0_sum, _, _ = run_temporal_experiment("none", 0, train_bundle, val_bundle, test_bundle, edge_index, cols)
    g0_sum["Configuration"] = "G0_snapshot"
    results.append(g0_sum)

    # G1: GraphSAGE + GRU over 3 months
    print("\n--- Running G1: GraphSAGE + GRU (3 months) ---")
    g1_sum, _, _ = run_temporal_experiment("gru", 3, train_bundle, val_bundle, test_bundle, edge_index, cols)
    g1_sum["Configuration"] = "G1_gru_3months"
    results.append(g1_sum)

    # G2: GraphSAGE + GRU over 6 months
    print("\n--- Running G2: GraphSAGE + GRU (6 months) ---")
    g2_sum, _, _ = run_temporal_experiment("gru", 6, train_bundle, val_bundle, test_bundle, edge_index, cols)
    g2_sum["Configuration"] = "G2_gru_6months"
    results.append(g2_sum)

    # G3: GraphSAGE + Temporal Attention over 6 months
    print("\n--- Running G3: GraphSAGE + Temporal Attention (6 months) ---")
    g3_sum, _, _ = run_temporal_experiment("attention", 6, train_bundle, val_bundle, test_bundle, edge_index, cols)
    g3_sum["Configuration"] = "G3_attention_6months"
    results.append(g3_sum)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_DIR / "temporal_summary_results.csv", index=False)
    print("\n=== Group G Temporal Modeling Results ===")
    print(df_res.to_markdown(index=False))

if __name__ == "__main__":
    main()
