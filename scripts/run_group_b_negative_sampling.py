#!/usr/bin/env python3
"""
Group B: High-Risk Negative Sampling for GraphSAGE.
Only changes training negative selection.
"""

from __future__ import annotations
import os
import gc
import json
import random
import copy
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
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
OUT_DIR = ROOT / "results/group_b_negative_sampling"
REGISTRY_PATH = ROOT / "results/experiment_registry.csv"

SEEDS = [11, 22, 33, 44, 55]
HIDDEN_DIM = 64
DROPOUT = 0.2
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 15

EXCLUDE_EXPLICIT = {
    "csd_uid", "csd_name", "province_code", "province_name", "ccs_uid",
    "year", "month", "outbreak_next_month", "dw_fill_method",
    "climate_extraction_method", "glw2020_chicken_extraction_method",
    "glw2015_duck_extraction_method", "road_data_coverage",
    "outbreak_current_month", "wildbird_hpai_present_next_month"
}

POULTRY_EXPOSURE_COLS = [
    'ccs_broilers_roasters_cornish_farms_reporting',
    'ccs_broilers_roasters_cornish_production_kg',
    'ccs_turkey_farms_reporting', 'ccs_turkey_production_kg',
    'glw2020_chicken_density_mean', 'glw2020_chicken_density_max',
    'glw2015_duck_density_mean', 'glw2015_duck_density_max'
]

WATER_HABITAT_COLS = [
    'dw_water_mean', 'dw_water_p95',
    'dw_flooded_vegetation_mean', 'dw_seasonal_pond_proxy'
]

class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        return self.lin(x).squeeze(-1)

@dataclass
class SnapshotBundle:
    snapshots: list[Data]
    rows: pd.DataFrame

def load_edge_index(all_csds: list[str]) -> tuple[torch.Tensor, dict[int, set[int]]]:
    edges = pd.read_csv(GRAPH_EDGE_CSV, dtype=str)
    csd_to_idx = {csd: idx for idx, csd in enumerate(all_csds)}
    src = edges["source_csd_uid"].map(csd_to_idx)
    dst = edges["target_csd_uid"].map(csd_to_idx)
    valid = src.notna() & dst.notna()
    src_arr = src[valid].astype(int).to_numpy()
    dst_arr = dst[valid].astype(int).to_numpy()
    
    neighbor_map = {}
    for s, d in zip(src_arr, dst_arr):
        neighbor_map.setdefault(s, set()).add(d)
        neighbor_map.setdefault(d, set()).add(s)
        
    edge_index = np.stack(
        [np.concatenate([src_arr, dst_arr]), np.concatenate([dst_arr, src_arr])],
        axis=0,
    )
    return torch.tensor(edge_index, dtype=torch.long), neighbor_map

def build_snapshots(
    df: pd.DataFrame,
    feature_cols: list[str],
    csd_to_idx: dict[str, int],
    edge_index: torch.Tensor,
) -> SnapshotBundle:
    snapshots: list[Data] = []
    ordered_rows: list[pd.DataFrame] = []
    num_nodes = len(csd_to_idx)

    poultry_farm_cols = [c for c in POULTRY_EXPOSURE_COLS if 'farms_reporting' in c or 'production_kg' in c]
    df['poultry_farm_sum'] = df[poultry_farm_cols].fillna(0.0).sum(axis=1)
    
    for (year, month), group in df.groupby(["year", "month"], sort=True):
        group = group.sort_values("csd_uid").reset_index(drop=True)
        node_idx = group["csd_uid"].map(csd_to_idx).to_numpy()

        x = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)
        y = np.zeros(num_nodes, dtype=np.float32)
        mask = np.zeros(num_nodes, dtype=bool)
        poultry_at_risk = np.zeros(num_nodes, dtype=bool)
        water_vecs = np.zeros((num_nodes, len(WATER_HABITAT_COLS)), dtype=np.float32)

        x[node_idx] = group[feature_cols].to_numpy(dtype=np.float32)
        y[node_idx] = group["outbreak_next_month"].to_numpy(dtype=np.float32)
        mask[node_idx] = True
        poultry_at_risk[node_idx] = (group["poultry_farm_sum"] > 0).to_numpy(dtype=bool)
        water_vecs[node_idx] = group[WATER_HABITAT_COLS].fillna(0.0).to_numpy(dtype=np.float32)

        snapshots.append(
            Data(
                x=torch.from_numpy(x),
                y=torch.from_numpy(y),
                mask=torch.from_numpy(mask),
                edge_index=edge_index,
                year=int(year),
                month=int(month),
                poultry_at_risk=torch.from_numpy(poultry_at_risk),
                water_vecs=torch.from_numpy(water_vecs),
            )
        )
        ordered_rows.append(
            group[["csd_uid", "csd_name", "province_code", "year", "month", "outbreak_next_month"]]
            .rename(columns={"outbreak_next_month": "actual"})
            .copy()
        )

    return SnapshotBundle(snapshots=snapshots, rows=pd.concat(ordered_rows, ignore_index=True))

def sample_negatives_month(
    pos_idx: np.ndarray,
    at_risk_neg_idx: np.ndarray,
    ratio: int,
    neighbor_map: dict[int, set[int]],
    water_vecs: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, float]:
    n_pos = len(pos_idx)
    n_neg_total = len(at_risk_neg_idx)
    n_target = min(ratio * n_pos, n_neg_total)
    if n_target <= 0 or n_neg_total == 0:
        return at_risk_neg_idx, 1.0

    n_spatial = n_target // 3
    n_env = n_target // 3
    n_rand = n_target - n_spatial - n_env

    sampled = set()

    # 1. Spatial neighbors
    spatial_candidates = set()
    for p in pos_idx:
        for neigh in neighbor_map.get(int(p), set()):
            if neigh in at_risk_neg_idx:
                spatial_candidates.add(neigh)
    
    spatial_candidates = list(spatial_candidates - sampled)
    if spatial_candidates:
        n_sp_act = min(n_spatial, len(spatial_candidates))
        sampled.update(rng.choice(spatial_candidates, size=n_sp_act, replace=False))

    # 2. Environmental matching
    env_candidates = set()
    if len(pos_idx) > 0 and len(at_risk_neg_idx) > 0:
        pos_w = water_vecs[pos_idx]
        neg_w = water_vecs[at_risk_neg_idx]
        
        pos_norms = np.linalg.norm(pos_w, axis=1, keepdims=True)
        pos_norms[pos_norms == 0] = 1e-8
        pos_w_norm = pos_w / pos_norms
        
        neg_norms = np.linalg.norm(neg_w, axis=1, keepdims=True)
        neg_norms[neg_norms == 0] = 1e-8
        neg_w_norm = neg_w / neg_norms
        
        cosine_sim = neg_w_norm @ pos_w_norm.T
        max_sim = cosine_sim.max(axis=1)
        
        n_env_act = min(n_env, len(at_risk_neg_idx))
        top_indices = np.argsort(max_sim)[::-1][:n_env_act * 2]
        for idx in top_indices:
            env_candidates.add(at_risk_neg_idx[idx])
            
    env_candidates = list(env_candidates - sampled)
    if env_candidates:
        n_env_act = min(n_env, len(env_candidates))
        sampled.update(rng.choice(env_candidates, size=n_env_act, replace=False))

    # 3. Random negatives
    remaining = n_target - len(sampled)
    rand_candidates = list(set(at_risk_neg_idx) - sampled)
    if remaining > 0 and rand_candidates:
        n_rand_act = min(remaining, len(rand_candidates))
        sampled.update(rng.choice(rand_candidates, size=n_rand_act, replace=False))

    sampled_neg = np.array(sorted(sampled), dtype=np.int64)
    ipw = float(n_neg_total) / float(len(sampled_neg)) if len(sampled_neg) > 0 else 1.0
    return sampled_neg, ipw

def evaluate_topk_monthly(df_eval: pd.DataFrame, k: int) -> tuple[float, float]:
    monthly_precision: list[float] = []
    monthly_recall: list[float] = []
    for (_, _), group in df_eval.groupby(["year", "month"], sort=True):
        group = group.sort_values("prob", ascending=False)
        top = group.head(k)
        monthly_precision.append(float(top["actual"].mean()))
        positives = float(group["actual"].sum())
        monthly_recall.append(float(top["actual"].sum() / positives) if positives > 0 else 0.0)
    return float(np.mean(monthly_precision)), float(np.mean(monthly_recall))

def compute_metrics(df_eval: pd.DataFrame) -> dict[str, float]:
    y_true = df_eval["actual"].to_numpy(dtype=np.float32)
    probs = df_eval["prob"].to_numpy(dtype=np.float32)
    p10, _ = evaluate_topk_monthly(df_eval, 10)
    _, r50 = evaluate_topk_monthly(df_eval, 50)
    return {
        "pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "macro_precision_at_10": p10,
        "macro_recall_at_50": r50,
        "brier_score": float(brier_score_loss(y_true, probs)),
    }

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
    edge_index, neighbor_map = load_edge_index(all_csds)

    train_bundle = build_snapshots(df_train, cols, csd_to_idx, edge_index)
    val_bundle = build_snapshots(df_val, cols, csd_to_idx, edge_index)
    test_bundle = build_snapshots(df_test, cols, csd_to_idx, edge_index)

    configs = {
        "B1_all_poultry": None,
        "B2_ratio_100": 100,
        "B3_ratio_50": 50,
        "B4_ratio_25": 25,
        "B5_ratio_10": 10,
        "B6_ratio_3": 3,
    }

    results = []

    for name, ratio in configs.items():
        print(f"\n--- Running Configuration: {name} (ratio={ratio}) ---")
        val_pred_df = val_bundle.rows.copy()
        test_pred_df = test_bundle.rows.copy()
        ensemble_val_cols = []
        ensemble_test_cols = []

        seed_metrics = []

        for seed in SEEDS:
            print(f"  Seed {seed}...")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            rng = np.random.RandomState(seed + 100)

            model = GraphSAGEModel(in_channels=len(cols))
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

            best_state = None
            best_val_ap = -1.0
            epochs_without_improvement = 0

            for epoch in range(1, MAX_EPOCHS + 1):
                model.train()
                
                # Setup monthly train masks
                for snap in train_bundle.snapshots:
                    optimizer.zero_grad()
                    logits = model(snap.x, snap.edge_index)
                    
                    pos_mask = (snap.y == 1) & snap.mask
                    neg_candidates = (snap.y == 0) & snap.mask & snap.poultry_at_risk
                    
                    pos_idx = torch.where(pos_mask)[0].cpu().numpy()
                    neg_idx = torch.where(neg_candidates)[0].cpu().numpy()
                    
                    if ratio is None or len(pos_idx) == 0:
                        # use all poultry negs
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
                    
                    bce = F.binary_cross_entropy_with_logits(loss_logits, loss_targets, reduction='none')
                    loss = (bce * loss_weights).mean()
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

            val_eval = val_bundle.rows.copy()
            val_eval["prob"] = val_probs
            test_eval = test_bundle.rows.copy()
            test_eval["prob"] = test_probs
            
            t_met = compute_metrics(test_eval)
            seed_metrics.append(t_met["pr_auc"])

        val_pred_df["ensemble_prob"] = val_pred_df[ensemble_val_cols].mean(axis=1)
        test_pred_df["ensemble_prob"] = test_pred_df[ensemble_test_cols].mean(axis=1)

        val_eval = val_bundle.rows.copy()
        val_eval["prob"] = val_pred_df["ensemble_prob"].values
        test_eval = test_bundle.rows.copy()
        test_eval["prob"] = test_pred_df["ensemble_prob"].values

        ens_val_metrics = compute_metrics(val_eval)
        ens_test_metrics = compute_metrics(test_eval)

        mean_ap = float(np.mean(seed_metrics))
        std_ap = float(np.std(seed_metrics))

        print(f"  Ensemble AP: {ens_test_metrics['pr_auc']:.4f} | Mean AP: {mean_ap:.4f} ± {std_ap:.4f}")

        results.append({
            "Configuration": name,
            "Mean AP": mean_ap,
            "Std AP": std_ap,
            "Ensemble AP": ens_test_metrics["pr_auc"],
            "ROC-AUC": ens_test_metrics["roc_auc"],
            "Recall@50": ens_test_metrics["macro_recall_at_50"],
            "Precision@10": ens_test_metrics["macro_precision_at_10"],
            "Brier": ens_test_metrics["brier_score"]
        })

        # Save prediction files
        val_pred_df.to_csv(OUT_DIR / f"{name}_val_predictions.csv", index=False)
        test_pred_df.to_csv(OUT_DIR / f"{name}_test_predictions.csv", index=False)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_DIR / "negative_sampling_summary.csv", index=False)
    print("\n=== Group B Negative Sampling Results ===")
    print(df_res.to_markdown(index=False))

if __name__ == "__main__":
    main()
