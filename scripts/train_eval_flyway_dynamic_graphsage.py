#!/usr/bin/env python3
"""
scripts/train_eval_flyway_dynamic_graphsage.py

Implements, trains, and evaluates the Flyway-Dynamic GraphSAGE model.
- Builds monthly dynamic edge weights W_ij(t) across 12 calendar months based on:
    W_ij(t) = alpha * A_ij + (1 - alpha) * Phi_ij * M(t)
    where intra-flyway edges (sharing same flyway basin) receive seasonal migration boost M(t),
    and cross-flyway edges remain attenuated across mountain/ecological barriers.
- Trains a 2-layer Weighted GraphSAGE model across 5 fixed seeds (42, 123, 456, 789, 2025).
- Evaluates on the locked 2025 test set under the controlled 1:5 pseudo-absence benchmark (1,000 resamples).
- Compares directly against the canonical static GraphSAGE baseline.
"""

import os
import copy
import random
from pathlib import Path
import numpy as np
import pandas as pd
import pyproj
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# ─────────────────────────────────────────────────────────────────────────────
# Paths & Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
GRAPH_PATH = ROOT / "data/processed/graph/csd_graph_knn_3_v1.csv"
CENTROIDS_PATH = ROOT / "data/processed/graph/csd_centroids_epsg3347.csv"

OUT_DIR = ROOT / "results/final_manuscript"
PREDS_DIR = OUT_DIR / "canonical_predictions"
PREDS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2025]
HIDDEN_DIM = 64
DROPOUT = 0.2
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 15

EXCLUDE_EXPLICIT = {
    "csd_uid", "csd_name", "province_code", "province_name", "ccs_uid",
    "year", "month", "outbreak_next_month",
    "dw_fill_method", "climate_extraction_method",
    "glw2020_chicken_extraction_method", "glw2015_duck_extraction_method",
    "road_data_coverage", "outbreak_current_month", "wildbird_hpai_present_next_month"
}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_migration_intensity(month: int) -> float:
    """Bimodal seasonal migration intensity function M(t)."""
    mu_spring, sigma_spring, kappa_spring = 4.5, 0.75, 1.0
    mu_fall, sigma_fall, kappa_fall = 9.5, 0.90, 0.85
    baseline = 0.08
    val = (baseline +
           kappa_spring * np.exp(-((month - mu_spring) ** 2) / (2 * (sigma_spring ** 2))) +
           kappa_fall * np.exp(-((month - mu_fall) ** 2) / (2 * (sigma_fall ** 2))))
    return float(np.clip(val, 0.0, 1.0))

def assign_flyways(df_centroids: pd.DataFrame) -> dict:
    """Maps each csd_uid to a flyway based on centroid longitude (EPSG:4326)."""
    transformer = pyproj.Transformer.from_crs("EPSG:3347", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(df_centroids["cx"].values, df_centroids["cy"].values)
    
    csd_flyways = {}
    for uid, lon in zip(df_centroids["csd_uid"].astype(str).str.strip(), lons):
        if lon < -120.0:
            csd_flyways[uid] = "Pacific"
        elif lon < -104.0:
            csd_flyways[uid] = "Central"
        elif lon < -85.0:
            csd_flyways[uid] = "Mississippi"
        else:
            csd_flyways[uid] = "Atlantic"
    return csd_flyways

# ─────────────────────────────────────────────────────────────────────────────
# Weighted GraphSAGE Architecture
# ─────────────────────────────────────────────────────────────────────────────
class WeightedSAGEConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias)  # Neighbor aggregation
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False) # Root node transformation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        src, tgt = edge_index[0], edge_index[1]
        num_nodes = x.size(0)

        if edge_weight is None:
            deg = torch.zeros(num_nodes, 1, device=x.device).scatter_add_(
                0, tgt.unsqueeze(-1), torch.ones(tgt.size(0), 1, device=x.device)
            ).clamp(min=1.0)
            msg = x[src]
            agg = torch.zeros(num_nodes, x.size(1), device=x.device).scatter_add_(
                0, tgt.unsqueeze(-1).expand(-1, x.size(1)), msg
            ) / deg
        else:
            w = edge_weight.unsqueeze(-1)
            norm = torch.zeros(num_nodes, 1, device=x.device).scatter_add_(
                0, tgt.unsqueeze(-1), w
            ).clamp(min=1e-8)
            msg = x[src] * w
            agg = torch.zeros(num_nodes, x.size(1), device=x.device).scatter_add_(
                0, tgt.unsqueeze(-1).expand(-1, x.size(1)), msg
            ) / norm

        return self.lin_l(agg) + self.lin_r(x)

class WeightedGraphSAGEModel(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = WeightedSAGEConv(in_channels, HIDDEN_DIM)
        self.conv2 = WeightedSAGEConv(HIDDEN_DIM, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        h = F.relu(self.conv2(h, edge_index, edge_weight))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        return self.out(h).squeeze(-1)

# ─────────────────────────────────────────────────────────────────────────────
# Snapshot Builder with Flyway Dynamic Edge Weights
# ─────────────────────────────────────────────────────────────────────────────
def build_flyway_snapshots(df: pd.DataFrame, feature_cols: list[str], edges_df: pd.DataFrame, 
                           scaler: StandardScaler, csd_flyways: dict, alpha: float = 0.70):
    snaps = []
    ym_pairs = df[["year", "month"]].drop_duplicates().sort_values(by=["year", "month"]).values
    
    # Pre-identify edge flyway relationship
    edges_work = edges_df.copy()
    src_uids = edges_work["source_csd_uid"].astype(str).str.strip()
    tgt_uids = edges_work["target_csd_uid"].astype(str).str.strip()
    
    is_intra = np.array([csd_flyways.get(u) == csd_flyways.get(v) for u, v in zip(src_uids, tgt_uids)])
    
    for y, m in ym_pairs:
        df_t = df[(df["year"] == y) & (df["month"] == m)].copy().reset_index(drop=True)
        X_raw = df_t[feature_cols].fillna(0.0).values
        X_scaled = scaler.transform(X_raw).astype(np.float32)
        y_t = df_t["outbreak_next_month"].values.astype(np.float32)

        uid_map = {uid: i for i, uid in enumerate(df_t["csd_uid"].astype(str).str.strip().values)}
        src = src_uids.map(uid_map).dropna().astype(int).values
        tgt = tgt_uids.map(uid_map).dropna().astype(int).values
        
        # Valid edge indices
        valid_mask = src_uids.isin(uid_map) & tgt_uids.isin(uid_map)
        edge_index = torch.tensor(np.stack([src, tgt], axis=0), dtype=torch.long)
        
        # Calculate monthly edge weights W_ij(m)
        m_phenology = compute_migration_intensity(m)
        
        # Intra-flyway: alpha + (1 - alpha) * M(t); Cross-flyway: alpha * 0.40
        w_values = np.where(is_intra[valid_mask], alpha + (1.0 - alpha) * m_phenology, alpha * 0.40).astype(np.float32)
        edge_weight = torch.from_numpy(w_values)

        snaps.append({
            "x": torch.from_numpy(X_scaled),
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "y": torch.from_numpy(y_t),
            "df_meta": df_t[["csd_uid", "year", "month", "outbreak_next_month"]].copy()
        })
    return snaps

def train_eval_flyway(train_snaps, val_snaps, test_snaps, in_channels: int, seed: int):
    set_seed(seed)
    model = WeightedGraphSAGEModel(in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_ap = -1.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for snap in train_snaps:
            optimizer.zero_grad()
            out = model(snap["x"], snap["edge_index"], snap["edge_weight"])
            loss = F.binary_cross_entropy_with_logits(out, snap["y"])
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds_list, val_y_list = [], []
        with torch.no_grad():
            for s in val_snaps:
                out = model(s["x"], s["edge_index"], s["edge_weight"])
                val_preds_list.extend(torch.sigmoid(out).cpu().numpy())
                val_y_list.extend(s["y"].numpy())

        val_ap = average_precision_score(val_y_list, val_preds_list) if sum(val_y_list) > 0 else 0
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_model_state)
    model.eval()

    test_preds_list = []
    with torch.no_grad():
        for s in test_snaps:
            out = model(s["x"], s["edge_index"], s["edge_weight"])
            test_preds_list.extend(torch.sigmoid(out).cpu().numpy())

    return np.array(test_preds_list, dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=== Starting Flyway-Dynamic GraphSAGE Experiment ===")
    
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    df_all = pd.read_csv(DATA_PATH, dtype={"csd_uid": str})
    df_all["csd_uid"] = df_all["csd_uid"].astype(str).str.strip()
    
    print(f"Loading edges from {GRAPH_PATH}...")
    edges_df = pd.read_csv(GRAPH_PATH, dtype=str)
    
    print(f"Loading centroids from {CENTROIDS_PATH}...")
    df_centroids = pd.read_csv(CENTROIDS_PATH, dtype={"csd_uid": str})
    csd_flyways = assign_flyways(df_centroids)
    
    flyway_counts = pd.Series(csd_flyways).value_counts()
    print("CSD distribution across continental flyways:")
    for fw, cnt in flyway_counts.items():
        print(f"  {fw:15}: {cnt:5d} CSDs")

    # Feature columns
    feature_cols = [c for c in df_all.columns if c not in EXCLUDE_EXPLICIT and pd.api.types.is_numeric_dtype(df_all[c])]
    feature_cols = sorted(feature_cols)
    print(f"Total features: {len(feature_cols)}")

    # Split data
    train_df = df_all[df_all["year"].isin([2022, 2023])].copy()
    val_df = df_all[df_all["year"] == 2024].copy()
    test_df = df_all[df_all["year"] == 2025].copy()

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].fillna(0.0).values)

    # 2. Build Flyway Snapshots
    print("Building Flyway-Dynamic spatio-temporal snapshots...")
    tr_snaps = build_flyway_snapshots(train_df, feature_cols, edges_df, scaler, csd_flyways)
    va_snaps = build_flyway_snapshots(val_df, feature_cols, edges_df, scaler, csd_flyways)
    te_snaps = build_flyway_snapshots(test_df, feature_cols, edges_df, scaler, csd_flyways)

    # 3. Train across 5 seeds
    print(f"Training Flyway-Dynamic GraphSAGE across 5 seeds: {SEEDS}...")
    all_seed_preds = []
    for seed in SEEDS:
        print(f"  Training Seed {seed}...")
        preds = train_eval_flyway(tr_snaps, va_snaps, te_snaps, len(feature_cols), seed)
        all_seed_preds.append(preds)
        print(f"  Seed {seed} complete.")

    ens_preds = np.mean(all_seed_preds, axis=0)

    # Assemble test metadata and save predictions
    df_test_meta = pd.concat([s["df_meta"] for s in te_snaps], axis=0).reset_index(drop=True)
    df_out = df_test_meta.copy()
    df_out["pred_prob_ensemble"] = ens_preds
    for i, s in enumerate(SEEDS):
        df_out[f"pred_prob_seed_{s}"] = all_seed_preds[i]

    out_preds_path = PREDS_DIR / "preds_GraphSAGE_flyway.csv"
    df_out.to_csv(out_preds_path, index=False)
    print(f"Saved Flyway-Dynamic GraphSAGE predictions to: {out_preds_path}")

    # 4. Evaluate under Controlled 1:5 Pseudo-Absence Benchmark
    print("\n--- Evaluating under Controlled 1:5 Pseudo-Absence Benchmark (1,000 Resamples) ---")
    farm_cols = ['ccs_broilers_roasters_cornish_farms_reporting', 'ccs_broilers_roasters_cornish_production_kg', 
                 'ccs_turkey_farms_reporting', 'ccs_turkey_production_kg']
    df_all['farm_sum'] = df_all[farm_cols].fillna(0).sum(axis=1)
    at_risk_csds = set(df_all.groupby('csd_uid')['farm_sum'].max().loc[lambda x: x > 0].index)
    eligible_mask = test_df['csd_uid'].isin(at_risk_csds) | (test_df['outbreak_next_month'] == 1)

    df_test_el = test_df[eligible_mask].copy().reset_index(drop=True)
    
    # Load canonical static GraphSAGE for exact paired comparison
    df_static = pd.read_csv(PREDS_DIR / "preds_GraphSAGE.csv", dtype={"csd_uid": str})
    df_static["csd_uid"] = df_static["csd_uid"].astype(str).str.strip()
    
    m_static = df_test_el.merge(df_static[["csd_uid", "year", "month", "pred_prob_ensemble"]], on=["csd_uid", "year", "month"], how="left")
    p_static = m_static["pred_prob_ensemble"].values

    m_flyway = df_test_el.merge(df_out[["csd_uid", "year", "month", "pred_prob_ensemble"]], on=["csd_uid", "year", "month"], how="left")
    p_flyway = m_flyway["pred_prob_ensemble"].values

    y = df_test_el['outbreak_next_month'].values
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    n_neg = n_pos * 5

    np.random.seed(42)
    n_iters = 1000

    static_ap_list = []
    flyway_ap_list = []
    delta_ap_list = []
    static_roc_list = []
    flyway_roc_list = []

    for _ in range(n_iters):
        chosen_neg = np.random.choice(neg_idx, size=n_neg, replace=False)
        sample_idx = np.concatenate([pos_idx, chosen_neg])
        y_s = y[sample_idx]

        ap_s = average_precision_score(y_s, p_static[sample_idx])
        ap_f = average_precision_score(y_s, p_flyway[sample_idx])
        roc_s = roc_auc_score(y_s, p_static[sample_idx])
        roc_f = roc_auc_score(y_s, p_flyway[sample_idx])

        static_ap_list.append(ap_s)
        flyway_ap_list.append(ap_f)
        delta_ap_list.append(ap_f - ap_s)
        static_roc_list.append(roc_s)
        flyway_roc_list.append(roc_f)

    delta_ap_arr = np.array(delta_ap_list)
    win_rate = np.mean(delta_ap_arr > 0) * 100.0

    print("=====================================================================")
    print("HEAD-TO-HEAD RESULTS: FLYWAY-DYNAMIC VS STATIC $kNN-3$ GRAPHSAGE")
    print("=====================================================================")
    print(f"Static GraphSAGE PR-AUC:         {np.mean(static_ap_list):.4f} +/- {np.std(static_ap_list):.4f}")
    print(f"Flyway-Dynamic GraphSAGE PR-AUC: {np.mean(flyway_ap_list):.4f} +/- {np.std(flyway_ap_list):.4f}")
    print(f"Delta PR-AUC (Flyway - Static):  {np.mean(delta_ap_arr):+.4f}")
    print(f"95% CI of Delta:                 [{np.percentile(delta_ap_arr, 2.5):+.4f}, {np.percentile(delta_ap_arr, 97.5):+.4f}]")
    print(f"Flyway Win Rate vs Static:       {win_rate:.1f}%")
    print("---------------------------------------------------------------------")
    print(f"Static GraphSAGE ROC-AUC:        {np.mean(static_roc_list):.4f}")
    print(f"Flyway-Dynamic GraphSAGE ROC-AUC:{np.mean(flyway_roc_list):.4f}")
    print("=====================================================================")

    # Full eligible domain metrics
    full_ap_static = average_precision_score(y, p_static)
    full_ap_flyway = average_precision_score(y, p_flyway)
    full_roc_static = roc_auc_score(y, p_static)
    full_roc_flyway = roc_auc_score(y, p_flyway)
    print("\nFULL ELIGIBLE DOMAIN (24,476 CSD-months, un-resampled):")
    print(f"Static GraphSAGE:   PR-AUC = {full_ap_static:.4f} | ROC-AUC = {full_roc_static:.4f}")
    print(f"Flyway GraphSAGE:   PR-AUC = {full_ap_flyway:.4f} | ROC-AUC = {full_roc_flyway:.4f}")

if __name__ == "__main__":
    main()
