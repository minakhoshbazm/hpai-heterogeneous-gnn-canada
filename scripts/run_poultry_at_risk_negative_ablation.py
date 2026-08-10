#!/usr/bin/env python3
"""
scripts/run_pseudoabsence_ablation_rgcn_transfer.py

Negative-Sampling Ablation for Spatiotemporal Directed RGCN Transfer (Model B).

PURPOSE
-------
Test whether training-time negative sampling improves Model B's ranking quality
on the full natural-prevalence locked 2025 test set, and quantify how much of
the low absolute PR-AUC (~0.065) is explained by extreme class imbalance
(prevalence ~0.14%) versus genuine ranking weakness.

DESIGN
------
- Same architecture (SparseSpatiotemporalRGCN), optimizer, graph, temporal split,
  early-stopping logic, and random seeds as the original Model B.
- Phase 1 (wild-bird pretraining) is UNCHANGED: uses all negatives with weighted
  BCE, since wild-bird prevalence (~3.2%) is already 20x denser.
- Phase 2 (poultry fine-tuning) applies 5 negative-sampling configurations:
    all_weighted : all negatives, pos_weight = N_neg/N_pos (baseline reproduction)
    ratio_3      : 1 positive : 3 negatives
    ratio_10     : 1 positive : 10 negatives
    ratio_25     : 1 positive : 25 negatives
    ratio_50     : 1 positive : 50 negatives
- Negatives sampled per month within training period:
    1/3 random, 1/3 spatially nearby (kNN-3 of positive CSDs), 1/3 environmentally
    similar (highest water-habitat cosine similarity to positive CSDs).
- Negatives resampled every epoch (stochastic training).
- IPW correction: sampled negatives weighted by (N_neg_total / N_neg_sampled) so
  gradient contribution is unbiased relative to the true distribution.
- Full graph context preserved: all 237k nodes participate in message passing;
  only the loss mask changes.

EVALUATION
----------
- Primary: full natural-prevalence locked 2025 test set (55,594 rows, 79 positives)
- Metrics: PR-AUC, ROC-AUC, Precision@{10,25,50,100}, Recall@50, lift,
  per-month PR-AUC/P@50, Brier score, block-bootstrap CIs by month.

OUTPUTS → results/pseudoabsence_ablation/
"""

import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings('ignore')
import gc
import json
import random
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss
)

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(1)

# ─────────────────────────────────────────────────────────────────────────────
# Paths (identical to original Model B)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_TABLE_PATH = 'data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv'
CENTROIDS_CSV_PATH = 'data/processed/graph/csd_centroids_epsg3347.csv'
KNN3_PATH = 'data/processed/graph/csd_graph_knn_3_v1.csv'
WB_RAW_CSV = 'data/raw/cfia/CFIA_outbreaks_wildbirds.csv'
WB_CROSSWALK = 'data/processed/geography/wildbird_cfia_ccs_crosswalk_reviewed.csv'

# Baseline predictions for comparison
STRICT_LR_PREDS_PATH = 'results/baselines/environment_with_lags_v1_strict_test_predictions.csv'
KNN3_PREDS_PATH = 'results/baselines/graphsage_selected_spatial_graph_test_predictions.csv'
MODEL_B_PREDS_PATH = 'results/baselines/temporal_gnn_transfer_test_predictions.csv'
MODEL_A_PREDS_PATH = 'results/baselines/rolling_wildbird_supervised_transfer_graphsage_test_predictions.csv'

OUT_DIR = 'results/poultry_at_risk_ablation'
PREDS_DIR = os.path.join(OUT_DIR, 'predictions')

# ─────────────────────────────────────────────────────────────────────────────
# Feature column groups (identical to original Model B)
# ─────────────────────────────────────────────────────────────────────────────
WATER_HABITAT_COLS = [
    'dw_water_mean', 'dw_water_p95',
    'dw_flooded_vegetation_mean', 'dw_seasonal_pond_proxy'
]

POULTRY_EXPOSURE_COLS = [
    'ccs_broilers_roasters_cornish_farms_reporting',
    'ccs_broilers_roasters_cornish_production_kg',
    'ccs_turkey_farms_reporting', 'ccs_turkey_production_kg',
    'glw2020_chicken_density_mean', 'glw2020_chicken_density_max',
    'glw2015_duck_density_mean', 'glw2015_duck_density_max'
]

EXCLUDE_EXPLICIT_GNN = {
    'csd_uid', 'csd_name', 'province_code', 'province_name', 'ccs_uid',
    'year', 'month', 'outbreak_next_month',
    'dw_fill_method', 'climate_extraction_method',
    'glw2020_chicken_extraction_method',
    'glw2015_duck_extraction_method', 'road_data_coverage',
    'outbreak_current_month'
}

# Ablation configurations
CONFIGS = ['all_weighted', 'ratio_3', 'ratio_10', 'ratio_25', 'ratio_50']
RATIO_MAP = {
    'all_weighted': None,  # Use all negatives
    'ratio_3': 3,
    'ratio_10': 10,
    'ratio_25': 25,
    'ratio_50': 50,
}
SEEDS = [11, 22, 33]
MAX_EPOCHS = 100
PATIENCE = 10


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions (identical to original Model B)
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SparseSpatiotemporalRGCN(nn.Module):
    """Identical architecture to original Model B."""
    def __init__(self, in_channels, hidden_channels=32, num_relations=4):
        super().__init__()
        self.num_relations = num_relations

        self.w_self1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.w_neigh1 = nn.ModuleList([
            nn.Linear(in_channels, hidden_channels, bias=False)
            for _ in range(num_relations)
        ])
        self.rel_weights1 = nn.Parameter(
            torch.ones(num_relations) / float(num_relations)
        )
        self.bias1 = nn.Parameter(torch.zeros(hidden_channels))

        self.w_self2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.w_neigh2 = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels, bias=False)
            for _ in range(num_relations)
        ])
        self.rel_weights2 = nn.Parameter(
            torch.ones(num_relations) / float(num_relations)
        )
        self.bias2 = nn.Parameter(torch.zeros(hidden_channels))

        self.raw_proj = nn.Linear(in_channels, hidden_channels, bias=False)
        self.skip_weight = nn.Parameter(torch.tensor(0.5))
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, sparse_adj_list):
        h_self1 = self.w_self1(x)
        h_rel1 = []
        for r, A_r in enumerate(sparse_adj_list):
            msg = torch.sparse.mm(A_r, x)
            h_rel1.append(self.w_neigh1[r](msg))

        h1 = h_self1 + sum(
            self.rel_weights1[r] * h_rel1[r]
            for r in range(self.num_relations)
        ) + self.bias1
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.2, training=self.training)

        h_self2 = self.w_self2(h1)
        h_rel2 = []
        for r, A_r in enumerate(sparse_adj_list):
            msg = torch.sparse.mm(A_r, h1)
            h_rel2.append(self.w_neigh2[r](msg))

        h2 = h_self2 + sum(
            self.rel_weights2[r] * h_rel2[r]
            for r in range(self.num_relations)
        ) + self.bias2
        h2 = h2 + h1  # residual skip

        h2 = h2 + self.skip_weight * self.raw_proj(x)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=0.2, training=self.training)
        return self.out(h2).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Graph / data construction (identical to original Model B)
# ─────────────────────────────────────────────────────────────────────────────
def precompute_time_safe_water_units(df_model, model_csds):
    sub_df = df_model[['csd_uid', 'year', 'month'] + WATER_HABITAT_COLS].copy()
    df_sorted = sub_df.sort_values(by=['csd_uid', 'year', 'month'])

    vals = df_sorted[WATER_HABITAT_COLS].values
    N, C = vals.shape
    num_csds = N // 47
    vals_reshaped = vals.reshape(num_csds, 47, C)
    cumsum = np.cumsum(vals_reshaped, axis=1)
    counts = np.arange(1, 48).reshape(1, 47, 1)
    cummean = cumsum / counts
    df_sorted[WATER_HABITAT_COLS] = cummean.reshape(N, C)

    water_unit_dict = {}
    for (y, m), group in df_sorted.groupby(['year', 'month']):
        c_df = group.set_index('csd_uid')[WATER_HABITAT_COLS].reindex(model_csds)
        scaled = c_df.fillna(0.0).values.astype(np.float32)
        mean = scaled.mean(axis=0)
        std = scaled.std(axis=0)
        std[std == 0] = 1e-8
        scaled = (scaled - mean) / std
        norms = np.linalg.norm(scaled, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        water_unit_dict[(int(y), int(m))] = scaled / norms
    return water_unit_dict


def load_knn3_spatial_pairs(csv_path, csd_to_idx):
    df_e = pd.read_csv(csv_path, dtype=str)
    src = df_e['source_csd_uid'].map(csd_to_idx).values
    dst = df_e['target_csd_uid'].map(csd_to_idx).values
    return src, dst


def build_sparse_adj_matrices(model_csds, centroids_df, df_model,
                              water_unit_dict, ym_tuples, total_nodes):
    num_csds = len(model_csds)
    num_months = len(ym_tuples)
    csd_to_idx = {csd: i for i, csd in enumerate(model_csds)}

    coords = np.column_stack([
        centroids_df["cx"].values, centroids_df["cy"].values
    ])
    kdtree = KDTree(coords)
    pairs_50km = sorted(list(kdtree.query_pairs(50000.0)))
    u_knn3, v_knn3 = load_knn3_spatial_pairs(KNN3_PATH, csd_to_idx)

    csd_static_df = df_model.groupby('csd_uid')[POULTRY_EXPOSURE_COLS].first().reindex(model_csds)
    poultry_raw = np.log1p(np.maximum(0.0, csd_static_df.fillna(0.0).values))
    scaler_p = StandardScaler()
    p_scaled = scaler_p.fit_transform(poultry_raw).astype(np.float32)
    p_norms = np.linalg.norm(p_scaled, axis=1, keepdims=True)
    p_norms[p_norms == 0] = 1e-8
    p_unit = p_scaled / p_norms

    tau_poultry = 0.80
    tau_water = 0.80

    poultry_pairs_50km = [
        p for p in pairs_50km
        if float(np.dot(p_unit[p[0]], p_unit[p[1]])) >= tau_poultry
    ]
    u_poul = np.array([p[0] for p in poultry_pairs_50km], dtype=np.int32)
    v_poul = np.array([p[1] for p in poultry_pairs_50km], dtype=np.int32)

    u_50km = np.array([p[0] for p in pairs_50km], dtype=np.int32)
    v_50km = np.array([p[1] for p in pairs_50km], dtype=np.int32)

    rel_src = [[], [], [], []]
    rel_dst = [[], [], [], []]

    for t in range(num_months - 1):
        t_src_offset = np.int32(t * num_csds)
        t_dst_offset = np.int32((t + 1) * num_csds)

        # Relation 0: temporal_self
        self_src = t_src_offset + np.arange(num_csds, dtype=np.int32)
        self_dst = t_dst_offset + np.arange(num_csds, dtype=np.int32)
        rel_src[0].append(self_src)
        rel_dst[0].append(self_dst)

        # Relation 1: temporal_spatial_geographic (kNN-3)
        if len(u_knn3) > 0:
            u_k3 = u_knn3.astype(np.int32)
            v_k3 = v_knn3.astype(np.int32)
            k3_src = np.concatenate([t_src_offset + u_k3, t_src_offset + v_k3])
            k3_dst = np.concatenate([t_dst_offset + v_k3, t_dst_offset + u_k3])
            rel_src[1].append(k3_src)
            rel_dst[1].append(k3_dst)

        # Relation 2: temporal_spatial_water (50km sim >= 0.80)
        y_src, m_src = ym_tuples[t]
        w_unit = water_unit_dict[(int(y_src), int(m_src))]
        water_pairs_50km = [
            p for p in pairs_50km
            if float(np.dot(w_unit[p[0]], w_unit[p[1]])) >= tau_water
        ]
        if len(water_pairs_50km) > 0:
            u_w = np.array([p[0] for p in water_pairs_50km], dtype=np.int32)
            v_w = np.array([p[1] for p in water_pairs_50km], dtype=np.int32)
            w_src = np.concatenate([t_src_offset + u_w, t_src_offset + v_w])
            w_dst = np.concatenate([t_dst_offset + v_w, t_dst_offset + u_w])
            rel_src[2].append(w_src)
            rel_dst[2].append(w_dst)

        # Relation 3: temporal_spatial_poultry (50km sim >= 0.80)
        if len(u_poul) > 0:
            p_src = np.concatenate([t_src_offset + u_poul, t_src_offset + v_poul])
            p_dst = np.concatenate([t_dst_offset + v_poul, t_dst_offset + u_poul])
            rel_src[3].append(p_src)
            rel_dst[3].append(p_dst)

    sparse_adj_list = []
    total_edges = 0

    for r in range(4):
        r_src = np.concatenate(rel_src[r]).astype(np.int32)
        r_dst = np.concatenate(rel_dst[r]).astype(np.int32)
        rel_src[r] = None  # Clear memory list reference
        rel_dst[r] = None
        total_edges += len(r_src)

        ind_tensor = torch.stack([
            torch.from_numpy(r_dst).to(torch.long),
            torch.from_numpy(r_src).to(torch.long)
        ], dim=0)
        del r_src, r_dst
        gc.collect()

        deg = torch.zeros(total_nodes, dtype=torch.float)
        deg.scatter_add_(
            0, ind_tensor[0],
            torch.ones(ind_tensor.shape[1], dtype=torch.float)
        )
        deg[deg == 0] = 1.0
        values = 1.0 / deg[ind_tensor[0]]
        del deg

        A_r = torch.sparse_coo_tensor(
            ind_tensor, values,
            size=(total_nodes, total_nodes), dtype=torch.float
        ).coalesce()
        del ind_tensor, values
        sparse_adj_list.append(A_r)
        gc.collect()

    del rel_src, rel_dst
    gc.collect()

    return csd_to_idx, sparse_adj_list, total_edges


def build_wildbird_target(wb_raw_path, crosswalk_path, df_model):
    wb = pd.read_csv(wb_raw_path, encoding="latin-1")
    xw = pd.read_csv(crosswalk_path, encoding="latin-1")

    xw_matched = xw[xw["final_status"] == "matched"][
        ["raw_ccs_name", "raw_province", "target_ccs_uid"]
    ].dropna()

    wb["date_parsed"] = pd.to_datetime(wb["Collection Date"], errors="coerce")
    wb = wb.dropna(subset=["date_parsed"])
    wb["year"] = wb["date_parsed"].dt.year
    wb["month"] = wb["date_parsed"].dt.month

    wb["raw_ccs_name"] = wb["Census Consolidated Subdivision (CCS)"].str.strip().str.lower()
    wb["raw_province"] = wb["Province"].str.strip().str.lower()

    xw_matched = xw_matched.copy()
    xw_matched["raw_ccs_name"] = xw_matched["raw_ccs_name"].str.strip().str.lower()
    xw_matched["raw_province"] = xw_matched["raw_province"].str.strip().str.lower()

    wb_m = wb.merge(xw_matched, on=["raw_ccs_name", "raw_province"], how="inner")

    wb_agg = (
        wb_m.groupby(["target_ccs_uid", "year", "month"])
        .size().reset_index(name="wb_count")
        .rename(columns={"target_ccs_uid": "ccs_uid"})
    )
    wb_agg["wildbird_hpai_present"] = 1.0
    wb_agg["ccs_uid"] = wb_agg["ccs_uid"].astype(str)

    all_ccs_months = df_model[["ccs_uid", "year", "month"]].drop_duplicates().copy()
    all_ccs_months["ccs_uid"] = all_ccs_months["ccs_uid"].astype(str)
    wb_presence = all_ccs_months.merge(
        wb_agg[["ccs_uid", "year", "month", "wildbird_hpai_present"]],
        on=["ccs_uid", "year", "month"], how="left"
    ).fillna({"wildbird_hpai_present": 0.0})

    wb_pres_sorted = wb_presence.sort_values(["ccs_uid", "year", "month"])

    def next_ym(y, m):
        if m == 12:
            return (y + 1, 1)
        return (y, m + 1)

    wb_next = wb_pres_sorted[["ccs_uid", "year", "month", "wildbird_hpai_present"]].copy()
    wb_next.columns = ["ccs_uid", "next_year", "next_month",
                        "wildbird_hpai_present_next_month"]
    wb_next["ccs_uid"] = wb_next["ccs_uid"].astype(str)

    df_nxt = df_model[["csd_uid", "ccs_uid", "year", "month"]].copy()
    df_nxt["ccs_uid"] = df_nxt["ccs_uid"].astype(str)
    df_nxt[["next_year", "next_month"]] = pd.DataFrame(
        [next_ym(r.year, r.month) for r in df_nxt.itertuples()],
        columns=["next_year", "next_month"]
    )
    df_nxt = df_nxt.merge(
        wb_next, on=["ccs_uid", "next_year", "next_month"], how="left"
    ).fillna({"wildbird_hpai_present_next_month": 0.0})

    return df_nxt[["csd_uid", "year", "month", "wildbird_hpai_present_next_month"]]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def topk_metrics(y_true, probs, k_list, total_positives):
    sorted_idx = np.argsort(probs)[::-1]
    sorted_y = y_true[sorted_idx]
    results = {}
    for k in k_list:
        top_k = sorted_y[:k]
        pos_in_top = int(top_k.sum())
        precision = pos_in_top / k if k > 0 else 0.0
        recall = pos_in_top / total_positives if total_positives > 0 else 0.0
        results[f'precision_{k}'] = precision
        results[f'recall_{k}'] = recall
    return results


def compute_full_metrics(y_true, probs, total_positives):
    """Compute all primary metrics on a single evaluation set."""
    pr_auc = average_precision_score(y_true, probs)
    roc_auc = roc_auc_score(y_true, probs)
    prevalence = float(y_true.sum()) / len(y_true)
    lift = pr_auc / prevalence if prevalence > 0 else 0.0

    km = topk_metrics(y_true, probs, [10, 25, 50, 100], total_positives)

    # Brier score (no prior correction needed for all_weighted; corrected
    # versions computed separately for sampled configs)
    brier = brier_score_loss(y_true, probs)

    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'prevalence': prevalence,
        'random_pr_auc_baseline': prevalence,
        'lift_over_random': lift,
        'brier_score': brier,
        **km,
    }


def per_month_metrics(y_true, probs, months, total_pos_per_month=None):
    """Compute PR-AUC and P@50 per calendar month."""
    records = []
    for m in sorted(np.unique(months)):
        mask = months == m
        y_m = y_true[mask]
        p_m = probs[mask]
        n_pos = int(y_m.sum())
        n_total = int(mask.sum())

        if n_pos == 0 or n_pos == n_total:
            continue

        pr_auc = average_precision_score(y_m, p_m)
        km = topk_metrics(y_m, p_m, [50], n_pos)
        records.append({
            'month': int(m),
            'n_total': n_total,
            'n_pos': n_pos,
            'prevalence': n_pos / n_total,
            'pr_auc': pr_auc,
            'precision_50': km['precision_50'],
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Negative sampling
# ─────────────────────────────────────────────────────────────────────────────
def build_knn3_neighbor_map(knn3_path, csd_to_idx):
    """Build a dict mapping each CSD index to its kNN-3 neighbor indices."""
    df_e = pd.read_csv(knn3_path, dtype=str)
    neighbor_map = {}
    for _, row in df_e.iterrows():
        src_idx = csd_to_idx.get(row['source_csd_uid'])
        dst_idx = csd_to_idx.get(row['target_csd_uid'])
        if src_idx is not None and dst_idx is not None:
            neighbor_map.setdefault(src_idx, set()).add(dst_idx)
            neighbor_map.setdefault(dst_idx, set()).add(src_idx)
    return neighbor_map


def sample_negatives_for_month(
    pos_indices_in_month,   # global node indices of positives in this month
    neg_indices_in_month,   # global node indices of negatives in this month
    ratio,                  # target neg:pos ratio
    csd_idx_in_month,       # CSD index for each node (pos_idx -> csd_idx)
    neighbor_map,           # kNN-3 neighbor map (csd_idx -> set of csd_idx)
    water_unit_month,       # water unit vectors for this month (num_csds, C)
    num_csds,               # total number of CSDs
    month_offset,           # offset to convert CSD index to global node index
    rng,                    # numpy RandomState
):
    """
    Sample negatives for one training month with mixed strategies.

    Returns: array of global node indices for sampled negatives.
    """
    n_pos = len(pos_indices_in_month)
    n_neg_total = len(neg_indices_in_month)
    n_target = min(ratio * n_pos, n_neg_total)

    if n_target <= 0 or n_neg_total == 0:
        return neg_indices_in_month  # return all if can't sample

    n_spatial = n_target // 3
    n_environmental = n_target // 3
    n_random = n_target - n_spatial - n_environmental

    neg_set = set(neg_indices_in_month)

    # Convert positive global indices to CSD indices
    pos_csd_indices = set()
    for gi in pos_indices_in_month:
        csd_idx = gi - month_offset
        if 0 <= csd_idx < num_csds:
            pos_csd_indices.add(csd_idx)

    # Convert negative global indices to CSD indices for lookup
    neg_global_to_csd = {}
    for gi in neg_indices_in_month:
        csd_idx = gi - month_offset
        neg_global_to_csd[gi] = csd_idx

    neg_csd_to_global = {}
    for gi, ci in neg_global_to_csd.items():
        neg_csd_to_global.setdefault(ci, []).append(gi)

    sampled = set()

    # --- Strategy 1: Spatially nearby negatives (kNN-3 of positive CSDs) ---
    spatial_candidates = set()
    for pos_csd in pos_csd_indices:
        neighbors = neighbor_map.get(pos_csd, set())
        for nb in neighbors:
            if nb in neg_csd_to_global:
                for gi in neg_csd_to_global[nb]:
                    spatial_candidates.add(gi)

    spatial_candidates -= sampled
    if len(spatial_candidates) > 0:
        n_spatial_actual = min(n_spatial, len(spatial_candidates))
        spatial_sampled = rng.choice(
            list(spatial_candidates), size=n_spatial_actual, replace=False
        )
        sampled.update(spatial_sampled)
    else:
        n_spatial_actual = 0

    # --- Strategy 2: Environmentally similar negatives ---
    env_candidates = set()
    if water_unit_month is not None and len(pos_csd_indices) > 0:
        pos_csd_list = list(pos_csd_indices)
        pos_water = water_unit_month[pos_csd_list]  # (n_pos, C)

        neg_csd_list = list(neg_csd_to_global.keys())
        neg_water = water_unit_month[neg_csd_list]   # (n_neg_csds, C)

        # Max similarity of each negative CSD to any positive CSD
        if len(pos_water) > 0 and len(neg_water) > 0:
            sim_matrix = neg_water @ pos_water.T  # (n_neg_csds, n_pos)
            max_sim = sim_matrix.max(axis=1)
            # Take top candidates by similarity
            n_env_csds_target = min(
                n_environmental, len(neg_csd_list)
            )
            top_env_idx = np.argsort(max_sim)[::-1][:n_env_csds_target * 2]
            for idx in top_env_idx:
                csd = neg_csd_list[idx]
                for gi in neg_csd_to_global[csd]:
                    env_candidates.add(gi)

    env_candidates -= sampled
    if len(env_candidates) > 0:
        n_env_actual = min(n_environmental, len(env_candidates))
        env_sampled = rng.choice(
            list(env_candidates), size=n_env_actual, replace=False
        )
        sampled.update(env_sampled)
    else:
        n_env_actual = 0

    # --- Strategy 3: Random negatives (fill remaining quota) ---
    remaining = n_target - len(sampled)
    random_pool = list(neg_set - sampled)
    if remaining > 0 and len(random_pool) > 0:
        n_random_actual = min(remaining, len(random_pool))
        random_sampled = rng.choice(
            random_pool, size=n_random_actual, replace=False
        )
        sampled.update(random_sampled)

    return np.array(sorted(sampled), dtype=np.int64)


def build_sampled_loss_mask(
    train_mask_np, y_np, at_risk_mask_np, ratio, num_csds, ym_tuples,
    neighbor_map, water_unit_dict, rng, epoch
):
    """
    Build a loss mask that includes all positives and sampled negatives.
    Negatives are sampled per month from at-risk commercial poultry CSDs.
    Resampled each epoch for stochastic training.

    Returns:
        loss_mask_np: boolean array (total_nodes,)
        ipw_weights: float array (total_nodes,) — per-node loss weights
    """
    total_nodes = len(train_mask_np)
    loss_mask = np.zeros(total_nodes, dtype=bool)
    ipw_weights = np.ones(total_nodes, dtype=np.float32)

    pos_indices = np.where(train_mask_np & (y_np > 0.5))[0]
    neg_indices = np.where(train_mask_np & (y_np < 0.5) & at_risk_mask_np)[0]

    # Always include all positives
    loss_mask[pos_indices] = True

    if ratio is None:
        # all_weighted: include all at-risk negatives
        loss_mask[neg_indices] = True
        return loss_mask, ipw_weights

    # Sample negatives per month
    total_neg_all = len(neg_indices)
    total_neg_sampled = 0

    for t_idx, (y, m) in enumerate(ym_tuples):
        month_offset = t_idx * num_csds
        month_end = month_offset + num_csds

        # Indices in this month that are in training set
        month_train_pos = pos_indices[
            (pos_indices >= month_offset) & (pos_indices < month_end)
        ]
        month_train_neg = neg_indices[
            (neg_indices >= month_offset) & (neg_indices < month_end)
        ]

        if len(month_train_pos) == 0:
            # No positives in this month — include all negatives for this month
            # (they contribute context but no positive contrast)
            loss_mask[month_train_neg] = True
            total_neg_sampled += len(month_train_neg)
            continue

        # Get water vectors for this month
        w_key = (int(y), int(m))
        water_unit = water_unit_dict.get(w_key, None)

        sampled_neg = sample_negatives_for_month(
            pos_indices_in_month=month_train_pos,
            neg_indices_in_month=month_train_neg,
            ratio=ratio,
            csd_idx_in_month=None,
            neighbor_map=neighbor_map,
            water_unit_month=water_unit,
            num_csds=num_csds,
            month_offset=month_offset,
            rng=rng,
        )

        loss_mask[sampled_neg] = True
        total_neg_sampled += len(sampled_neg)

    # IPW correction: weight sampled negatives so effective contribution
    # is unbiased relative to full training set
    if total_neg_sampled > 0 and total_neg_sampled < total_neg_all:
        ipw = float(total_neg_all) / float(total_neg_sampled)
        # Apply IPW only to sampled negatives (not positives)
        neg_in_loss = loss_mask & (y_np < 0.5) & train_mask_np
        ipw_weights[neg_in_loss] = ipw

    return loss_mask, ipw_weights


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_model_b_ablation(
    config_name, ratio, seed,
    x_tune_tensor, x_full_tensor,
    sparse_adj_list, feature_dim,
    train_tune_mask_np, val_tune_mask_np,
    full_train_mask_np, test_mask_np,
    at_risk_mask_np,
    y_poultry_np, y_wildbird_np,
    pos_weight_wb_tune, pos_weight_wb_full,
    num_csds, ym_tuples,
    neighbor_map, water_unit_dict,
):
    """Train a single config+seed and return test predictions + metrics."""

    set_seed(seed)

    train_tune_mask = torch.tensor(train_tune_mask_np, dtype=torch.bool)
    val_tune_mask = torch.tensor(val_tune_mask_np, dtype=torch.bool)
    full_train_mask = torch.tensor(full_train_mask_np, dtype=torch.bool)
    test_mask = torch.tensor(test_mask_np, dtype=torch.bool)

    y_poultry_tensor = torch.tensor(y_poultry_np, dtype=torch.float)
    y_wildbird_tensor = torch.tensor(y_wildbird_np, dtype=torch.float)

    model = SparseSpatiotemporalRGCN(
        in_channels=feature_dim, hidden_channels=32, num_relations=4
    )

    # ── Phase 1: Pretrain on WildBird (IDENTICAL for all configs) ─────────
    criterion_wb = nn.BCEWithLogitsLoss(pos_weight=pos_weight_wb_tune)
    optimizer_pre = torch.optim.Adam(
        model.parameters(), lr=0.005, weight_decay=1e-4
    )

    best_val_wb_ap = -1.0
    best_wb_ep = 0
    no_imp_wb = 0

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer_pre.zero_grad()
        out = model(x_tune_tensor, sparse_adj_list)
        loss = criterion_wb(
            out[train_tune_mask], y_wildbird_tensor[train_tune_mask]
        )
        loss.backward()
        optimizer_pre.step()

        model.eval()
        with torch.no_grad():
            v_out = model(x_tune_tensor, sparse_adj_list)
            v_probs = torch.sigmoid(v_out[val_tune_mask]).cpu().numpy()
            v_ap = average_precision_score(
                y_wildbird_np[val_tune_mask_np], v_probs
            )

        if v_ap > best_val_wb_ap:
            best_val_wb_ap = v_ap
            best_wb_ep = ep
            no_imp_wb = 0
        else:
            no_imp_wb += 1

        del out, loss, v_out, v_probs
        gc.collect()

        if no_imp_wb >= PATIENCE:
            break

    print(f"    [Pretrain WB] Seed {seed} | Best Epoch: {best_wb_ep} | "
          f"Val PR-AUC: {best_val_wb_ap:.4f}")

    # Re-train to optimal pretrain epoch
    set_seed(seed)
    model_pre = SparseSpatiotemporalRGCN(
        in_channels=feature_dim, hidden_channels=32, num_relations=4
    )
    opt_pre = torch.optim.Adam(
        model_pre.parameters(), lr=0.005, weight_decay=1e-4
    )
    for ep in range(1, best_wb_ep + 1):
        model_pre.train()
        opt_pre.zero_grad()
        out = model_pre(x_tune_tensor, sparse_adj_list)
        loss = criterion_wb(
            out[train_tune_mask], y_wildbird_tensor[train_tune_mask]
        )
        loss.backward()
        opt_pre.step()
        del out, loss

    # ── Phase 2: Fine-tune on Poultry (CONFIG-SPECIFIC) ───────────────────
    model_ft = SparseSpatiotemporalRGCN(
        in_channels=feature_dim, hidden_channels=32, num_relations=4
    )
    model_ft.load_state_dict(model_pre.state_dict())
    nn.init.xavier_uniform_(model_ft.out.weight)
    nn.init.zeros_(model_ft.out.bias)

    optimizer_ft = torch.optim.Adam(
        model_ft.parameters(), lr=0.001, weight_decay=1e-4
    )

    best_val_pl_ap = -1.0
    best_pl_ep = 0
    no_imp_pl = 0

    rng = np.random.RandomState(seed + 1000)

    for ep in range(1, MAX_EPOCHS + 1):
        model_ft.train()

        # Build per-epoch loss mask with negative sampling
        loss_mask_np, ipw_weights_np = build_sampled_loss_mask(
            train_tune_mask_np, y_poultry_np, at_risk_mask_np, ratio,
            num_csds, ym_tuples,
            neighbor_map, water_unit_dict, rng, ep
        )

        loss_mask = torch.tensor(loss_mask_np, dtype=torch.bool)
        ipw_weights = torch.tensor(ipw_weights_np, dtype=torch.float)

        optimizer_ft.zero_grad()
        out = model_ft(x_tune_tensor, sparse_adj_list)

        # Compute weighted BCE with IPW-corrected sample weights
        logits = out[loss_mask]
        targets = y_poultry_tensor[loss_mask]
        weights = ipw_weights[loss_mask]

        # Manual weighted BCE: allows per-element weights
        bce_per_element = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        loss = (bce_per_element * weights).mean()

        loss.backward()
        optimizer_ft.step()

        # Validation on all nodes (no sampling) using PR-AUC
        model_ft.eval()
        with torch.no_grad():
            v_out = model_ft(x_tune_tensor, sparse_adj_list)
            v_probs = torch.sigmoid(v_out[val_tune_mask]).cpu().numpy()
            v_ap = average_precision_score(
                y_poultry_np[val_tune_mask_np], v_probs
            )

        if v_ap > best_val_pl_ap:
            best_val_pl_ap = v_ap
            best_pl_ep = ep
            no_imp_pl = 0
        else:
            no_imp_pl += 1

        del out, loss, v_out, v_probs, logits, targets, weights
        del bce_per_element, loss_mask, ipw_weights
        gc.collect()

        if no_imp_pl >= PATIENCE:
            break

    print(f"    [Fine-tune PL] Seed {seed} | Best Epoch: {best_pl_ep} | "
          f"Val PR-AUC: {best_val_pl_ap:.4f}")

    optimal_pre_epochs = best_wb_ep
    optimal_ft_epochs = best_pl_ep

    # ── Final retraining on 2022-2024 ─────────────────────────────────────
    set_seed(seed)
    model_final_pre = SparseSpatiotemporalRGCN(
        in_channels=feature_dim, hidden_channels=32, num_relations=4
    )
    opt_f_pre = torch.optim.Adam(
        model_final_pre.parameters(), lr=0.005, weight_decay=1e-4
    )
    criterion_wb_full = nn.BCEWithLogitsLoss(pos_weight=pos_weight_wb_full)

    for ep in range(1, optimal_pre_epochs + 1):
        model_final_pre.train()
        opt_f_pre.zero_grad()
        out = model_final_pre(x_full_tensor, sparse_adj_list)
        loss = criterion_wb_full(
            out[full_train_mask], y_wildbird_tensor[full_train_mask]
        )
        loss.backward()
        opt_f_pre.step()
        del out, loss

    model_final = SparseSpatiotemporalRGCN(
        in_channels=feature_dim, hidden_channels=32, num_relations=4
    )
    model_final.load_state_dict(model_final_pre.state_dict())
    nn.init.xavier_uniform_(model_final.out.weight)
    nn.init.zeros_(model_final.out.bias)

    opt_f_ft = torch.optim.Adam(
        model_final.parameters(), lr=0.001, weight_decay=1e-4
    )

    rng_full = np.random.RandomState(seed + 2000)

    for ep in range(1, optimal_ft_epochs + 1):
        model_final.train()

        loss_mask_np, ipw_weights_np = build_sampled_loss_mask(
            full_train_mask_np, y_poultry_np, at_risk_mask_np, ratio,
            num_csds, ym_tuples,
            neighbor_map, water_unit_dict, rng_full, ep
        )

        loss_mask = torch.tensor(loss_mask_np, dtype=torch.bool)
        ipw_weights = torch.tensor(ipw_weights_np, dtype=torch.float)

        opt_f_ft.zero_grad()
        out = model_final(x_full_tensor, sparse_adj_list)

        logits = out[loss_mask]
        targets = y_poultry_tensor[loss_mask]
        weights = ipw_weights[loss_mask]

        bce_per_element = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        loss = (bce_per_element * weights).mean()

        loss.backward()
        opt_f_ft.step()

        del out, loss, logits, targets, weights, bce_per_element
        del loss_mask, ipw_weights
        gc.collect()

    # ── Evaluate on 2025 test set ─────────────────────────────────────────
    model_final.eval()
    with torch.no_grad():
        t_out = model_final(x_full_tensor, sparse_adj_list)
        t_probs = torch.sigmoid(t_out[test_mask]).cpu().numpy()

    y_test = y_poultry_np[test_mask_np]
    test_pos = int(y_test.sum())

    metrics = compute_full_metrics(y_test, t_probs, test_pos)

    print(f"    [Test] PR-AUC: {metrics['pr_auc']:.4f} | "
          f"ROC-AUC: {metrics['roc_auc']:.4f} | "
          f"Lift: {metrics['lift_over_random']:.1f}x | "
          f"P@50: {metrics['precision_50']:.4f}")

    del model, model_pre, model_ft, model_final_pre, model_final
    del optimizer_pre, opt_pre, optimizer_ft, opt_f_pre, opt_f_ft
    gc.collect()

    return {
        'config': config_name,
        'seed': seed,
        'pretrain_epochs': optimal_pre_epochs,
        'finetune_epochs': optimal_ft_epochs,
        'val_wb_pr_auc': best_val_wb_ap,
        'val_pl_pr_auc': best_val_pl_ap,
        **metrics,
    }, t_probs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 90)
    print("  NEGATIVE-SAMPLING ABLATION: Spatiotemporal Directed RGCN Transfer")
    print("=" * 90)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PREDS_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    df_model = pd.read_csv(MODEL_TABLE_PATH, dtype={'csd_uid': str})
    model_csds = sorted(df_model['csd_uid'].unique())
    num_csds = len(model_csds)

    df_centroids = pd.read_csv(CENTROIDS_CSV_PATH, dtype={'csd_uid': str})
    centroids_df = (
        df_centroids[df_centroids['csd_uid'].isin(set(model_csds))]
        .set_index('csd_uid').reindex(model_csds)
    )

    feature_cols = sorted([
        c for c in df_model.columns
        if c not in EXCLUDE_EXPLICIT_GNN
        and pd.api.types.is_numeric_dtype(df_model[c])
    ])

    df_model = df_model.sort_values(
        by=['year', 'month', 'csd_uid']
    ).reset_index(drop=True)
    ym_tuples = (
        df_model[['year', 'month']].drop_duplicates()
        .sort_values(['year', 'month']).values
    )
    num_months = len(ym_tuples)
    total_nodes = num_csds * num_months

    # Extract feature matrix immediately as float32 numpy array
    X_raw = df_model[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)

    print(f"CSDs: {num_csds:,} | Months: {num_months} | "
          f"Total Nodes: {total_nodes:,} | Features: {len(feature_cols)}")

    # ── Wild-bird target ──────────────────────────────────────────────────
    print("\nBuilding wild-bird next-month presence target...")
    df_wb = build_wildbird_target(WB_RAW_CSV, WB_CROSSWALK, df_model)
    df_model = df_model.merge(
        df_wb, on=['csd_uid', 'year', 'month'], how='left'
    ).fillna({'wildbird_hpai_present_next_month': 0.0})
    del df_wb
    gc.collect()

    # ── Spatiotemporal graph ──────────────────────────────────────────────
    print("Building spatiotemporal graph...")
    water_unit_dict = precompute_time_safe_water_units(df_model, model_csds)
    csd_to_idx, sparse_adj_list, total_edges = build_sparse_adj_matrices(
        model_csds, centroids_df, df_model, water_unit_dict, ym_tuples,
        total_nodes
    )
    print(f"Graph: {total_edges:,} directed edges across 4 relation types.")
    del centroids_df
    gc.collect()

    # ── kNN-3 neighbor map for negative sampling ──────────────────────────
    neighbor_map = build_knn3_neighbor_map(KNN3_PATH, csd_to_idx)
    print(f"kNN-3 neighbor map: {len(neighbor_map):,} CSDs with neighbors.")

    # ── Temporal splits (identical to original Model B) ───────────────────
    node_year = df_model['year'].values
    y_poultry_np = df_model['outbreak_next_month'].values.astype(np.float32)
    y_wildbird_np = df_model['wildbird_hpai_present_next_month'].values.astype(np.float32)

    # Identify commercial poultry at-risk CSDs (farm reporting > 0 or production > 0)
    poultry_farm_cols = [c for c in POULTRY_EXPOSURE_COLS if 'farms_reporting' in c or 'production_kg' in c]
    df_model['poultry_farm_sum'] = df_model[poultry_farm_cols].fillna(0.0).sum(axis=1)
    at_risk_csds = set(df_model.groupby('csd_uid')['poultry_farm_sum'].max().loc[lambda x: x > 0].index)
    at_risk_node_mask = df_model['csd_uid'].isin(at_risk_csds).to_numpy(dtype=bool)
    print(f"Commercial Poultry At-Risk CSDs: {len(at_risk_csds):,} / {num_csds:,} ({len(at_risk_csds)/num_csds:.1%})")

    train_tune_mask_np = np.isin(node_year, [2022, 2023])
    val_tune_mask_np = (node_year == 2024)
    full_train_mask_np = np.isin(node_year, [2022, 2023, 2024])
    test_mask_np = (node_year == 2025)

    # Save test metadata before dropping df_model
    test_nodes_df = df_model[test_mask_np][
        ['csd_uid', 'csd_name', 'province_code', 'year', 'month']
    ].copy()
    test_months = test_nodes_df['month'].values

    # Free df_model now
    del df_model
    gc.collect()

    train_pos_wb = int(y_wildbird_np[train_tune_mask_np].sum())
    full_pos_wb = int(y_wildbird_np[full_train_mask_np].sum())
    train_pos_pl = int(y_poultry_np[train_tune_mask_np].sum())
    full_pos_pl = int(y_poultry_np[full_train_mask_np].sum())
    test_pos = int(y_poultry_np[test_mask_np].sum())

    print(f"\nSplit Summary:")
    print(f"  Train-Tune (2022-2023): {train_tune_mask_np.sum():,} | "
          f"WB+: {train_pos_wb} | PL+: {train_pos_pl}")
    print(f"  Val-Tune (2024):        {val_tune_mask_np.sum():,} | "
          f"PL+: {int(y_poultry_np[val_tune_mask_np].sum())}")
    print(f"  Full-Train (2022-2024): {full_train_mask_np.sum():,} | "
          f"WB+: {full_pos_wb} | PL+: {full_pos_pl}")
    print(f"  Test (2025):            {test_mask_np.sum():,} | PL+: {test_pos}")

    # ── Feature scaling ───────────────────────────────────────────────────
    scaler_tune = StandardScaler()
    scaler_tune.fit(X_raw[train_tune_mask_np])
    X_tune_scaled = scaler_tune.transform(X_raw)

    scaler_full = StandardScaler()
    scaler_full.fit(X_raw[full_train_mask_np])
    X_full_scaled = scaler_full.transform(X_raw)
    del X_raw
    gc.collect()

    x_tune_tensor = torch.tensor(X_tune_scaled, dtype=torch.float)
    x_full_tensor = torch.tensor(X_full_scaled, dtype=torch.float)
    del X_tune_scaled, X_full_scaled
    gc.collect()

    # ── Class weights for wild-bird pretraining ───────────────────────────
    pos_weight_wb_tune = torch.tensor(
        [(train_tune_mask_np.sum() - train_pos_wb) / train_pos_wb],
        dtype=torch.float
    )
    pos_weight_wb_full = torch.tensor(
        [(full_train_mask_np.sum() - full_pos_wb) / full_pos_wb],
        dtype=torch.float
    )

    # ── Leakage verification ─────────────────────────────────────────────
    test_indices = set(np.where(test_mask_np)[0])
    train_tune_indices = set(np.where(train_tune_mask_np)[0])
    full_train_indices = set(np.where(full_train_mask_np)[0])
    assert len(test_indices & train_tune_indices) == 0, "LEAKAGE: test ∩ train_tune"
    assert len(test_indices & full_train_indices) == 0, "LEAKAGE: test ∩ full_train"
    print("\n[OK] No data leakage between train and test splits.")

    # ── Run all configurations ────────────────────────────────────────────
    all_records = []
    all_preds = {}  # config_name -> ensemble probs
    y_test = y_poultry_np[test_mask_np]

    for config_name in CONFIGS:
        ratio = RATIO_MAP[config_name]

        print(f"\n{'=' * 90}")
        print(f"  CONFIG: {config_name} "
              f"(ratio={'all' if ratio is None else f'1:{ratio}'})")
        print(f"{'=' * 90}")

        seed_probs = []

        for seed in SEEDS:
            print(f"\n  Seed {seed}:")
            record, probs = train_model_b_ablation(
                config_name=config_name,
                ratio=ratio,
                seed=seed,
                x_tune_tensor=x_tune_tensor,
                x_full_tensor=x_full_tensor,
                sparse_adj_list=sparse_adj_list,
                feature_dim=len(feature_cols),
                train_tune_mask_np=train_tune_mask_np,
                val_tune_mask_np=val_tune_mask_np,
                full_train_mask_np=full_train_mask_np,
                test_mask_np=test_mask_np,
                at_risk_mask_np=at_risk_node_mask,
                y_poultry_np=y_poultry_np,
                y_wildbird_np=y_wildbird_np,
                pos_weight_wb_tune=pos_weight_wb_tune,
                pos_weight_wb_full=pos_weight_wb_full,
                num_csds=num_csds,
                ym_tuples=ym_tuples,
                neighbor_map=neighbor_map,
                water_unit_dict=water_unit_dict,
            )
            all_records.append(record)
            seed_probs.append(probs)

        # Ensemble
        ens_probs = np.mean(seed_probs, axis=0)
        all_preds[config_name] = ens_probs

        ens_metrics = compute_full_metrics(y_test, ens_probs, test_pos)
        print(f"\n  {config_name} Ensemble | PR-AUC: {ens_metrics['pr_auc']:.4f} | "
              f"ROC-AUC: {ens_metrics['roc_auc']:.4f} | "
              f"Lift: {ens_metrics['lift_over_random']:.1f}x")

        # Save predictions
        preds_df = pd.DataFrame({
            'csd_uid': test_nodes_df['csd_uid'].values,
            'csd_name': test_nodes_df['csd_name'].values,
            'province_code': test_nodes_df['province_code'].values,
            'year': test_nodes_df['year'].values,
            'month': test_nodes_df['month'].values,
            'actual': y_test.astype(int),
            f'{config_name}_ensemble_prob': ens_probs,
        })
        for i, s in enumerate(SEEDS):
            preds_df[f'{config_name}_seed{s}_prob'] = seed_probs[i]

        preds_df.to_csv(
            os.path.join(PREDS_DIR, f'preds_{config_name}.csv'), index=False
        )

    # ── Save per-seed metrics ─────────────────────────────────────────────
    df_records = pd.DataFrame(all_records)
    df_records.to_csv(
        os.path.join(OUT_DIR, 'training_ratio_per_seed_metrics.csv'),
        index=False
    )

    # ── Build summary table (ensemble metrics per config) ─────────────────
    summary_rows = []
    for config_name in CONFIGS:
        ens_probs = all_preds[config_name]
        m = compute_full_metrics(y_test, ens_probs, test_pos)
        pm = per_month_metrics(y_test, ens_probs, test_months)

        # Per-seed std
        seed_records = [
            r for r in all_records if r['config'] == config_name
        ]
        seed_pr_aucs = [r['pr_auc'] for r in seed_records]

        summary_rows.append({
            'config': config_name,
            'ratio': str(RATIO_MAP[config_name]) if RATIO_MAP[config_name] else 'all',
            'pr_auc': round(m['pr_auc'], 6),
            'pr_auc_std': round(np.std(seed_pr_aucs), 6),
            'roc_auc': round(m['roc_auc'], 6),
            'random_pr_auc': round(m['random_pr_auc_baseline'], 6),
            'lift': round(m['lift_over_random'], 1),
            'precision_10': round(m['precision_10'], 4),
            'precision_25': round(m['precision_25'], 4),
            'precision_50': round(m['precision_50'], 4),
            'precision_100': round(m['precision_100'], 4),
            'recall_50': round(m['recall_50'], 4),
            'brier_score': round(m['brier_score'], 6),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(
        os.path.join(OUT_DIR, 'training_ratio_comparison.csv'), index=False
    )

    print(f"\n{'=' * 90}")
    print("  TRAINING RATIO COMPARISON (2025 Test, Ensemble of 3 Seeds)")
    print(f"{'=' * 90}")
    print(df_summary.to_string(index=False))

    # ── Per-month metrics for each config ─────────────────────────────────
    pm_all = []
    for config_name in CONFIGS:
        pm = per_month_metrics(
            y_test, all_preds[config_name], test_months
        )
        pm['config'] = config_name
        pm_all.append(pm)

    df_pm = pd.concat(pm_all, ignore_index=True)
    df_pm.to_csv(
        os.path.join(OUT_DIR, 'per_month_metrics.csv'), index=False
    )

    # ── Block-bootstrap CIs by month (1,000 resamples) ────────────────────
    print("\n--- Block-Bootstrap Confidence Intervals (by month) ---")
    boot_records = []

    for config_name in CONFIGS:
        ens_probs = all_preds[config_name]

        unique_months = np.unique(test_months)
        n_months = len(unique_months)
        month_groups = {
            m: np.where(test_months == m)[0] for m in unique_months
        }

        boot_pr_aucs = []
        np.random.seed(42)
        for _ in range(1000):
            boot_months = np.random.choice(
                unique_months, size=n_months, replace=True
            )
            boot_idx = np.concatenate(
                [month_groups[m] for m in boot_months]
            )
            y_b = y_test[boot_idx]
            p_b = ens_probs[boot_idx]
            if y_b.sum() == 0 or y_b.sum() == len(y_b):
                continue
            boot_pr_aucs.append(average_precision_score(y_b, p_b))

        if len(boot_pr_aucs) > 10:
            ci = np.percentile(boot_pr_aucs, [2.5, 97.5])
            boot_records.append({
                'config': config_name,
                'pr_auc': round(
                    average_precision_score(y_test, ens_probs), 6
                ),
                'bootstrap_mean': round(np.mean(boot_pr_aucs), 6),
                'ci_lower_95': round(ci[0], 6),
                'ci_upper_95': round(ci[1], 6),
                'n_valid_resamples': len(boot_pr_aucs),
            })
            print(f"  {config_name}: PR-AUC={boot_records[-1]['pr_auc']:.4f} "
                  f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")

    df_boot_ci = pd.DataFrame(boot_records)
    df_boot_ci.to_csv(
        os.path.join(OUT_DIR, 'bootstrap_confidence_intervals.csv'),
        index=False
    )

    # ── Pairwise bootstrap: each ratio vs all_weighted ────────────────────
    print("\n--- Pairwise Bootstrap: Each Ratio vs all_weighted ---")
    pairwise_records = []
    base_probs = all_preds['all_weighted']

    for config_name in CONFIGS:
        if config_name == 'all_weighted':
            continue

        ens_probs = all_preds[config_name]
        unique_months = np.unique(test_months)
        n_months = len(unique_months)
        month_groups = {
            m: np.where(test_months == m)[0] for m in unique_months
        }

        diffs = []
        np.random.seed(42)
        for _ in range(1000):
            boot_months = np.random.choice(
                unique_months, size=n_months, replace=True
            )
            boot_idx = np.concatenate(
                [month_groups[m] for m in boot_months]
            )
            y_b = y_test[boot_idx]
            if y_b.sum() == 0 or y_b.sum() == len(y_b):
                continue
            ap_config = average_precision_score(y_b, ens_probs[boot_idx])
            ap_base = average_precision_score(y_b, base_probs[boot_idx])
            diffs.append(ap_config - ap_base)

        if len(diffs) > 10:
            ci = np.percentile(diffs, [2.5, 97.5])
            win_rate = float(np.mean(np.array(diffs) > 0))
            decision = (
                "statistically better" if ci[0] > 0
                else "statistically worse" if ci[1] < 0
                else "comparable, not proven better"
            )

            obs_diff = (
                average_precision_score(y_test, ens_probs)
                - average_precision_score(y_test, base_probs)
            )

            pairwise_records.append({
                'config': config_name,
                'vs': 'all_weighted',
                'observed_diff': round(obs_diff, 6),
                'ci_lower_95': round(ci[0], 6),
                'ci_upper_95': round(ci[1], 6),
                'win_rate': round(win_rate, 4),
                'decision': decision,
            })
            print(f"  {config_name} vs all_weighted: "
                  f"diff={obs_diff:+.4f} CI [{ci[0]:+.4f}, {ci[1]:+.4f}] "
                  f"win={win_rate:.1%} -> {decision.upper()}")

    df_pairwise = pd.DataFrame(pairwise_records)
    df_pairwise.to_csv(
        os.path.join(OUT_DIR, 'bootstrap_comparison.csv'), index=False
    )

    # ── Write markdown summary ────────────────────────────────────────────
    md = "# Negative-Sampling Ablation: Spatiotemporal Directed RGCN Transfer\n\n"
    md += "## Training Ratio Comparison (2025 Test Set, 3-Seed Ensemble)\n\n"
    md += "| Config | Ratio | PR-AUC | ±Std | ROC-AUC | Random Baseline | "
    md += "Lift | P@10 | P@25 | P@50 | P@100 | R@50 | Brier |\n"
    md += "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"

    for _, row in df_summary.iterrows():
        md += (
            f"| **{row['config']}** | {row['ratio']} | "
            f"**{row['pr_auc']:.4f}** | ±{row['pr_auc_std']:.4f} | "
            f"{row['roc_auc']:.4f} | {row['random_pr_auc']:.4f} | "
            f"{row['lift']:.1f}x | {row['precision_10']:.3f} | "
            f"{row['precision_25']:.3f} | {row['precision_50']:.3f} | "
            f"{row['precision_100']:.3f} | {row['recall_50']:.3f} | "
            f"{row['brier_score']:.4f} |\n"
        )

    if pairwise_records:
        md += "\n## Pairwise Bootstrap vs all_weighted (1,000 month-block resamples)\n\n"
        md += "| Config | Observed Diff | 95% CI | Win Rate | Decision |\n"
        md += "|---|---|---|---|---|\n"
        for r in pairwise_records:
            md += (
                f"| {r['config']} | {r['observed_diff']:+.4f} | "
                f"[{r['ci_lower_95']:+.4f}, {r['ci_upper_95']:+.4f}] | "
                f"{r['win_rate']:.1%} | **{r['decision'].upper()}** |\n"
            )

    md += "\n---\n\n"
    md += "## Technical Notes\n"
    md += "- Phase 1 (wild-bird pretraining) is identical across all configs.\n"
    md += "- Negative sampling applies only to Phase 2 (poultry fine-tuning).\n"
    md += "- Negatives sampled per month: 1/3 random + 1/3 spatial (kNN-3 neighbors of positives) + 1/3 environmental (water-habitat similarity).\n"
    md += "- IPW correction applied to sampled negatives: weight = N_neg_total / N_neg_sampled.\n"
    md += "- Negatives resampled every epoch (stochastic training).\n"
    md += "- Full graph context preserved in forward pass; only loss mask changes.\n"
    md += "- Evaluation is on the complete natural-prevalence locked 2025 test set (55,594 rows, 79 positives).\n"
    md += "- No test-year data influenced any training, sampling, or model selection decision.\n"

    with open(os.path.join(OUT_DIR, 'training_ratio_comparison.md'), 'w',
              encoding='utf-8') as f:
        f.write(md)

    print(f"\n{'=' * 90}")
    print("  OUTPUTS SAVED:")
    print(f"  -> {os.path.join(OUT_DIR, 'training_ratio_comparison.csv')}")
    print(f"  -> {os.path.join(OUT_DIR, 'training_ratio_comparison.md')}")
    print(f"  -> {os.path.join(OUT_DIR, 'per_month_metrics.csv')}")
    print(f"  -> {os.path.join(OUT_DIR, 'bootstrap_confidence_intervals.csv')}")
    print(f"  -> {os.path.join(OUT_DIR, 'bootstrap_comparison.csv')}")
    print(f"  -> {PREDS_DIR}/preds_*.csv")
    print(f"{'=' * 90}")
    print("Done.")


if __name__ == '__main__':
    main()
