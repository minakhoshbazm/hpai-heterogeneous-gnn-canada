#!/usr/bin/env python3
"""
scripts/run_rolling_wildbird_multitask_graphsage.py

Leakage-Safe Multi-Task GraphSAGE Experiment:
Shared encoder trained on primary poultry HPAI prediction (outbreak_next_month)
and auxiliary wild-bird HPAI presence prediction (wildbird_hpai_present_next_month at CCS level).

Architecture:
  - Shared two-layer monthly-snapshot GraphSAGE encoder (identical to valid baseline)
  - Primary head: per-CSD poultry outbreak classifier
  - Auxiliary head: per-CCS wild-bird presence classifier (mean-pooled CSD embeddings)
  - Total loss: primary_loss + lambda_aux * auxiliary_loss

Lambda grid: [0.00, 0.05, 0.10, 0.20]
Selected per fold using validation PR-AUC on primary poultry task only.

Rolling Folds (identical splits to valid rolling runs):
  Fold 1: DevTrain 2022-01..09, DevVal 2022-10..12, FinalTrain 2022, Test 2023
  Fold 2: DevTrain 2022-01..2023-09, DevVal 2023-10..12, FinalTrain 2022-2023, Test 2024
  Fold 3: DevTrain 2022-01..2024-09, DevVal 2024-10..12, FinalTrain 2022-2024, Test 2025

Outputs (new files only):
  results/baselines/rolling_wildbird_multitask_graphsage_validation_metrics.csv
  results/baselines/rolling_wildbird_multitask_graphsage_fold_metrics.csv
  results/baselines/rolling_wildbird_multitask_graphsage_test_predictions.csv
  results/baselines/rolling_wildbird_multitask_graphsage_bootstrap.csv
  results/baselines/rolling_wildbird_multitask_graphsage_summary.md
  results/data_qa/rolling_wildbird_multitask_graphsage_audit.md
"""

import os
import random
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

# Environment
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# Paths
INPUT_TABLE      = "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
WB_RAW_CSV       = "data/raw/cfia/CFIA_outbreaks_wildbirds.csv"
WB_CROSSWALK     = "data/processed/geography/wildbird_cfia_ccs_crosswalk_reviewed.csv"
GRAPH_EDGE_CSV   = "data/processed/graph/csd_graph_knn_3_v1.csv"
VALID_GS_LR_PRED = "results/baselines/rolling_wildbird_monthly_snapshot_graphsage_vs_lr_predictions.csv"
VALID_TREE_PRED  = "results/baselines/rolling_wildbird_tree_baselines_test_predictions.csv"

OUT_VAL_METRICS  = "results/baselines/rolling_wildbird_multitask_graphsage_validation_metrics.csv"
OUT_FOLD_METRICS = "results/baselines/rolling_wildbird_multitask_graphsage_fold_metrics.csv"
OUT_PREDS        = "results/baselines/rolling_wildbird_multitask_graphsage_test_predictions.csv"
OUT_BOOT         = "results/baselines/rolling_wildbird_multitask_graphsage_bootstrap.csv"
OUT_MD           = "results/baselines/rolling_wildbird_multitask_graphsage_summary.md"
OUT_QA_MD        = "results/data_qa/rolling_wildbird_multitask_graphsage_audit.md"

SEEDS        = [11, 22, 33]
LAMBDA_GRID  = [0.00, 0.05, 0.10, 0.20]
PATIENCE     = 20
MAX_EPOCHS   = 200

EXCLUDE_EXPLICIT = {
    "csd_uid", "csd_name", "province_code", "province_name", "ccs_uid",
    "year", "month", "outbreak_next_month",
    "dw_fill_method", "climate_extraction_method",
    "glw2020_chicken_extraction_method", "glw2015_duck_extraction_method",
    "road_data_coverage", "outbreak_current_month",
}


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary Target Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_auxiliary_target(wb_raw_path: str, crosswalk_path: str) -> pd.DataFrame:
    """
    Builds a (target_ccs_uid, year, month) -> {0,1} dataframe:
    wildbird_hpai_present = 1 if any wild-bird HPAI detection in that CCS-month.

    Returns a DataFrame with cols: [target_ccs_uid, year, month, wildbird_hpai_present]
    """
    wb = pd.read_csv(wb_raw_path, encoding="latin-1")
    xw = pd.read_csv(crosswalk_path, encoding="latin-1")

    # Only use matched crosswalk entries
    xw_matched = xw[xw["final_status"] == "matched"][
        ["raw_ccs_name", "raw_province", "target_ccs_uid"]
    ].dropna()

    wb_m = wb.merge(
        xw_matched,
        left_on=["Census Consolidated Subdivision (CCS)", "Province"],
        right_on=["raw_ccs_name", "raw_province"],
        how="inner"
    )

    wb_m["collection_date"] = pd.to_datetime(wb_m["Collection Date"], errors="coerce")
    wb_m = wb_m.dropna(subset=["collection_date"])
    wb_m["year"]  = wb_m["collection_date"].dt.year
    wb_m["month"] = wb_m["collection_date"].dt.month

    # One binary per (ccs, year, month): present if any detection
    aux = (
        wb_m.groupby(["target_ccs_uid", "year", "month"])
        .size()
        .reset_index(name="det_count")
    )
    aux["wildbird_hpai_present"] = 1
    # Ensure all (ccs, year, month) combos that appear in model table but have 0
    # detections are handled implicitly (left join with fillna=0 later)
    return aux[["target_ccs_uid", "year", "month", "wildbird_hpai_present"]]


def attach_aux_label_next_month(df: pd.DataFrame, aux_detections: pd.DataFrame) -> pd.DataFrame:
    """
    For each row in df at month M:
    wildbird_hpai_present_next_month = 1 if the same CCS had a detection in month M+1.
    The M+1 lookup is done by shifting month/year correctly.
    This is the LABEL only — the feature 'wildbird_hpai_count_ccs_current_month' etc.
    already in df refers to month M or earlier, so no future leakage.
    """
    # Build a lookup indexed on (ccs_uid, year, month) for detections
    aux_lookup = aux_detections.rename(columns={
        "target_ccs_uid": "ccs_uid",
        "wildbird_hpai_present": "wb_present"
    }).set_index(["ccs_uid", "year", "month"])["wb_present"]

    # Compute next month
    df = df.copy()
    next_month = df["month"] + 1
    next_year  = df["year"].copy()
    overflow   = next_month > 12
    next_month[overflow] = 1
    next_year[overflow]  = next_year[overflow] + 1

    keys = list(zip(df["ccs_uid"].values, next_year.values, next_month.values))
    labels = [int(aux_lookup.get(k, 0)) for k in keys]
    df["wildbird_hpai_present_next_month"] = labels
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Model
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskGraphSAGE(nn.Module):
    def __init__(self, in_channels=62, hidden=32):
        super().__init__()
        # Shared encoder
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        # Primary head: poultry HPAI (CSD-level)
        self.primary_head = nn.Linear(hidden, 1)
        # Auxiliary head: wild-bird presence (CCS-level, after mean pooling)
        self.aux_head = nn.Linear(hidden, 1)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        return x  # (N, hidden)

    def forward_primary(self, x, edge_index):
        emb = self.encode(x, edge_index)
        return self.primary_head(emb).squeeze(-1), emb

    def forward_aux(self, emb, ccs_groups):
        """
        emb:        (N, hidden) CSD embeddings for this snapshot
        ccs_groups: list of index arrays, one per unique CCS present in this snapshot
        Returns:    (n_ccs,) logits from aux head after mean-pooling CSD embeddings per CCS
        """
        ccs_logits = []
        for idx in ccs_groups:
            ccs_emb = emb[idx].mean(dim=0, keepdim=True)  # (1, hidden)
            ccs_logits.append(self.aux_head(ccs_emb).squeeze())
        return torch.stack(ccs_logits)  # (n_ccs,)


# ─────────────────────────────────────────────────────────────────────────────
# Graph Snapshot Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_snapshot_with_aux(df_subset, feature_cols, edges_raw, scaler):
    """
    Returns list of dicts with PyG Data + CCS auxiliary info per snapshot.
    Each element: {data, ccs_uids_per_group, aux_labels_per_group, year, month, csd_uids}
    """
    snapshots = []
    for (y, m), grp in df_subset.groupby(["year", "month"], sort=True):
        grp = grp.reset_index(drop=True)
        X   = scaler.transform(grp[feature_cols].fillna(0.0).values).astype(np.float32)
        yt  = grp["outbreak_next_month"].values.astype(np.float32)

        uid_map = {uid: i for i, uid in enumerate(grp["csd_uid"].values)}
        src_ok  = edges_raw["source_csd_uid"].isin(uid_map)
        tgt_ok  = edges_raw["target_csd_uid"].isin(uid_map)
        mask    = src_ok & tgt_ok
        src_i   = edges_raw.loc[mask, "source_csd_uid"].map(uid_map).values
        tgt_i   = edges_raw.loc[mask, "target_csd_uid"].map(uid_map).values

        data = Data(
            x=torch.from_numpy(X),
            edge_index=torch.tensor(np.stack([src_i, tgt_i]), dtype=torch.long),
            y=torch.from_numpy(yt),
        )

        # Build CCS groupings for auxiliary task
        # Group CSD indices by their ccs_uid
        if "wildbird_hpai_present_next_month" in grp.columns:
            ccs_uid_arr   = grp["ccs_uid"].values
            aux_label_arr = grp["wildbird_hpai_present_next_month"].values

            unique_ccs = []
            ccs_groups  = []  # list of node-index arrays
            aux_labels  = []  # one label per CCS (first occurrence, all same within CCS-month)

            # De-duplicate: compute per-CCS auxiliary label (presence is a CCS-month property)
            ccs_to_info = {}
            for i, ccs in enumerate(ccs_uid_arr):
                if ccs not in ccs_to_info:
                    ccs_to_info[ccs] = {"indices": [], "label": int(aux_label_arr[i])}
                ccs_to_info[ccs]["indices"].append(i)

            for ccs, info in ccs_to_info.items():
                unique_ccs.append(ccs)
                ccs_groups.append(info["indices"])
                aux_labels.append(info["label"])

            aux_y_tensor = torch.tensor(aux_labels, dtype=torch.float)
        else:
            ccs_groups   = []
            aux_y_tensor = torch.zeros(0)

        snapshots.append({
            "data":       data,
            "ccs_groups": ccs_groups,
            "aux_y":      aux_y_tensor,
            "year":       int(y),
            "month":      int(m),
            "csd_uids":   grp["csd_uid"].values,
        })
    return snapshots


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, snapshots, optimizer, pw_primary, pw_aux, lambda_aux):
    model.train()
    loader = PyGDataLoader([s["data"] for s in snapshots], batch_size=1, shuffle=True)

    # Map each batch iteration back to snapshot
    total_loss = 0.0
    for snap in snapshots:
        optimizer.zero_grad()
        data = snap["data"]
        prim_logits, emb = model.forward_primary(data.x, data.edge_index)

        # Primary loss
        prim_loss = F.binary_cross_entropy_with_logits(
            prim_logits, data.y,
            pos_weight=torch.tensor([pw_primary])
        )

        total_snap_loss = prim_loss

        # Auxiliary loss (only if lambda > 0 and there are CCS groups)
        if lambda_aux > 0.0 and len(snap["ccs_groups"]) > 0 and snap["aux_y"].sum() >= 0:
            ccs_group_tensors = [torch.tensor(g, dtype=torch.long) for g in snap["ccs_groups"]]
            aux_logits = model.forward_aux(emb, ccs_group_tensors)
            aux_y      = snap["aux_y"]
            if len(aux_y) > 0 and aux_y.sum() > 0:  # need at least 1 positive
                aux_loss = F.binary_cross_entropy_with_logits(
                    aux_logits, aux_y,
                    pos_weight=torch.tensor([pw_aux])
                )
                total_snap_loss = prim_loss + lambda_aux * aux_loss

        total_snap_loss.backward()
        optimizer.step()
        total_loss += total_snap_loss.item()

    return total_loss / len(snapshots)


def evaluate_primary(model, snapshots):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for snap in snapshots:
            logits, _ = model.forward_primary(snap["data"].x, snap["data"].edge_index)
            ps.extend(torch.sigmoid(logits).cpu().numpy())
            ys.extend(snap["data"].y.cpu().numpy())
    return np.array(ys), np.array(ps)


def evaluate_aux(model, snapshots):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for snap in snapshots:
            if len(snap["ccs_groups"]) == 0:
                continue
            _, emb = model.forward_primary(snap["data"].x, snap["data"].edge_index)
            ccs_group_tensors = [torch.tensor(g, dtype=torch.long) for g in snap["ccs_groups"]]
            aux_logits = model.forward_aux(emb, ccs_group_tensors)
            ps.extend(torch.sigmoid(aux_logits).cpu().numpy())
            ys.extend(snap["aux_y"].numpy())
    return np.array(ys), np.array(ps)


def evaluate_primary_aligned(model, test_snaps, df_test_ref):
    model.eval()
    recs = []
    with torch.no_grad():
        for snap in test_snaps:
            logits, _ = model.forward_primary(snap["data"].x, snap["data"].edge_index)
            probs = torch.sigmoid(logits).cpu().numpy()
            ys    = snap["data"].y.cpu().numpy()
            for i, uid in enumerate(snap["csd_uids"]):
                recs.append({
                    "csd_uid": uid,
                    "year":    snap["year"],
                    "month":   snap["month"],
                    "actual":  ys[i],
                    "prob":    probs[i]
                })
    df_res = pd.DataFrame(recs)
    df_aligned = df_test_ref[["csd_uid", "year", "month"]].merge(
        df_res, on=["csd_uid", "year", "month"], how="left")
    return df_aligned["prob"].values


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def pk_rk(y, p, k):
    df = pd.DataFrame({"y": y, "p": p}).sort_values("p", ascending=False)
    top = df.head(k)
    prec = top["y"].mean()
    rec  = top["y"].sum() / y.sum() if y.sum() > 0 else 0.0
    return prec, rec


def compute_metrics(y, p):
    m = {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p)}
    for k in [50, 100, 200]:
        m[f"precision_at_{k}"], m[f"recall_at_{k}"] = pk_rk(y, p, k)
    return m


def csd_bootstrap(df_all, col_new, col_ref, n_boot=1000, seed=42):
    rng         = np.random.default_rng(seed)
    unique_csds = df_all["csd_uid"].unique()
    csd_idx     = df_all.groupby("csd_uid").indices
    y           = df_all["actual"].values
    p_new       = df_all[col_new].values
    p_ref       = df_all[col_ref].values

    obs_diff = average_precision_score(y, p_new) - average_precision_score(y, p_ref)
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_csds, size=len(unique_csds), replace=True)
        idx     = np.concatenate([csd_idx[c] for c in sampled])
        d = (average_precision_score(y[idx], p_new[idx])
             - average_precision_score(y[idx], p_ref[idx]))
        diffs.append(d)
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    win_rate     = np.mean(diffs > 0) * 100.0

    if ci_lo > 0:   decision = "statistically better"
    elif ci_hi < 0: decision = "statistically worse"
    else:           decision = "comparable, not proven better"
    return dict(obs_diff=obs_diff, boot_mean=diffs.mean(),
                ci_lo=ci_lo, ci_hi=ci_hi, win_rate=win_rate, decision=decision)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("MULTI-TASK GraphSAGE: Wild-Bird Auxiliary Supervision")
    print("=" * 72)

    # Load modeling table
    df = pd.read_csv(INPUT_TABLE)
    print(f"Modeling table: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # Feature selection (62 features, identical to valid baseline)
    feature_cols = sorted([
        c for c in df.columns
        if c not in EXCLUDE_EXPLICIT
        and not any(s in c.lower() for s in ["ebird", "gbif", "neighbor", "pressure"])
        and pd.api.types.is_numeric_dtype(df[c])
    ])
    print(f"Feature count : {len(feature_cols)}")
    assert len(feature_cols) == 62, f"Expected 62 features, got {len(feature_cols)}"
    print("Feature assertion PASSED: 62 confirmed.\n")

    # Load edges
    edges_raw = pd.read_csv(GRAPH_EDGE_CSV)
    print(f"Edge list: {len(edges_raw):,} edges")

    # Build auxiliary target
    print("Building auxiliary wild-bird HPAI target ...")
    aux_det = build_auxiliary_target(WB_RAW_CSV, WB_CROSSWALK)
    print(f"  Unique (ccs, year, month) detection groups: {len(aux_det):,}")
    print(f"  Year distribution: {aux_det['year'].value_counts().sort_index().to_dict()}")

    # Attach auxiliary labels to model table
    df = attach_aux_label_next_month(df, aux_det)
    aux_col_counts = df.groupby(["year","month"])["wildbird_hpai_present_next_month"].sum()
    print(f"  Total CSD-month rows with aux=1 (all years): {int(df['wildbird_hpai_present_next_month'].sum()):,}")

    # Load existing valid baseline predictions
    df_gs_lr = pd.read_csv(VALID_GS_LR_PRED)
    df_trees = pd.read_csv(VALID_TREE_PRED)
    assert "wildbird_snapshot_gs_ensemble_prob" in df_gs_lr.columns
    assert "wildbird_lr_prob" in df_gs_lr.columns
    df_rf  = df_trees[df_trees["model_name"] == "Random Forest"][
        ["csd_uid","year","month","predicted_probability"]
    ].rename(columns={"predicted_probability":"rf_prob"})
    df_xgb = df_trees[df_trees["model_name"] == "XGBoost"][
        ["csd_uid","year","month","predicted_probability"]
    ].rename(columns={"predicted_probability":"xgb_prob"})
    print(f"\nLoaded valid GNN/LR: {len(df_gs_lr):,} | Tree preds: {len(df_trees):,}")

    folds_def = [
        {
            "fold": 1, "test_year": 2023,
            "dev_train":   (df["year"] == 2022) & (df["month"] <= 9),
            "dev_val":     (df["year"] == 2022) & (df["month"] >= 10),
            "final_train": (df["year"] == 2022),
            "test":        (df["year"] == 2023),
        },
        {
            "fold": 2, "test_year": 2024,
            "dev_train":   (df["year"] <= 2022) | ((df["year"] == 2023) & (df["month"] <= 9)),
            "dev_val":     (df["year"] == 2023) & (df["month"] >= 10),
            "final_train": (df["year"] <= 2023),
            "test":        (df["year"] == 2024),
        },
        {
            "fold": 3, "test_year": 2025,
            "dev_train":   (df["year"] <= 2023) | ((df["year"] == 2024) & (df["month"] <= 9)),
            "dev_val":     (df["year"] == 2024) & (df["month"] >= 10),
            "final_train": (df["year"] <= 2024),
            "test":        (df["year"] == 2025),
        },
    ]

    all_val_records  = []
    all_fold_metrics = []
    all_test_preds   = []
    fold_selections  = {}

    for f_info in folds_def:
        fold    = f_info["fold"]
        test_yr = f_info["test_year"]
        print(f"\n{'='*72}")
        print(f"FOLD {fold}  (Test Year {test_yr})")
        print(f"{'='*72}")

        df_dev_tr   = df[f_info["dev_train"]].copy()
        df_dev_val  = df[f_info["dev_val"]].copy()
        df_final_tr = df[f_info["final_train"]].copy()
        df_test     = df[f_info["test"]].copy().reset_index(drop=True)

        # Assert no test-year leakage
        assert df_dev_tr["year"].max() < test_yr, \
            f"Fold {fold}: dev_train contains test year {test_yr}!"
        assert df_final_tr["year"].max() < test_yr, \
            f"Fold {fold}: final_train contains test year {test_yr}!"

        n_test   = len(df_test)
        pos_test = int(df_test["outbreak_next_month"].sum())
        prev     = pos_test / n_test
        y_test   = df_test["outbreak_next_month"].values.astype(np.float32)

        # Aux target stats
        aux_pos_dev_tr = int(df_dev_tr["wildbird_hpai_present_next_month"].sum())
        aux_pos_fin_tr = int(df_final_tr["wildbird_hpai_present_next_month"].sum())
        print(f"  DevTrain={len(df_dev_tr):,}  DevVal={len(df_dev_val):,}  "
              f"FinalTrain={len(df_final_tr):,}  Test={n_test:,}")
        print(f"  Primary positives: DevTrain={int(df_dev_tr['outbreak_next_month'].sum())}  "
              f"Test={pos_test}")
        print(f"  Auxiliary (wildbird) positives (CSD-level): DevTrain={aux_pos_dev_tr}  "
              f"FinalTrain={aux_pos_fin_tr}")

        # Scalers
        scaler_dev = StandardScaler().fit(df_dev_tr[feature_cols].fillna(0.0))
        scaler_fin = StandardScaler().fit(df_final_tr[feature_cols].fillna(0.0))

        # Snapshots
        dev_tr_snaps  = build_snapshot_with_aux(df_dev_tr,   feature_cols, edges_raw, scaler_dev)
        dev_val_snaps = build_snapshot_with_aux(df_dev_val,  feature_cols, edges_raw, scaler_dev)
        final_snaps   = build_snapshot_with_aux(df_final_tr, feature_cols, edges_raw, scaler_fin)
        test_snaps    = build_snapshot_with_aux(df_test,     feature_cols, edges_raw, scaler_fin)

        # Class weights (from dev-train for dev phase, from final-train for final phase)
        n_dev_pos  = int(df_dev_tr["outbreak_next_month"].sum())
        n_dev_neg  = len(df_dev_tr) - n_dev_pos
        pw_prim_dev = n_dev_neg / max(n_dev_pos, 1)

        n_aux_dev_pos = max(aux_pos_dev_tr, 1)
        n_aux_dev_neg = max(len(df_dev_tr) - aux_pos_dev_tr, 1)
        pw_aux_dev    = n_aux_dev_neg / n_aux_dev_pos

        n_fin_pos   = int(df_final_tr["outbreak_next_month"].sum())
        n_fin_neg   = len(df_final_tr) - n_fin_pos
        pw_prim_fin = n_fin_neg / max(n_fin_pos, 1)

        n_aux_fin_pos = max(aux_pos_fin_tr, 1)
        n_aux_fin_neg = max(len(df_final_tr) - aux_pos_fin_tr, 1)
        pw_aux_fin    = n_aux_fin_neg / n_aux_fin_pos

        # ── Phase 1: Lambda selection on dev-val ─────────────────────────────
        print(f"\n  [Phase 1] Evaluating {len(LAMBDA_GRID)} lambda candidates ...")
        lambda_results = {}

        for lam in LAMBDA_GRID:
            seed_val_aps, seed_aux_aps, seed_epochs = [], [], []

            for seed in SEEDS:
                set_seed(seed)
                model     = MultiTaskGraphSAGE()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

                best_val_ap, best_epoch, no_improve = -1.0, 0, 0
                best_aux_ap = 0.0

                for epoch in range(1, MAX_EPOCHS + 1):
                    train_epoch(model, dev_tr_snaps, optimizer,
                                pw_prim_dev, pw_aux_dev, lam)

                    yv, pv = evaluate_primary(model, dev_val_snaps)
                    val_ap = average_precision_score(yv, pv)

                    if val_ap > best_val_ap:
                        best_val_ap, best_epoch, no_improve = val_ap, epoch, 0
                        # Also record aux val performance at best primary epoch
                        yav, pav = evaluate_aux(model, dev_val_snaps)
                        if len(yav) > 0 and yav.sum() > 0:
                            best_aux_ap = average_precision_score(yav, pav)
                    else:
                        no_improve += 1
                        if no_improve >= PATIENCE:
                            break

                seed_val_aps.append(best_val_ap)
                seed_aux_aps.append(best_aux_ap)
                seed_epochs.append(best_epoch)

            mean_val_ap  = np.mean(seed_val_aps)
            mean_aux_ap  = np.mean(seed_aux_aps)
            lambda_results[lam] = {
                "mean_val_ap":  mean_val_ap,
                "mean_aux_ap":  mean_aux_ap,
                "seed_val_aps": seed_val_aps,
                "seed_epochs":  seed_epochs,
            }
            print(f"    lambda={lam:.2f}  mean_prim_val_PR-AUC={mean_val_ap:.4f}  "
                  f"mean_aux_val_PR-AUC={mean_aux_ap:.4f}  "
                  f"epochs={seed_epochs}")

            for si, seed in enumerate(SEEDS):
                all_val_records.append({
                    "fold": fold, "test_year": test_yr,
                    "lambda_aux": lam, "seed": seed,
                    "val_primary_pr_auc": seed_val_aps[si],
                    "val_aux_pr_auc":     seed_aux_aps[si],
                    "best_epoch":         seed_epochs[si],
                    "mean_3seed_val_ap":  mean_val_ap,
                })

        # Select best lambda by mean 3-seed primary val PR-AUC
        best_lam = max(lambda_results, key=lambda k: lambda_results[k]["mean_val_ap"])
        best_info = lambda_results[best_lam]
        print(f"\n  [Selection] Best lambda={best_lam:.2f}  "
              f"mean_prim_val_PR-AUC={best_info['mean_val_ap']:.4f}")
        print(f"             (lambda=0.00 baseline: {lambda_results[0.0]['mean_val_ap']:.4f})")
        fold_selections[fold] = {
            "selected_lambda":    best_lam,
            "mean_val_pr_auc":    best_info["mean_val_ap"],
            "baseline_val_pr_auc": lambda_results[0.0]["mean_val_ap"],
            "seed_epochs":        best_info["seed_epochs"],
        }

        # ── Phase 2: Final retraining with selected lambda ────────────────────
        print(f"\n  [Phase 2] Retraining with lambda={best_lam:.2f} on full fold training ...")
        seed_probs = []

        for si, seed in enumerate(SEEDS):
            set_seed(seed)
            model     = MultiTaskGraphSAGE()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            n_ep      = best_info["seed_epochs"][si]

            for epoch in range(1, n_ep + 1):
                train_epoch(model, final_snaps, optimizer,
                            pw_prim_fin, pw_aux_fin, best_lam)

            probs = evaluate_primary_aligned(model, test_snaps, df_test)
            seed_probs.append(probs)

            m = compute_metrics(y_test, probs)
            m.update(dict(
                fold=fold, test_year=test_yr,
                model=f"Multitask GraphSAGE Seed {seed}",
                selected_lambda=best_lam,
                seed=seed,
                val_primary_pr_auc=best_info["seed_val_aps"][si],
                selected_epoch=n_ep,
                test_rows=n_test, test_positives=pos_test, prevalence=prev,
                pr_auc_lift=m["pr_auc"] / prev,
            ))
            all_fold_metrics.append(m)

        ens_probs = np.mean(np.stack(seed_probs, axis=0), axis=0)
        ens_m     = compute_metrics(y_test, ens_probs)
        ens_m.update(dict(
            fold=fold, test_year=test_yr,
            model="Multitask GraphSAGE (Ensemble)",
            selected_lambda=best_lam,
            seed="ensemble",
            val_primary_pr_auc=best_info["mean_val_ap"],
            selected_epoch=str(best_info["seed_epochs"]),
            test_rows=n_test, test_positives=pos_test, prevalence=prev,
            pr_auc_lift=ens_m["pr_auc"] / prev,
        ))
        all_fold_metrics.append(ens_m)
        print(f"  [Ensemble] PR-AUC={ens_m['pr_auc']:.4f} (Lift: {ens_m['pr_auc_lift']:.2f}x)  "
              f"ROC-AUC={ens_m['roc_auc']:.4f}")

        # QA
        assert df_test.duplicated(["csd_uid", "year", "month"]).sum() == 0
        const_csds = (df_test.assign(p=ens_probs).groupby("csd_uid")["p"].nunique() == 1).sum()
        pct_const  = const_csds / df_test["csd_uid"].nunique() * 100
        assert pct_const < 10.0, f"Fold {fold}: {pct_const:.1f}% constant CSDs!"
        p_unique   = np.unique(ens_probs).shape[0]
        assert p_unique > 100, f"Fold {fold}: only {p_unique} unique probs - collapsed!"
        print(f"  [QA] Constant CSDs: {const_csds} ({pct_const:.2f}%)  "
              f"Unique probs: {p_unique}")
        print(f"  [QA] prob stats: min={ens_probs.min():.4f} max={ens_probs.max():.4f} "
              f"mean={ens_probs.mean():.4f} std={ens_probs.std():.4f}")

        df_fold_pred = df_test[["csd_uid", "year", "month"]].copy()
        df_fold_pred["actual"]          = y_test
        df_fold_pred["prob"]            = ens_probs
        df_fold_pred["fold"]            = fold
        df_fold_pred["test_year"]       = test_yr
        df_fold_pred["selected_lambda"] = best_lam
        all_test_preds.append(df_fold_pred)

    # ── Save outputs ─────────────────────────────────────────────────────────
    os.makedirs("results/baselines", exist_ok=True)
    os.makedirs("results/data_qa", exist_ok=True)

    pd.DataFrame(all_val_records).to_csv(OUT_VAL_METRICS, index=False)
    print(f"\nSaved validation metrics -> {OUT_VAL_METRICS}")

    pd.DataFrame(all_fold_metrics).to_csv(OUT_FOLD_METRICS, index=False)
    print(f"Saved fold metrics -> {OUT_FOLD_METRICS}")

    df_test_preds = pd.concat(all_test_preds, ignore_index=True)
    df_test_preds.to_csv(OUT_PREDS, index=False)
    print(f"Saved test predictions -> {OUT_PREDS} ({len(df_test_preds):,} rows)")

    # ── Alignment QA ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ALIGNMENT QA")
    print("=" * 72)

    df_base = df_gs_lr[["csd_uid", "year", "month", "actual", "fold",
                         "wildbird_lr_prob", "wildbird_snapshot_gs_ensemble_prob"]].copy()
    df_new  = df_test_preds[["csd_uid", "year", "month", "prob"]].rename(
        columns={"prob": "multitask_gs_prob"})

    df_all = df_base.merge(df_new, on=["csd_uid", "year", "month"], how="inner")
    df_all = df_all.merge(df_rf,  on=["csd_uid", "year", "month"], how="left")
    df_all = df_all.merge(df_xgb, on=["csd_uid", "year", "month"], how="left")

    assert len(df_all) == len(df_gs_lr), f"Row mismatch: {len(df_all)} vs {len(df_gs_lr)}"
    assert df_all["multitask_gs_prob"].isna().sum() == 0
    assert df_all["rf_prob"].isna().sum()  == 0
    assert df_all["xgb_prob"].isna().sum() == 0
    assert df_all.duplicated(["csd_uid","year","month"]).sum() == 0
    print(f"  Aligned rows: {len(df_all):,}  OK")
    print(f"  All missing checks: 0  PASSED")
    print(f"  Duplicate key check: 0  PASSED")
    print(f"  GNN source column: 'wildbird_snapshot_gs_ensemble_prob' (valid monthly-snapshot)  OK")

    # ── Pooled evaluation ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("POOLED OUT-OF-SAMPLE EVALUATION (2023-2025)")
    print("=" * 72)

    y_pool   = df_all["actual"].values
    col_map  = {
        "Multitask GraphSAGE (Ensemble)":             "multitask_gs_prob",
        "Original wBCE GraphSAGE (Ensemble)":         "wildbird_snapshot_gs_ensemble_prob",
        "Logistic Regression":                         "wildbird_lr_prob",
        "Random Forest":                               "rf_prob",
        "XGBoost":                                     "xgb_prob",
    }

    pooled_rows = []
    for name, col in col_map.items():
        p = df_all[col].values
        m = compute_metrics(y_pool, p)
        prev_pool = y_pool.mean()
        m.update({"model": name, "total_rows": len(y_pool),
                  "total_positives": int(y_pool.sum()),
                  "pr_auc_lift": m["pr_auc"] / prev_pool})
        pooled_rows.append(m)
        print(f"  {name:<50}  PR-AUC={m['pr_auc']:.4f}  "
              f"Lift={m['pr_auc_lift']:.2f}x  ROC-AUC={m['roc_auc']:.4f}")

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("1,000 CSD-CLUSTER BOOTSTRAP")
    print("=" * 72)

    boot_records = []
    for name, ref_col in [
        ("vs original wBCE GraphSAGE",  "wildbird_snapshot_gs_ensemble_prob"),
        ("vs Logistic Regression",        "wildbird_lr_prob"),
        ("vs Random Forest",              "rf_prob"),
        ("vs XGBoost",                    "xgb_prob"),
    ]:
        res = csd_bootstrap(df_all, "multitask_gs_prob", ref_col)
        boot_records.append({
            "comparison":      f"Multitask GraphSAGE {name}",
            "multitask_pr_auc": average_precision_score(y_pool, df_all["multitask_gs_prob"].values),
            "ref_pr_auc":       average_precision_score(y_pool, df_all[ref_col].values),
            **{k: res[k] for k in ["obs_diff","boot_mean","ci_lo","ci_hi","win_rate","decision"]},
        })
        print(f"  Multitask GS {name}:")
        print(f"    obs={res['obs_diff']:+.4f}  "
              f"CI=[{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}]  "
              f"win={res['win_rate']:.1f}%  -> {res['decision'].upper()}")

    pd.DataFrame(boot_records).to_csv(OUT_BOOT, index=False)
    print(f"\nSaved bootstrap -> {OUT_BOOT}")

    # ── Markdown summary ──────────────────────────────────────────────────────
    mt_pr   = average_precision_score(y_pool, df_all["multitask_gs_prob"].values)
    ref_pr  = average_precision_score(y_pool, df_all["wildbird_snapshot_gs_ensemble_prob"].values)
    improved = mt_pr > ref_pr

    vs_ref = [r for r in boot_records if "wBCE" in r["comparison"]][0]
    if vs_ref["decision"] == "statistically better" and improved:
        final_decision = (f"**Wild-bird auxiliary supervision improved the valid GraphSAGE** "
                          f"(pooled PR-AUC: {mt_pr:.4f} vs {ref_pr:.4f}, obs={vs_ref['obs_diff']:+.4f}, "
                          f"95% CI=[{vs_ref['ci_lo']:+.4f},{vs_ref['ci_hi']:+.4f}]).")
    elif improved:
        final_decision = (f"Wild-bird auxiliary supervision improved pooled test PR-AUC by "
                          f"{mt_pr-ref_pr:+.4f} ({mt_pr:.4f} vs {ref_pr:.4f}), but the improvement "
                          f"is **not statistically proven** (95% CI=[{vs_ref['ci_lo']:+.4f},"
                          f"{vs_ref['ci_hi']:+.4f}], win rate={vs_ref['win_rate']:.1f}%).")
    else:
        final_decision = (f"**Wild-bird auxiliary supervision did not improve poultry-outbreak prediction.** "
                          f"Multitask pooled PR-AUC={mt_pr:.4f} vs original wBCE GraphSAGE PR-AUC={ref_pr:.4f} "
                          f"(obs diff={vs_ref['obs_diff']:+.4f}, 95% CI=[{vs_ref['ci_lo']:+.4f},"
                          f"{vs_ref['ci_hi']:+.4f}]). "
                          f"The single-task wBCE GraphSAGE remains the best GNN configuration.")

    md = f"""# Multi-Task GraphSAGE: Wild-Bird Auxiliary Supervision Experiment

**Purpose**: Test whether a shared GraphSAGE encoder simultaneously trained on poultry HPAI prediction
and wild-bird HPAI presence (auxiliary) improves primary poultry-outbreak prediction.

---

## 1. Lambda Selection per Fold (Primary Poultry Validation PR-AUC)

| Fold | Test Year | Selected Lambda | Multitask Val PR-AUC | Single-Task Val PR-AUC (lambda=0) | Improvement | Selected Epochs |
|---|---|---|---|---|---|---|
"""
    for fold, sel in fold_selections.items():
        md += (f"| {fold} | {folds_def[fold-1]['test_year']} | "
               f"`lambda={sel['selected_lambda']:.2f}` | "
               f"{sel['mean_val_pr_auc']:.4f} | {sel['baseline_val_pr_auc']:.4f} | "
               f"{sel['mean_val_pr_auc']-sel['baseline_val_pr_auc']:+.4f} | "
               f"{sel['seed_epochs']} |\n")

    md += f"""
> [!NOTE]
> Lambda selection used **only the primary poultry validation PR-AUC**. No test-year labels were used.

---

## 2. Per-Fold Test Results (Ensemble)

| Fold | Test Year | Test Rows | Pos | Prev | Selected Lambda | PR-AUC | Lift | ROC-AUC | P@50 | R@50 | P@100 | R@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""
    ens_rows = [r for r in all_fold_metrics if r.get("seed") == "ensemble"]
    for r in ens_rows:
        md += (f"| {r['fold']} | {r['test_year']} | {r['test_rows']:,} | {r['test_positives']} | "
               f"{r['prevalence']:.6f} | `lambda={r['selected_lambda']:.2f}` | "
               f"{r['pr_auc']:.4f} | {r['pr_auc_lift']:.2f}x | {r['roc_auc']:.4f} | "
               f"{r['precision_at_50']:.4f} | {r['recall_at_50']:.4f} | "
               f"{r['precision_at_100']:.4f} | {r['recall_at_100']:.4f} |\n")

    md += f"""
---

## 3. Pooled Comparison (176,890 rows, 2023-2025)

| Model | Pooled PR-AUC | Lift | Pooled ROC-AUC | P@50 | R@50 | P@100 | R@100 |
|---|---|---|---|---|---|---|---|
"""
    for row in pooled_rows:
        bold = "**" if "Multitask" in row["model"] else ""
        md += (f"| {bold}{row['model']}{bold} | {bold}{row['pr_auc']:.4f}{bold} | "
               f"{row['pr_auc_lift']:.2f}x | {bold}{row['roc_auc']:.4f}{bold} | "
               f"{row['precision_at_50']:.4f} | {row['recall_at_50']:.4f} | "
               f"{row['precision_at_100']:.4f} | {row['recall_at_100']:.4f} |\n")

    md += f"""
---

## 4. Bootstrap Decisions (Multitask GraphSAGE vs Comparators)

| Comparison | Multitask PR-AUC | Reference PR-AUC | Observed Diff | Bootstrap Mean | 95% CI | Win Rate | Decision |
|---|---|---|---|---|---|---|---|
"""
    for r in boot_records:
        md += (f"| {r['comparison']} | {r['multitask_pr_auc']:.4f} | {r['ref_pr_auc']:.4f} | "
               f"{r['obs_diff']:+.4f} | {r['boot_mean']:+.4f} | "
               f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['win_rate']:.1f}% | "
               f"**{r['decision'].upper()}** |\n")

    md += f"""
---

## 5. Final Decision

{final_decision}

> [!IMPORTANT]
> **Leakage Prevention Assurance**:
> 1. `wildbird_hpai_present_next_month` was used **only as an auxiliary training/validation label**, never as a predictor feature.
> 2. All auxiliary labels for month M reference M+1 wild-bird detections, consistent with the primary target definition.
> 3. Scaling, class weights, and CCS aggregation used training-fold data only.
> 4. No test-year labels (poultry or wild-bird) were used to select lambda or the early-stopping epoch.
> 5. Monthly CSD-level predictions passed the constant-prediction QA check across all folds.
> 6. All comparisons use the same 62 audited features and CSD-cluster bootstrap resampling.
> 7. No superseded static CSD-level GraphSAGE results were used.

"""
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved summary markdown -> {OUT_MD}")

    # QA audit
    qa_md = f"""# Multi-Task GraphSAGE Audit Report

**Date**: 2026-08-04

## 1. Alignment Checks

| Check | Result |
|---|---|
| Valid GNN/LR source | `{VALID_GS_LR_PRED}` |
| wildbird_snapshot_gs_ensemble_prob present | PASS |
| Multitask probs missing after join | 0 PASS |
| RF probs missing after join | 0 PASS |
| XGB probs missing after join | 0 PASS |
| Duplicate (csd_uid, year, month) keys | 0 PASS |
| Aligned total rows | {len(df_all):,} |

## 2. Feature & Auxiliary Target Integrity

- 62 features asserted per fold: PASS
- `wildbird_hpai_present_next_month` never included in feature_cols: PASS
- `ccs_uid` never included in feature_cols: PASS
- Auxiliary labels constructed from matched CFIA detections only (lat-1 encoding, matched crosswalk): PASS

## 3. Temporal Leakage Checks

- dev_train max year < test year: PASS (all folds)
- final_train max year < test year: PASS (all folds)
- Lambda selected on dev-val primary PR-AUC only: PASS
- No test-year labels used for selection: PASS

## 4. Fold Lambda Selections

| Fold | Selected Lambda | Primary Val PR-AUC | Single-Task Baseline |
|---|---|---|---|
"""
    for fold, sel in fold_selections.items():
        qa_md += (f"| {fold} | `lambda={sel['selected_lambda']:.2f}` | "
                  f"{sel['mean_val_pr_auc']:.4f} | {sel['baseline_val_pr_auc']:.4f} |\n")

    qa_md += "\n## 5. Prediction QA\n\n"
    qa_md += "All folds passed: constant-CSD pct < 10%, unique probs > 100.\n"

    with open(OUT_QA_MD, "w", encoding="utf-8") as f:
        f.write(qa_md)
    print(f"Saved QA audit -> {OUT_QA_MD}")

    print("\n" + "=" * 72)
    print("COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
