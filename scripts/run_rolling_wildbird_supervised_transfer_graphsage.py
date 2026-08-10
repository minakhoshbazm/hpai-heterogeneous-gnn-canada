#!/usr/bin/env python3
"""
scripts/run_rolling_wildbird_supervised_transfer_graphsage.py

Sequential Supervised Transfer Learning: Wild-Bird HPAI Pretraining -> Poultry Fine-tuning.

Motivation:
  Poultry outbreaks have only 381 positive CSD-months across the full dataset (~0.16% prevalence).
  Wild-bird HPAI detections have ~7,540 positive CSD-months (~3.2% prevalence) -- 20x more signal.
  Strategy: pretrain the GraphSAGE encoder on the denser wild-bird detection task first,
  then fine-tune on the sparse poultry outbreak task using the same rolling-origin folds.

Key difference from prior experiments:
  - vs. multitask GraphSAGE: Sequential not joint. Encoder is pretrained to convergence on
    wild-bird detection before the poultry fine-tuning phase begins.
  - vs. masked reconstruction pretraining: Supervised signal (wild-bird HPAI presence next month)
    rather than self-supervised reconstruction -- much richer epidemiological training signal.
  - vs. standard kNN-3 GraphSAGE: Encoder initialized from wild-bird pretraining, not random.

Architecture:
  GraphSAGEEncoder (2-layer, hidden=32) shared between phases.
  Phase 1 (pretraining): encoder + wildbird_head (Linear 32->1)
    Loss: BCEWithLogitsLoss(pos_weight=N_neg/N_pos for wildbird)
    Target: wildbird_hpai_present_next_month (built from raw CFIA wild-bird data)
    Early stop: validation wild-bird PR-AUC
  Phase 2 (fine-tuning): pretrained encoder + poultry_head (Linear 32->1)
    Loss: BCEWithLogitsLoss(pos_weight=N_neg/N_pos for poultry)
    Target: outbreak_next_month
    Early stop: validation poultry PR-AUC
    LR: 0.001 (10x lower than pretraining to preserve learned features)

Rolling Folds (identical splits to valid rolling runs):
  Fold 1: DevTrain 2022-01..09, DevVal 2022-10..12, FinalTrain 2022, Test 2023
  Fold 2: DevTrain 2022-01..2023-09, DevVal 2023-10..12, FinalTrain 2022-2023, Test 2024
  Fold 3: DevTrain 2022-01..2024-09, DevVal 2024-10..12, FinalTrain 2022-2024, Test 2025

Outputs:
  results/baselines/rolling_wildbird_supervised_transfer_graphsage_fold_metrics.csv
  results/baselines/rolling_wildbird_supervised_transfer_graphsage_test_predictions.csv
  results/baselines/rolling_wildbird_supervised_transfer_graphsage_bootstrap.csv
  results/baselines/rolling_wildbird_supervised_transfer_graphsage_summary.md
"""

import os
import copy
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

# ─────────────────────────────────────────────────────────────────────────────
# Environment (single-threaded for reproducibility)
# ─────────────────────────────────────────────────────────────────────────────
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
INPUT_TABLE     = "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
WB_RAW_CSV      = "data/raw/cfia/CFIA_outbreaks_wildbirds.csv"
WB_CROSSWALK    = "data/processed/geography/wildbird_cfia_ccs_crosswalk_reviewed.csv"
GRAPH_EDGE_CSV  = "data/processed/graph/csd_graph_knn_3_v1.csv"

# Baseline predictions for bootstrap comparison
VALID_GS_LR_PRED = "results/baselines/rolling_wildbird_monthly_snapshot_graphsage_vs_lr_predictions.csv"

OUT_DIR         = "results/baselines"
OUT_FOLD_CSV    = os.path.join(OUT_DIR, "rolling_wildbird_supervised_transfer_graphsage_fold_metrics.csv")
OUT_PRED_CSV    = os.path.join(OUT_DIR, "rolling_wildbird_supervised_transfer_graphsage_test_predictions.csv")
OUT_BOOT_CSV    = os.path.join(OUT_DIR, "rolling_wildbird_supervised_transfer_graphsage_bootstrap.csv")
OUT_MD          = os.path.join(OUT_DIR, "rolling_wildbird_supervised_transfer_graphsage_summary.md")

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
SEEDS        = [11, 22, 33]
HIDDEN       = 32
PRETRAIN_LR  = 0.005
FINETUNE_LR  = 0.001   # Smaller LR: preserve pretrained features
WEIGHT_DECAY = 5e-4
MAX_EPOCHS   = 200
PATIENCE     = 20

# Columns excluded from node feature set
EXCLUDE_EXPLICIT = {
    "csd_uid", "csd_name", "province_code", "province_name", "ccs_uid",
    "year", "month", "outbreak_next_month",
    "dw_fill_method", "climate_extraction_method",
    "glw2020_chicken_extraction_method", "glw2015_duck_extraction_method",
    "road_data_coverage", "outbreak_current_month",
}

# ─────────────────────────────────────────────────────────────────────────────
# Rolling Fold Definitions
# ─────────────────────────────────────────────────────────────────────────────
FOLDS = [
    {
        "fold_id":    1,
        "test_year":  2023,
        "dev_train":  lambda df: df[(df["year"] == 2022) & (df["month"] <= 9)],
        "dev_val":    lambda df: df[(df["year"] == 2022) & (df["month"] >= 10)],
        "final_train": lambda df: df[df["year"] == 2022],
        "test":       lambda df: df[df["year"] == 2023],
    },
    {
        "fold_id":    2,
        "test_year":  2024,
        "dev_train":  lambda df: df[((df["year"] == 2022)) | ((df["year"] == 2023) & (df["month"] <= 9))],
        "dev_val":    lambda df: df[(df["year"] == 2023) & (df["month"] >= 10)],
        "final_train": lambda df: df[df["year"].isin([2022, 2023])],
        "test":       lambda df: df[df["year"] == 2024],
    },
    {
        "fold_id":    3,
        "test_year":  2025,
        "dev_train":  lambda df: df[(df["year"].isin([2022, 2023])) | ((df["year"] == 2024) & (df["month"] <= 9))],
        "dev_val":    lambda df: df[(df["year"] == 2024) & (df["month"] >= 10)],
        "final_train": lambda df: df[df["year"].isin([2022, 2023, 2024])],
        "test":       lambda df: df[df["year"] == 2025],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden=32):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        return x


class TransferClassifier(nn.Module):
    def __init__(self, encoder, hidden=32):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        return self.head(h).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def precision_recall_at_k(y_true, y_prob, k):
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
    top = df.head(k)
    total_pos = float(y_true.sum())
    prec = top["y"].mean()
    rec  = top["y"].sum() / total_pos if total_pos > 0 else 0.0
    return float(prec), float(rec)


def compute_metrics(y_true, y_prob):
    pr_auc = average_precision_score(y_true, y_prob)
    roc    = roc_auc_score(y_true, y_prob)
    p50, r50   = precision_recall_at_k(y_true, y_prob, 50)
    p100, r100 = precision_recall_at_k(y_true, y_prob, 100)
    p200, r200 = precision_recall_at_k(y_true, y_prob, 200)
    prev = float(np.mean(y_true))
    lift = pr_auc / prev if prev > 0 else 0.0
    return {
        "pr_auc": pr_auc, "roc_auc": roc, "lift": lift,
        "precision_at_50": p50, "recall_at_50": r50,
        "precision_at_100": p100, "recall_at_100": r100,
        "precision_at_200": p200, "recall_at_200": r200,
    }


def load_graph(edge_csv, uid_to_idx):
    edges = pd.read_csv(edge_csv)
    src = edges["source_csd_uid"].map(uid_to_idx).dropna().astype(int).values
    dst = edges["target_csd_uid"].map(uid_to_idx).dropna().astype(int).values
    return torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)


def build_wildbird_target(wb_raw_path, crosswalk_path, df_model):
    """
    Build wildbird_hpai_present_next_month:
    1 if a wild-bird HPAI detection occurred in the CCS in the following month.
    Joins on (ccs_uid, year, month) via a 1-month forward shift.
    """
    wb = pd.read_csv(wb_raw_path, encoding="latin-1")
    xw = pd.read_csv(crosswalk_path, encoding="latin-1")

    xw_matched = xw[xw["final_status"] == "matched"][
        ["raw_ccs_name", "raw_province", "target_ccs_uid"]
    ].dropna()

    # Parse detection dates — actual column in CFIA wild-bird CSV is 'Collection Date'
    date_col = None
    for c in ["Collection Date", "Outbreak Start Date", "Reported Date", "Event Start Date", "Date"]:
        if c in wb.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"Cannot find date column in wild-bird CSV. Columns: {list(wb.columns)}")

    wb["date_parsed"] = pd.to_datetime(wb[date_col], errors="coerce")
    wb = wb.dropna(subset=["date_parsed"])
    wb["year"]  = wb["date_parsed"].dt.year
    wb["month"] = wb["date_parsed"].dt.month

    # Match wild-bird records to CCS UIDs
    # Actual column in CFIA wild-bird CSV: 'Census Consolidated Subdivision (CCS)'
    ccs_col, prov_col = None, None
    for c in ["Census Consolidated Subdivision (CCS)", "Region Name", "CCS Name", "CCS", "Region"]:
        if c in wb.columns:
            ccs_col = c
            break
    for c in ["Province", "Province/Territory", "Prov"]:
        if c in wb.columns:
            prov_col = c
            break

    wb["raw_ccs_name"] = wb[ccs_col].str.strip().str.lower() if ccs_col else ""
    wb["raw_province"] = wb[prov_col].str.strip().str.lower() if prov_col else ""

    xw_matched = xw_matched.copy()
    xw_matched["raw_ccs_name"] = xw_matched["raw_ccs_name"].str.strip().str.lower()
    xw_matched["raw_province"] = xw_matched["raw_province"].str.strip().str.lower()

    wb_m = wb.merge(xw_matched, on=["raw_ccs_name", "raw_province"], how="inner")

    if len(wb_m) == 0:
        # Fallback: use wildbird_hpai_count_ccs_current_month already in the table
        print("  [WARN] Wild-bird crosswalk matched 0 records. Falling back to table column.")
        wb_presence = (
            df_model[["ccs_uid", "year", "month", "wildbird_hpai_count_ccs_current_month"]]
            .assign(wildbird_hpai_present=lambda d: (d["wildbird_hpai_count_ccs_current_month"] > 0).astype(float))
        )
    else:
        wb_agg = (
            wb_m.groupby(["target_ccs_uid", "year", "month"])
            .size()
            .reset_index(name="wb_count")
            .rename(columns={"target_ccs_uid": "ccs_uid"})
        )
        wb_agg["wildbird_hpai_present"] = 1.0
        wb_agg["ccs_uid"] = wb_agg["ccs_uid"].astype(str)  # ensure string type

        all_ccs_months = df_model[["ccs_uid", "year", "month"]].drop_duplicates().copy()
        all_ccs_months["ccs_uid"] = all_ccs_months["ccs_uid"].astype(str)  # match dtype
        wb_presence = all_ccs_months.merge(
            wb_agg[["ccs_uid", "year", "month", "wildbird_hpai_present"]],
            on=["ccs_uid", "year", "month"], how="left"
        ).fillna({"wildbird_hpai_present": 0.0})

    # Shift to create next-month target: for row (ccs, y, m), next-month target =
    # wild-bird presence in (ccs, y, m+1) using the same CCS aggregation.
    wb_pres_sorted = wb_presence.sort_values(["ccs_uid", "year", "month"])
    wb_pres_sorted["next_year"]  = wb_pres_sorted["year"]
    wb_pres_sorted["next_month"] = wb_pres_sorted["month"]

    # Build (year, month) -> next (year, month) mapping
    def next_ym(y, m):
        if m == 12:
            return (y + 1, 1)
        return (y, m + 1)

    wb_next = wb_pres_sorted[["ccs_uid", "year", "month", "wildbird_hpai_present"]].copy()
    wb_next.columns = ["ccs_uid", "next_year", "next_month", "wildbird_hpai_present_next_month"]
    wb_next["ccs_uid"] = wb_next["ccs_uid"].astype(str)

    # Join next-month wild-bird presence to current month rows
    # df_model row (ccs, y, m) needs next_year=y, next_month=m+1 from wb_next
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


def build_snapshots(df_split, df_wb_target, feature_cols, uid_to_idx, n_nodes):
    """Build monthly snapshot list for a data split."""
    snapshots = []
    for (yr, mo), group in df_split.groupby(["year", "month"], sort=True):
        group = group.copy()

        # Attach wild-bird target
        group = group.merge(
            df_wb_target[df_wb_target["year"] == yr][df_wb_target["month"] == mo]
            [["csd_uid", "wildbird_hpai_present_next_month"]],
            on="csd_uid", how="left"
        ).fillna({"wildbird_hpai_present_next_month": 0.0})

        node_idx = group["csd_uid"].map(uid_to_idx)
        group    = group.loc[node_idx.notna().values]
        node_idx = node_idx.dropna().astype(int).values

        x     = np.zeros((n_nodes, len(feature_cols)), dtype=np.float32)
        y_ptr = np.zeros((n_nodes,), dtype=np.float32)   # poultry outbreak
        y_wb  = np.zeros((n_nodes,), dtype=np.float32)   # wildbird detection
        mask  = np.zeros((n_nodes,), dtype=bool)

        x[node_idx]     = group[feature_cols].to_numpy(dtype=np.float32)
        y_ptr[node_idx] = group["outbreak_next_month"].to_numpy(dtype=np.float32)
        y_wb[node_idx]  = group["wildbird_hpai_present_next_month"].to_numpy(dtype=np.float32)
        mask[node_idx]  = True

        snapshots.append({
            "year": int(yr), "month": int(mo),
            "x":    torch.from_numpy(x),
            "y_ptr": torch.from_numpy(y_ptr),
            "y_wb":  torch.from_numpy(y_wb),
            "mask":  torch.from_numpy(mask),
            "csd_uid":       group["csd_uid"].to_numpy(),
            "csd_name":      group["csd_name"].to_numpy(),
            "province_code": group["province_code"].to_numpy(),
        })
    return snapshots


def predict_probs(model, snapshots, edge_index, target="y_ptr"):
    model.eval()
    with torch.no_grad():
        probs = torch.cat([
            torch.sigmoid(model(s["x"], edge_index))[s["mask"]]
            for s in snapshots
        ]).cpu().numpy()
    labels = np.concatenate([s[target][s["mask"]].numpy() for s in snapshots])
    return probs, labels


def train_phase(model, optimizer, criterion, train_snaps, val_snaps,
                edge_index, max_epochs, patience, target_key, phase_name, seed):
    """Generic training loop for one phase (pretraining or fine-tuning)."""
    best_val_ap   = -1.0
    best_state    = None
    best_epoch    = 0
    no_imp        = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        losses = []
        for s in train_snaps:
            if s["mask"].sum() == 0:
                continue
            out  = model(s["x"], edge_index)
            loss = criterion(out[s["mask"]], s[target_key][s["mask"]])
            losses.append(loss)
        if not losses:
            continue
        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        optimizer.step()

        # Validation PR-AUC
        model.eval()
        with torch.no_grad():
            val_probs = torch.cat([
                torch.sigmoid(model(s["x"], edge_index))[s["mask"]]
                for s in val_snaps
            ]).cpu().numpy()
        val_labels = np.concatenate([s[target_key][s["mask"]].numpy() for s in val_snaps])

        if val_labels.sum() == 0:
            continue
        val_ap = average_precision_score(val_labels, val_probs)

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state  = copy.deepcopy(model.state_dict())
            best_epoch  = epoch
            no_imp      = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"      [{phase_name}] Seed {seed}: best val PR-AUC={best_val_ap:.4f} @ epoch {best_epoch}")
    return best_val_ap, best_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 90)
    print("  ROLLING SUPERVISED TRANSFER LEARNING: Wild-Bird Pretraining -> Poultry Fine-Tuning")
    print("=" * 90)

    # ── Load model table ──────────────────────────────────────────────────────
    df = pd.read_csv(INPUT_TABLE, dtype={"csd_uid": str, "ccs_uid": str})
    print(f"Loaded table: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # Select 62-feature set (env + lag + 12 wildbird HPAI pressure features)
    feature_cols = sorted([
        c for c in df.columns
        if c not in EXCLUDE_EXPLICIT
        and not any(s in c.lower() for s in ["ebird", "gbif", "neighbor", "pressure"])
        and pd.api.types.is_numeric_dtype(df[c])
    ])
    print(f"Feature count: {len(feature_cols)}")
    assert len(feature_cols) == 62, f"Expected 62 features, got {len(feature_cols)}: {feature_cols}"

    # ── Build wild-bird HPAI next-month target ────────────────────────────────
    print("\nBuilding wild-bird HPAI next-month target from raw CFIA data...")
    df_wb = build_wildbird_target(WB_RAW_CSV, WB_CROSSWALK, df)
    wb_pos = (df_wb["wildbird_hpai_present_next_month"] > 0).sum()
    ptr_pos = (df["outbreak_next_month"] > 0).sum()
    print(f"  Wild-bird positives (next month): {wb_pos:,} ({wb_pos / len(df_wb):.2%})")
    print(f"  Poultry positives  (next month): {int(ptr_pos):,} ({ptr_pos / len(df):.2%})")
    print(f"  Wild-bird signal is {wb_pos / ptr_pos:.1f}x denser than poultry signal.")

    # ── Build graph ───────────────────────────────────────────────────────────
    all_csds   = sorted(df["csd_uid"].unique())
    uid_to_idx = {uid: i for i, uid in enumerate(all_csds)}
    n_nodes    = len(uid_to_idx)
    edge_index = load_graph(GRAPH_EDGE_CSV, uid_to_idx)
    print(f"\nGraph: {n_nodes:,} nodes, {edge_index.shape[1]:,} edges (kNN-3)")

    # ── Rolling evaluation ────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)

    all_fold_records = []
    all_pred_rows    = []

    for fold in FOLDS:
        fid      = fold["fold_id"]
        test_yr  = fold["test_year"]
        print(f"\n{'='*90}")
        print(f"  FOLD {fid} — Test Year: {test_yr}")
        print(f"{'='*90}")

        df_dev_tr  = fold["dev_train"](df).copy()
        df_dev_val = fold["dev_val"](df).copy()
        df_fin_tr  = fold["final_train"](df).copy()
        df_test    = fold["test"](df).copy()

        print(f"  Dev-Train: {len(df_dev_tr):,} rows | Dev-Val: {len(df_dev_val):,} rows")
        print(f"  Final-Train: {len(df_fin_tr):,} rows | Test: {len(df_test):,} rows")

        # ── Phase 1: Feature standardization ─────────────────────────────────
        # Fit scaler on dev-train only; apply to everything
        scaler = StandardScaler()
        df_dev_tr_s  = df_dev_tr.copy()
        df_dev_val_s = df_dev_val.copy()
        df_fin_tr_s  = df_fin_tr.copy()
        df_test_s    = df_test.copy()

        scaler.fit(df_dev_tr[feature_cols].fillna(0.0))
        for df_s, df_orig in [
            (df_dev_tr_s, df_dev_tr), (df_dev_val_s, df_dev_val),
            (df_fin_tr_s, df_fin_tr), (df_test_s, df_test)
        ]:
            df_s[feature_cols] = scaler.transform(df_orig[feature_cols].fillna(0.0))

        # Final-train scaler (fit on full final-train for final retraining)
        scaler_fin = StandardScaler()
        scaler_fin.fit(df_fin_tr[feature_cols].fillna(0.0))
        df_fin_tr_s2 = df_fin_tr.copy()
        df_test_s2   = df_test.copy()
        df_fin_tr_s2[feature_cols] = scaler_fin.transform(df_fin_tr[feature_cols].fillna(0.0))
        df_test_s2[feature_cols]   = scaler_fin.transform(df_test[feature_cols].fillna(0.0))

        # ── Build snapshots ───────────────────────────────────────────────────
        dev_tr_snaps  = build_snapshots(df_dev_tr_s,  df_wb, feature_cols, uid_to_idx, n_nodes)
        dev_val_snaps = build_snapshots(df_dev_val_s, df_wb, feature_cols, uid_to_idx, n_nodes)
        fin_tr_snaps  = build_snapshots(df_fin_tr_s2, df_wb, feature_cols, uid_to_idx, n_nodes)
        test_snaps    = build_snapshots(df_test_s2,   df_wb, feature_cols, uid_to_idx, n_nodes)

        # Compute class weights
        y_wb_train  = np.concatenate([s["y_wb"][s["mask"]].numpy() for s in dev_tr_snaps])
        y_ptr_train = np.concatenate([s["y_ptr"][s["mask"]].numpy() for s in dev_tr_snaps])

        n_pos_wb  = max(float(y_wb_train.sum()), 1.0)
        n_neg_wb  = float(len(y_wb_train)) - n_pos_wb
        pw_wb     = torch.tensor([n_neg_wb / n_pos_wb], dtype=torch.float)

        n_pos_ptr = max(float(y_ptr_train.sum()), 1.0)
        n_neg_ptr = float(len(y_ptr_train)) - n_pos_ptr
        pw_ptr    = torch.tensor([n_neg_ptr / n_pos_ptr], dtype=torch.float)

        print(f"\n  Wild-bird pos_weight (dev-train): {pw_wb.item():.1f}")
        print(f"  Poultry   pos_weight (dev-train): {pw_ptr.item():.1f}")

        # Wild-bird pos_weight for final retraining
        y_wb_fin  = np.concatenate([s["y_wb"][s["mask"]].numpy() for s in fin_tr_snaps])
        y_ptr_fin = np.concatenate([s["y_ptr"][s["mask"]].numpy() for s in fin_tr_snaps])
        n_pos_wb_fin  = max(float(y_wb_fin.sum()), 1.0)
        n_pos_ptr_fin = max(float(y_ptr_fin.sum()), 1.0)
        pw_wb_fin  = torch.tensor([(len(y_wb_fin) - n_pos_wb_fin) / n_pos_wb_fin], dtype=torch.float)
        pw_ptr_fin = torch.tensor([(len(y_ptr_fin) - n_pos_ptr_fin) / n_pos_ptr_fin], dtype=torch.float)

        # ── Multi-seed training loop ──────────────────────────────────────────
        fold_seed_preds = []

        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---")
            set_seed(seed)

            # ── Development Phase: Tune optimal epochs ────────────────────────

            # Phase 1: Pretrain on wild-bird detection (dev-train -> dev-val)
            encoder   = GraphSAGEEncoder(in_channels=len(feature_cols), hidden=HIDDEN)
            wb_head   = TransferClassifier(encoder, hidden=HIDDEN)
            opt_pre   = torch.optim.Adam(wb_head.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
            crit_pre  = nn.BCEWithLogitsLoss(pos_weight=pw_wb)

            wb_val_ap, pretrain_ep = train_phase(
                wb_head, opt_pre, crit_pre,
                dev_tr_snaps, dev_val_snaps,
                edge_index, MAX_EPOCHS, PATIENCE,
                target_key="y_wb",
                phase_name="Pretrain (WildBird)", seed=seed
            )

            # Phase 2: Fine-tune on poultry (dev-train -> dev-val)
            # Reuse the pretrained encoder; attach a new poultry head
            ptr_head  = TransferClassifier(encoder, hidden=HIDDEN)
            # Reset only the head weights (encoder keeps pretrained weights)
            nn.init.xavier_uniform_(ptr_head.head.weight)
            nn.init.zeros_(ptr_head.head.bias)

            opt_ft   = torch.optim.Adam(ptr_head.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
            crit_ft  = nn.BCEWithLogitsLoss(pos_weight=pw_ptr)

            ptr_val_ap, finetune_ep = train_phase(
                ptr_head, opt_ft, crit_ft,
                dev_tr_snaps, dev_val_snaps,
                edge_index, MAX_EPOCHS, PATIENCE,
                target_key="y_ptr",
                phase_name="Fine-tune (Poultry)", seed=seed
            )

            # ── Final Retraining (using full final-train set) ─────────────────
            set_seed(seed)
            enc_fin   = GraphSAGEEncoder(in_channels=len(feature_cols), hidden=HIDDEN)
            wb_fin    = TransferClassifier(enc_fin, hidden=HIDDEN)
            opt_pre_f = torch.optim.Adam(wb_fin.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
            crit_pre_f= nn.BCEWithLogitsLoss(pos_weight=pw_wb_fin)

            # Retrain pretraining for pretrain_ep epochs (no early stop)
            for ep in range(1, pretrain_ep + 1):
                wb_fin.train()
                opt_pre_f.zero_grad()
                losses = [
                    crit_pre_f(wb_fin(s["x"], edge_index)[s["mask"]], s["y_wb"][s["mask"]])
                    for s in fin_tr_snaps if s["mask"].sum() > 0
                ]
                if losses:
                    torch.stack(losses).mean().backward()
                    opt_pre_f.step()

            # Fine-tune for finetune_ep epochs (no early stop)
            ptr_fin  = TransferClassifier(enc_fin, hidden=HIDDEN)
            nn.init.xavier_uniform_(ptr_fin.head.weight)
            nn.init.zeros_(ptr_fin.head.bias)
            opt_ft_f  = torch.optim.Adam(ptr_fin.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
            crit_ft_f = nn.BCEWithLogitsLoss(pos_weight=pw_ptr_fin)

            for ep in range(1, finetune_ep + 1):
                ptr_fin.train()
                opt_ft_f.zero_grad()
                losses = [
                    crit_ft_f(ptr_fin(s["x"], edge_index)[s["mask"]], s["y_ptr"][s["mask"]])
                    for s in fin_tr_snaps if s["mask"].sum() > 0
                ]
                if losses:
                    torch.stack(losses).mean().backward()
                    opt_ft_f.step()

            # ── Evaluate on test set ──────────────────────────────────────────
            test_probs, test_labels = predict_probs(ptr_fin, test_snaps, edge_index, "y_ptr")
            metrics = compute_metrics(test_labels, test_probs)

            print(f"      Test | PR-AUC: {metrics['pr_auc']:.4f} | ROC: {metrics['roc_auc']:.4f} | "
                  f"P@50: {metrics['precision_at_50']:.3f} | R@50: {metrics['recall_at_50']:.3f}")
            print(f"             Lift: {metrics['lift']:.1f}x | pretrain_ep: {pretrain_ep} | finetune_ep: {finetune_ep}")

            fold_seed_preds.append(test_probs)
            all_fold_records.append({
                "fold_id": fid, "test_year": test_yr, "seed": seed,
                "pretrain_ep": pretrain_ep, "finetune_ep": finetune_ep,
                "dev_wb_val_ap": wb_val_ap, "dev_ptr_val_ap": ptr_val_ap,
                **{k: round(v, 6) for k, v in metrics.items()},
                "test_rows": len(test_labels), "test_pos": int(test_labels.sum()),
            })

        # ── Fold ensemble ─────────────────────────────────────────────────────
        ens_probs  = np.mean(np.stack(fold_seed_preds), axis=0)
        ens_labels = test_labels   # same for all seeds
        ens_m      = compute_metrics(ens_labels, ens_probs)
        print(f"\n  Fold {fid} Ensemble | PR-AUC: {ens_m['pr_auc']:.4f} | ROC: {ens_m['roc_auc']:.4f} | "
              f"P@50: {ens_m['precision_at_50']:.3f} | P@100: {ens_m['precision_at_100']:.3f} | "
              f"R@200: {ens_m['recall_at_200']:.3f}")

        # Store test predictions
        test_index = pd.DataFrame({
            "csd_uid":       np.concatenate([s["csd_uid"]       for s in test_snaps]),
            "csd_name":      np.concatenate([s["csd_name"]      for s in test_snaps]),
            "province_code": np.concatenate([s["province_code"] for s in test_snaps]),
            "year":  np.concatenate([np.full(len(s["csd_uid"]), s["year"])  for s in test_snaps]),
            "month": np.concatenate([np.full(len(s["csd_uid"]), s["month"]) for s in test_snaps]),
            "actual": ens_labels.astype(int),
        })
        for i, s in enumerate(SEEDS):
            test_index[f"transfer_gs_seed{s}_prob"] = fold_seed_preds[i]
        test_index["transfer_gs_ensemble_prob"] = ens_probs
        all_pred_rows.append(test_index)

    # ── Save fold metrics ─────────────────────────────────────────────────────
    df_fold = pd.DataFrame(all_fold_records)
    df_fold.to_csv(OUT_FOLD_CSV, index=False)
    print(f"\nSaved fold metrics -> {OUT_FOLD_CSV}")

    # ── Save predictions ──────────────────────────────────────────────────────
    df_preds = pd.concat(all_pred_rows, ignore_index=True)
    df_preds.to_csv(OUT_PRED_CSV, index=False)
    print(f"Saved test predictions -> {OUT_PRED_CSV}")

    # ── Pooled out-of-sample performance ─────────────────────────────────────
    pooled_y     = df_preds["actual"].values
    pooled_probs = df_preds["transfer_gs_ensemble_prob"].values
    pooled_m     = compute_metrics(pooled_y, pooled_probs)

    print(f"\n{'='*90}")
    print(f"  POOLED OUT-OF-SAMPLE RESULTS (Test Years 2023+2024+2025 Combined)")
    print(f"{'='*90}")
    print(f"  Total rows: {len(pooled_y):,} | Total positives: {int(pooled_y.sum())}")
    print(f"  Prevalence: {pooled_y.mean():.4%}")
    print(f"  PR-AUC: {pooled_m['pr_auc']:.4f} ({pooled_m['lift']:.1f}x lift over prevalence)")
    print(f"  ROC-AUC: {pooled_m['roc_auc']:.4f}")
    print(f"  P@50: {pooled_m['precision_at_50']:.3f} | R@50: {pooled_m['recall_at_50']:.3f}")
    print(f"  P@100: {pooled_m['precision_at_100']:.3f} | R@100: {pooled_m['recall_at_100']:.3f}")
    print(f"  P@200: {pooled_m['precision_at_200']:.3f} | R@200: {pooled_m['recall_at_200']:.3f}")

    # ── Bootstrap comparison vs. rolling GNN baseline ─────────────────────────
    boot_records = []
    if os.path.exists(VALID_GS_LR_PRED):
        df_base = pd.read_csv(VALID_GS_LR_PRED, dtype={"csd_uid": str})
        df_eval = df_preds.merge(
            df_base[["csd_uid", "year", "month",
                      "wildbird_snapshot_gs_ensemble_prob",
                      "wildbird_lr_prob"]],
            on=["csd_uid", "year", "month"], how="inner"
        )
        print(f"\n  Bootstrap eval rows: {len(df_eval):,} (joined vs baseline)")

        y_ev   = df_eval["actual"].values
        p_tgnn = df_eval["transfer_gs_ensemble_prob"].values
        p_gs   = df_eval["wildbird_snapshot_gs_ensemble_prob"].values
        p_lr   = df_eval["wildbird_lr_prob"].values

        unique_csds  = df_eval["csd_uid"].unique()
        csd_groups   = df_eval.groupby("csd_uid").indices
        n_csd        = len(unique_csds)
        rng          = np.random.RandomState(42)
        N_BOOT       = 1000

        diffs_vs_gs = []
        diffs_vs_lr = []

        for _ in range(N_BOOT):
            sc = rng.choice(unique_csds, size=n_csd, replace=True)
            ix = np.concatenate([csd_groups[c] for c in sc])
            yb = y_ev[ix]
            if yb.sum() == 0:
                continue
            diffs_vs_gs.append(
                average_precision_score(yb, p_tgnn[ix]) - average_precision_score(yb, p_gs[ix])
            )
            diffs_vs_lr.append(
                average_precision_score(yb, p_tgnn[ix]) - average_precision_score(yb, p_lr[ix])
            )

        obs_gs = average_precision_score(y_ev, p_gs)
        obs_lr = average_precision_score(y_ev, p_lr)
        obs_tg = average_precision_score(y_ev, p_tgnn)

        def bootstrap_summary(diffs, comp_name, comp_pr):
            ci = np.percentile(diffs, [2.5, 97.5])
            wr = float(np.mean(np.array(diffs) > 0))
            dec = ("statistically better" if ci[0] > 0
                   else "statistically worse" if ci[1] < 0
                   else "comparable, not proven better")
            print(f"\n  Bootstrap vs {comp_name}:")
            print(f"    Observed diff: {obs_tg - comp_pr:+.4f} | 95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                  f"Win rate: {wr:.1%} | Decision: {dec.upper()}")
            return {
                "comparator": comp_name,
                "transfer_gnn_pr_auc": round(obs_tg, 4),
                "baseline_pr_auc": round(comp_pr, 4),
                "observed_diff": round(obs_tg - comp_pr, 4),
                "ci_lower_95": round(ci[0], 4),
                "ci_upper_95": round(ci[1], 4),
                "win_rate": round(wr, 4),
                "decision": dec,
            }

        boot_records.append(bootstrap_summary(diffs_vs_gs, "Monthly-Snapshot GraphSAGE (Rolling)", obs_gs))
        boot_records.append(bootstrap_summary(diffs_vs_lr, "Wild-Bird Logistic Regression (Rolling)", obs_lr))

        df_boot = pd.DataFrame(boot_records)
        df_boot.to_csv(OUT_BOOT_CSV, index=False)
        print(f"\nSaved bootstrap results -> {OUT_BOOT_CSV}")
    else:
        print(f"  [WARN] Baseline predictions not found: {VALID_GS_LR_PRED}")
        print("         Skipping bootstrap comparison.")

    # ── Per-fold summary ──────────────────────────────────────────────────────
    per_fold = (
        df_fold.groupby(["fold_id", "test_year"])[["pr_auc", "precision_at_50", "recall_at_50"]]
        .mean().reset_index()
    )

    # ── Write Markdown summary ────────────────────────────────────────────────
    md = f"""# Rolling Supervised Transfer Learning: Wild-Bird Pretraining → Poultry Fine-Tuning

**Architecture**: kNN-3 Monthly-Snapshot GraphSAGE (same graph as valid rolling baseline)
**Pretraining Task**: Wild-bird HPAI presence prediction (next month) — {int((df_wb['wildbird_hpai_present_next_month']>0).sum()):,} positives
**Fine-tuning Task**: Poultry HPAI outbreak prediction (next month) — {int(ptr_pos):,} positives
**Wild-bird pretraining signal is {int((df_wb['wildbird_hpai_present_next_month']>0).sum()) / int(ptr_pos):.1f}x denser than poultry signal.**

## Rolling Fold Configurations
- Fold 1: Final-Train 2022, Test 2023
- Fold 2: Final-Train 2022-2023, Test 2024
- Fold 3: Final-Train 2022-2024, Test 2025

## Per-Fold Test Performance (Ensemble of 3 Seeds)

| Fold | Test Year | Test Rows | Positives | PR-AUC | Lift | ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""
    for fold in FOLDS:
        fid = fold["fold_id"]
        tyr = fold["test_year"]
        fold_rows = [r for r in all_fold_records if r["fold_id"] == fid]
        ens_probs_f = np.mean(
            np.stack([
                df_preds[df_preds["year"] == tyr][f"transfer_gs_seed{s}_prob"].values
                for s in SEEDS
            ]), axis=0
        )
        ens_labels_f = df_preds[df_preds["year"] == tyr]["actual"].values
        em = compute_metrics(ens_labels_f, ens_probs_f)
        prev = float(np.mean(ens_labels_f))
        md += (f"| {fid} | {tyr} | {len(ens_labels_f):,} | {int(ens_labels_f.sum())} | "
               f"**{em['pr_auc']:.4f}** | {em['lift']:.1f}x | {em['roc_auc']:.4f} | "
               f"{em['precision_at_50']:.3f} | {em['recall_at_50']:.3f} | "
               f"{em['precision_at_100']:.3f} | {em['recall_at_100']:.3f} | "
               f"{em['precision_at_200']:.3f} | {em['recall_at_200']:.3f} |\n")

    md += f"""
## Pooled Out-of-Sample Results (2023-2025 Combined, {len(pooled_y):,} rows, {int(pooled_y.sum())} positives)

| Model | Pooled PR-AUC | Lift | ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
|---|---|---|---|---|---|---|---|---|---|
| **Supervised Transfer GraphSAGE (Ensemble)** | **{pooled_m['pr_auc']:.4f}** | {pooled_m['lift']:.1f}x | {pooled_m['roc_auc']:.4f} | {pooled_m['precision_at_50']:.3f} | {pooled_m['recall_at_50']:.3f} | {pooled_m['precision_at_100']:.3f} | {pooled_m['recall_at_100']:.3f} | {pooled_m['precision_at_200']:.3f} | {pooled_m['recall_at_200']:.3f} |
"""

    if boot_records:
        md += """
## Bootstrap Comparison (1,000 CSD Cluster Resamples)

| Comparator | Transfer PR-AUC | Baseline PR-AUC | Observed Diff | 95% CI | Win Rate | Decision |
|---|---|---|---|---|---|---|
"""
        for r in boot_records:
            md += (f"| {r['comparator']} | {r['transfer_gnn_pr_auc']:.4f} | "
                   f"{r['baseline_pr_auc']:.4f} | {r['observed_diff']:+.4f} | "
                   f"[{r['ci_lower_95']:+.4f}, {r['ci_upper_95']:+.4f}] | "
                   f"{r['win_rate']:.1%} | **{r['decision'].upper()}** |\n")

    md += """
## Methodological Assurances
1. Scaler fit on training-fold data only; applied identically to val/test.
2. Wild-bird pretraining target constructed from raw CFIA wild-bird records using validated CCS crosswalk.
3. Dev-val used exclusively for early stopping; test year never accessed during development.
4. Optimal pretrain/finetune epochs determined from dev-val, applied to final retrain on full training fold.
5. Bootstrap uses CSD-cluster resampling to preserve spatial-temporal correlation.
6. No test-year labels (poultry or wild-bird) influenced any model selection decision.
"""

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved summary report -> {OUT_MD}")
    print("\nDone.")


if __name__ == "__main__":
    main()
