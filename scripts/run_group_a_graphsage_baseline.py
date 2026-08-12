#!/usr/bin/env python3
"""
Group A baseline reproduction for the HPAI GraphSAGE improvement plan.

This script intentionally avoids importing older baseline scripts because
multiple legacy variants use conflicting feature sets, graphs, and temporal
splits. It runs a fresh, auditable baseline with:

- Data: model_table_v1_environment_with_lags_wildbird.csv
- Graph: csd_graph_knn_3_v1.csv
- Features: all numeric columns excluding identifiers/targets/time fields
- Train: 2022-2023
- Validation: 2024
- Locked test: 2025
- Architecture: 2-layer GraphSAGE, mean aggregation, hidden=64, dropout=0.2
- Loss: BCEWithLogitsLoss with train-only positive class weighting
- Seeds: 11, 22, 33, 44, 55

Outputs are written under results/group_a_baseline_graphsage_knn3_2022_2025/.
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
INPUT_MODEL_TABLE = ROOT / "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
GRAPH_EDGE_CSV = ROOT / "data/processed/graph/csd_graph_knn_3_v1.csv"
OUT_DIR = ROOT / "results/group_a_baseline_graphsage_knn3_2022_2025"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
CURVE_DIR = OUT_DIR / "loss_curves"

SEEDS = [11, 22, 33, 44, 55]
HIDDEN_DIM = 64
DROPOUT = 0.2
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 15

EXCLUDE_EXPLICIT = {
    "csd_uid",
    "csd_name",
    "province_code",
    "province_name",
    "ccs_uid",
    "year",
    "month",
    "outbreak_next_month",
    "dw_fill_method",
    "climate_extraction_method",
    "glw2020_chicken_extraction_method",
    "glw2015_duck_extraction_method",
    "road_data_coverage",
    "outbreak_current_month",
    "wildbird_hpai_present_next_month",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in EXCLUDE_EXPLICIT:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    cols = sorted(cols)
    if len(cols) != 62:
        raise ValueError(f"Expected 62 numeric baseline features, found {len(cols)}")
    return cols


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


def load_edge_index(all_csds: list[str]) -> torch.Tensor:
    edges = pd.read_csv(GRAPH_EDGE_CSV, dtype=str)
    csd_to_idx = {csd: idx for idx, csd in enumerate(all_csds)}
    src = edges["source_csd_uid"].map(csd_to_idx)
    dst = edges["target_csd_uid"].map(csd_to_idx)
    valid = src.notna() & dst.notna()
    src_arr = src[valid].astype(int).to_numpy()
    dst_arr = dst[valid].astype(int).to_numpy()
    edge_index = np.stack(
        [np.concatenate([src_arr, dst_arr]), np.concatenate([dst_arr, src_arr])],
        axis=0,
    )
    return torch.tensor(edge_index, dtype=torch.long)


def build_snapshots(
    df: pd.DataFrame,
    feature_cols: list[str],
    csd_to_idx: dict[str, int],
    edge_index: torch.Tensor,
) -> SnapshotBundle:
    snapshots: list[Data] = []
    ordered_rows: list[pd.DataFrame] = []
    num_nodes = len(csd_to_idx)

    for (year, month), group in df.groupby(["year", "month"], sort=True):
        group = group.sort_values("csd_uid").reset_index(drop=True)
        node_idx = group["csd_uid"].map(csd_to_idx).to_numpy()

        x = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)
        y = np.zeros(num_nodes, dtype=np.float32)
        mask = np.zeros(num_nodes, dtype=bool)

        x[node_idx] = group[feature_cols].to_numpy(dtype=np.float32)
        y[node_idx] = group["outbreak_next_month"].to_numpy(dtype=np.float32)
        mask[node_idx] = True

        snapshots.append(
            Data(
                x=torch.from_numpy(x),
                y=torch.from_numpy(y),
                mask=torch.from_numpy(mask),
                edge_index=edge_index,
                year=int(year),
                month=int(month),
            )
        )
        ordered_rows.append(
            group[["csd_uid", "csd_name", "province_code", "year", "month", "outbreak_next_month"]]
            .rename(columns={"outbreak_next_month": "actual"})
            .copy()
        )

    return SnapshotBundle(snapshots=snapshots, rows=pd.concat(ordered_rows, ignore_index=True))


def predict_snapshots(model: GraphSAGEModel, snapshots: list[Data]) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for snap in snapshots:
            logits = model(snap.x, snap.edge_index)
            probs = torch.sigmoid(logits)[snap.mask].cpu().numpy()
            outputs.append(probs)
    return np.concatenate(outputs)


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


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading audited baseline inputs...")
    df = pd.read_csv(INPUT_MODEL_TABLE, dtype={"csd_uid": str})
    cols = feature_columns(df)
    print(f"Feature count verified: {len(cols)}")

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
    edge_index = load_edge_index(all_csds)

    train_bundle = build_snapshots(df_train, cols, csd_to_idx, edge_index)
    val_bundle = build_snapshots(df_val, cols, csd_to_idx, edge_index)
    test_bundle = build_snapshots(df_test, cols, csd_to_idx, edge_index)

    train_labels = np.concatenate([snap.y[snap.mask].numpy() for snap in train_bundle.snapshots])
    train_pos = float(train_labels.sum())
    train_neg = float(len(train_labels) - train_pos)
    pos_weight = torch.tensor([train_neg / train_pos], dtype=torch.float32)

    lineage = {
        "dataset": str(INPUT_MODEL_TABLE.relative_to(ROOT)),
        "graph": str(GRAPH_EDGE_CSV.relative_to(ROOT)),
        "feature_count": len(cols),
        "feature_names": cols,
        "split": {"train_years": [2022, 2023], "validation_years": [2024], "locked_test_years": [2025]},
        "architecture": {
            "layers": 2,
            "aggregator": "mean",
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "loss": "BCEWithLogitsLoss(pos_weight=train_neg/train_pos)",
        },
        "random_seeds": SEEDS,
        "lineage_note": (
            "Legacy scripts conflict on hidden size and validation split. "
            "This Group A reproduction uses the audited 2022-2023/2024/2025 split "
            "and hidden_dim=64 to match the current strict baseline registry."
        ),
    }
    save_json(OUT_DIR / "run_manifest.json", lineage)

    seed_rows: list[dict[str, float | int | str]] = []
    ensemble_val_cols: list[str] = []
    ensemble_test_cols: list[str] = []
    val_pred_df = val_bundle.rows.copy()
    test_pred_df = test_bundle.rows.copy()

    for seed in SEEDS:
        print(f"Running seed {seed}...")
        set_seed(seed)
        model = GraphSAGEModel(in_channels=len(cols), hidden_channels=HIDDEN_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_state = None
        best_epoch = 0
        best_val_ap = -1.0
        epochs_without_improvement = 0
        curve_rows: list[dict[str, float | int]] = []

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            train_losses: list[float] = []
            for snap in train_bundle.snapshots:
                optimizer.zero_grad()
                logits = model(snap.x, snap.edge_index)
                loss = criterion(logits[snap.mask], snap.y[snap.mask])
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            val_losses: list[float] = []
            model.eval()
            with torch.no_grad():
                for snap in val_bundle.snapshots:
                    logits = model(snap.x, snap.edge_index)
                    loss = criterion(logits[snap.mask], snap.y[snap.mask])
                    val_losses.append(float(loss.item()))

            val_probs = predict_snapshots(model, val_bundle.snapshots)
            val_eval = val_bundle.rows.copy()
            val_eval["prob"] = val_probs
            val_metrics = compute_metrics(val_eval)
            mean_train_loss = float(np.mean(train_losses))
            mean_val_loss = float(np.mean(val_losses))
            curve_rows.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": mean_train_loss,
                    "val_loss": mean_val_loss,
                    "val_pr_auc": val_metrics["pr_auc"],
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_macro_precision_at_10": val_metrics["macro_precision_at_10"],
                    "val_macro_recall_at_50": val_metrics["macro_recall_at_50"],
                    "val_brier_score": val_metrics["brier_score"],
                }
            )

            if val_metrics["pr_auc"] > best_val_ap:
                best_val_ap = val_metrics["pr_auc"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                break

        if best_state is None:
            raise RuntimeError(f"No best model state captured for seed {seed}")

        model.load_state_dict(best_state)
        checkpoint_path = CHECKPOINT_DIR / f"seed_{seed}.pt"
        torch.save(
            {
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "feature_names": cols,
                "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT,
                "best_epoch": best_epoch,
                "best_val_pr_auc": best_val_ap,
            },
            checkpoint_path,
        )

        curve_path = CURVE_DIR / f"seed_{seed}_loss_curve.csv"
        pd.DataFrame(curve_rows).to_csv(curve_path, index=False)

        val_probs = predict_snapshots(model, val_bundle.snapshots)
        test_probs = predict_snapshots(model, test_bundle.snapshots)

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

        val_metrics = compute_metrics(val_eval)
        test_metrics = compute_metrics(test_eval)
        seed_rows.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
                "loss_curve_path": str(curve_path.relative_to(ROOT)),
                "val_prediction_path": str((OUT_DIR / "validation_predictions.csv").relative_to(ROOT)),
                "test_prediction_path": str((OUT_DIR / "test_predictions.csv").relative_to(ROOT)),
                "val_pr_auc": val_metrics["pr_auc"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_macro_precision_at_10": val_metrics["macro_precision_at_10"],
                "val_macro_recall_at_50": val_metrics["macro_recall_at_50"],
                "val_brier_score": val_metrics["brier_score"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_macro_precision_at_10": test_metrics["macro_precision_at_10"],
                "test_macro_recall_at_50": test_metrics["macro_recall_at_50"],
                "test_brier_score": test_metrics["brier_score"],
            }
        )

    val_pred_df["ensemble_prob"] = val_pred_df[ensemble_val_cols].mean(axis=1)
    test_pred_df["ensemble_prob"] = test_pred_df[ensemble_test_cols].mean(axis=1)
    val_pred_df.to_csv(OUT_DIR / "validation_predictions.csv", index=False)
    test_pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    ensemble_val_eval = val_bundle.rows.copy()
    ensemble_val_eval["prob"] = val_pred_df["ensemble_prob"].to_numpy()
    ensemble_test_eval = test_bundle.rows.copy()
    ensemble_test_eval["prob"] = test_pred_df["ensemble_prob"].to_numpy()
    ensemble_val_metrics = compute_metrics(ensemble_val_eval)
    ensemble_test_metrics = compute_metrics(ensemble_test_eval)

    metrics_df = pd.DataFrame(seed_rows)
    ensemble_row = {
        "seed": "ensemble",
        "best_epoch": "",
        "checkpoint_path": "",
        "loss_curve_path": "",
        "val_prediction_path": str((OUT_DIR / "validation_predictions.csv").relative_to(ROOT)),
        "test_prediction_path": str((OUT_DIR / "test_predictions.csv").relative_to(ROOT)),
        "val_pr_auc": ensemble_val_metrics["pr_auc"],
        "val_roc_auc": ensemble_val_metrics["roc_auc"],
        "val_macro_precision_at_10": ensemble_val_metrics["macro_precision_at_10"],
        "val_macro_recall_at_50": ensemble_val_metrics["macro_recall_at_50"],
        "val_brier_score": ensemble_val_metrics["brier_score"],
        "test_pr_auc": ensemble_test_metrics["pr_auc"],
        "test_roc_auc": ensemble_test_metrics["roc_auc"],
        "test_macro_precision_at_10": ensemble_test_metrics["macro_precision_at_10"],
        "test_macro_recall_at_50": ensemble_test_metrics["macro_recall_at_50"],
        "test_brier_score": ensemble_test_metrics["brier_score"],
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([ensemble_row])], ignore_index=True)
    metrics_df.to_csv(OUT_DIR / "seed_metrics.csv", index=False)

    summary = {
        "seed_mean_test_pr_auc": float(pd.to_numeric(metrics_df.loc[metrics_df["seed"] != "ensemble", "test_pr_auc"]).mean()),
        "seed_sd_test_pr_auc": float(pd.to_numeric(metrics_df.loc[metrics_df["seed"] != "ensemble", "test_pr_auc"]).std()),
        "ensemble_test_pr_auc": ensemble_test_metrics["pr_auc"],
        "ensemble_test_roc_auc": ensemble_test_metrics["roc_auc"],
        "ensemble_test_macro_precision_at_10": ensemble_test_metrics["macro_precision_at_10"],
        "ensemble_test_macro_recall_at_50": ensemble_test_metrics["macro_recall_at_50"],
        "ensemble_test_brier_score": ensemble_test_metrics["brier_score"],
    }
    save_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
