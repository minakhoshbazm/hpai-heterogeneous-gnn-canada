#!/usr/bin/env python3
"""
scripts/train_wildbird_graphsage_knn3_corrected.py

Trains a corrected wild‑bird‑enhanced kNN‑3 GraphSAGE model.

- Input table: data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv
- Graph: data/processed/graph/csd_graph_knn_3_v1.csv (edges defined by csd_uid)
- Features: strict 50 environment+lag plus 12 wild‑bird HPAI pressure (62 total)
- Excludes identifiers, uid fields, target columns, raw outbreak_current_month, metadata, eBird, GBIF, neighbor‑pressure.

Training protocol:
- Train: 2022‑2024 (2024‑Q4 used for early‑stopping validation)
- Test: 2025
- Seeds: [11, 22, 33]
- Architecture: two‑layer GraphSAGE (hidden 32) with dropout 0.2, final linear output.
- Class‑weight handling via positive/negative counts.
- Scaler fitted on training split only (StandardScaler).

Outputs (new, never overwritten):
- results/baselines/wildbird_graphsage_knn3_corrected_multiseed_metrics.csv
- results/baselines/wildbird_graphsage_knn3_corrected_test_predictions.csv
- results/baselines/wildbird_graphsage_knn3_corrected_summary.md
"""

import os
import random
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Environment configuration (single‑threaded to avoid BLAS issues)
# ---------------------------------------------------------------------------
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_MODEL_TABLE = "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
GRAPH_EDGE_CSV = "data/processed/graph/csd_graph_knn_3_v1.csv"

OUT_METRICS = "results/baselines/wildbird_graphsage_knn3_corrected_multiseed_metrics.csv"
OUT_PRED = "results/baselines/wildbird_graphsage_knn3_corrected_test_predictions.csv"
OUT_MD = "results/baselines/wildbird_graphsage_knn3_corrected_summary.md"

SEEDS = [11, 22, 33]

# ---------------------------------------------------------------------------
# Columns to exclude from features
# ---------------------------------------------------------------------------
EXCLUDE_EXPLICIT = {
    "csd_uid", "csd_name", "province_code", "province_name", "ccs_uid",
    "year", "month", "outbreak_next_month",
    "dw_fill_method", "climate_extraction_method",
    "glw2020_chicken_extraction_method", "glw2015_duck_extraction_method",
    "road_data_coverage", "outbreak_current_month",
    # eBird / GBIF / neighbor pressure columns would be named with these prefixes –
    # we simply drop any column containing these substrings.
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def precision_recall_at_k(y_true, y_prob, k):
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df = df.sort_values(by="y_prob", ascending=False)
    top = df.head(k)
    prec = top["y_true"].mean()
    total_pos = y_true.sum()
    rec = top["y_true"].sum() / total_pos if total_pos > 0 else 0.0
    return prec, rec

def evaluate_metrics(y_true, y_prob):
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    p50, r50 = precision_recall_at_k(y_true, y_prob, 50)
    p100, r100 = precision_recall_at_k(y_true, y_prob, 100)
    p200, r200 = precision_recall_at_k(y_true, y_prob, 200)
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision_at_50": p50,
        "recall_at_50": r50,
        "precision_at_100": p100,
        "recall_at_100": r100,
        "precision_at_200": p200,
        "recall_at_200": r200,
    }

class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden=32):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.out(x).squeeze(-1)

def load_graph(edge_csv: str, node_ids: pd.Series):
    # Edge list uses csd_uid identifiers; we need to map to consecutive indices
    edges = pd.read_csv(edge_csv)
    # Build mapping from csd_uid to index based on the full set of nodes we will see
    uid_to_idx = {uid: i for i, uid in enumerate(node_ids)}
    # Filter edges to those where both endpoints are present in uid_to_idx
    src = edges["source_csd_uid"].map(uid_to_idx).dropna().astype(int).values
    tgt = edges["target_csd_uid"].map(uid_to_idx).dropna().astype(int).values
    edge_index = np.stack([src, tgt], axis=0)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index, uid_to_idx

def main():
    print("=== Training corrected wild-bird GraphSAGE (kNN-3) ===")

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    df = pd.read_csv(INPUT_MODEL_TABLE)
    print(f"Loaded modeling table: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # Select feature columns (numeric, not excluded, not containing excluded substrings)
    feature_cols = []
    for c in df.columns:
        if c in EXCLUDE_EXPLICIT:
            continue
        if any(sub in c.lower() for sub in ["ebird", "gbif", "neighbor", "pressure"]):
            # neighbor‑pressure features are excluded per requirement
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)
    feature_cols = sorted(feature_cols)
    print(f"Selected {len(feature_cols)} feature columns (expected 62)")
    assert len(feature_cols) == 62, f"Expected 62 features, got {len(feature_cols)}"

    # -------------------------------------------------------------------
    # Train/val/test splits – same as baseline script
    # -------------------------------------------------------------------
    df_train = df[(df['year'] <= 2023) | ((df['year'] == 2024) & (df['month'] <= 9))].copy()
    df_val = df[(df['year'] == 2024) & (df['month'] >= 10)].copy()
    df_test = df[df['year'] == 2025].copy()

    # Keep only needed columns for node features
    node_ids = pd.concat([df_train['csd_uid'], df_val['csd_uid'], df_test['csd_uid']]).unique()
    edge_index, uid_to_idx = load_graph(GRAPH_EDGE_CSV, pd.Series(node_ids))
    n_nodes = len(uid_to_idx)
    print(f"Graph: {n_nodes} nodes, {edge_index.shape[1]} edges")

    # -------------------------------------------------------------------
    # Scale features (fit on training only)
    # -------------------------------------------------------------------
    scaler = StandardScaler()
    df_train_s = df_train.copy()
    df_val_s = df_val.copy()
    df_test_s = df_test.copy()
    df_train_s[feature_cols] = scaler.fit_transform(df_train[feature_cols].fillna(0.0))
    df_val_s[feature_cols] = scaler.transform(df_val[feature_cols].fillna(0.0))
    df_test_s[feature_cols] = scaler.transform(df_test[feature_cols].fillna(0.0))

    # -------------------------------------------------------------------
    # Monthly graph snapshots
    #
    # A graph node is one CSD, but its features are CSD-month values, so every
    # (year, month) gets its own feature matrix over the shared spatial edge
    # index. Writing all months into a single node feature matrix would let the
    # last month processed overwrite every earlier one, which both erases the
    # temporal signal and leaks later-period covariates into training.
    # -------------------------------------------------------------------
    def build_snapshots(df_split):
        snapshots = []
        for (yr, mo), group in df_split.groupby(['year', 'month'], sort=True):
            node_idx = group['csd_uid'].map(uid_to_idx)
            group = group.loc[node_idx.notna().values]
            node_idx = node_idx.dropna().astype(int).values

            x = np.zeros((n_nodes, len(feature_cols)), dtype=np.float32)
            x[node_idx] = group[feature_cols].to_numpy(dtype=np.float32)

            y = np.zeros((n_nodes,), dtype=np.float32)
            y[node_idx] = group['outbreak_next_month'].to_numpy(dtype=np.float32)

            mask = np.zeros((n_nodes,), dtype=bool)
            mask[node_idx] = True

            snapshots.append({
                'year': int(yr),
                'month': int(mo),
                'x': torch.from_numpy(x),
                'y': torch.from_numpy(y),
                'mask': torch.from_numpy(mask),
                'csd_uid': group['csd_uid'].to_numpy(),
                'csd_name': group['csd_name'].to_numpy(),
                'province_code': group['province_code'].to_numpy(),
            })
        return snapshots

    train_snaps = build_snapshots(df_train_s)
    val_snaps = build_snapshots(df_val_s)
    test_snaps = build_snapshots(df_test_s)
    print(f"Snapshots: train={len(train_snaps)}, val={len(val_snaps)}, test={len(test_snaps)}")

    # Labels concatenated in snapshot order; predictions below use the same order.
    y_train = torch.cat([s['y'][s['mask']] for s in train_snaps])
    y_val = torch.cat([s['y'][s['mask']] for s in val_snaps])
    y_test = torch.cat([s['y'][s['mask']] for s in test_snaps])

    def predict(model, snapshots):
        model.eval()
        with torch.no_grad():
            return torch.cat([
                torch.sigmoid(model(s['x'], edge_index))[s['mask']] for s in snapshots
            ]).cpu().numpy()

    # Row identifiers aligned to the concatenated test prediction vector.
    test_index = pd.DataFrame({
        'csd_uid': np.concatenate([s['csd_uid'] for s in test_snaps]),
        'csd_name': np.concatenate([s['csd_name'] for s in test_snaps]),
        'province_code': np.concatenate([s['province_code'] for s in test_snaps]),
        'year': np.concatenate([np.full(len(s['csd_uid']), s['year']) for s in test_snaps]),
        'month': np.concatenate([np.full(len(s['csd_uid']), s['month']) for s in test_snaps]),
        'actual': y_test.numpy(),
    })

    # -------------------------------------------------------------------
    # Training loop per seed
    # -------------------------------------------------------------------
    seed_results = []
    test_probs_seeds = []
    patience = 20

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)
        model = GraphSAGEModel(in_channels=len(feature_cols), hidden=32)
        model.to("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        # Class weight (positive/negative ratio in training set)
        pos = y_train.sum().item()
        neg = y_train.shape[0] - pos
        pos_weight = torch.tensor([neg / pos], dtype=torch.float)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_ap = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, 201):
            model.train()
            optimizer.zero_grad()
            # One forward pass per training month; loss averaged over snapshots.
            losses = [
                criterion(model(s['x'], edge_index)[s['mask']], s['y'][s['mask']])
                for s in train_snaps
            ]
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()

            # Validation PR‑AUC
            val_probs = predict(model, val_snaps)
            val_ap = average_precision_score(y_val.numpy(), val_probs)
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping epoch {epoch}, best Val PR-AUC {best_val_ap:.4f}")
                    break
        # Load best model
        model.load_state_dict(best_state)
        test_probs = predict(model, test_snaps)
        # Sanity checks
        within_csd_std = (
            pd.Series(test_probs).groupby(test_index['csd_uid'].values).std().mean()
        )
        prob_stats = {
            "min": float(test_probs.min()),
            "max": float(test_probs.max()),
            "mean": float(test_probs.mean()),
            "std": float(test_probs.std()),
            "unique": int(np.unique(test_probs).shape[0]),
            "mean_within_csd_std": float(within_csd_std),
        }
        print("Prediction stats:", prob_stats)
        assert prob_stats["std"] > 1e-6, "Predicted probabilities are constant or collapsed"
        # A model that ignores the month would emit one score per CSD for the whole
        # test year, which is what the previous collapsed feature matrix produced.
        assert prob_stats["mean_within_csd_std"] > 1e-6, (
            "Predictions do not vary across months within a CSD; "
            "node features are not month-specific"
        )
        # Top‑20 predictions with actual labels
        top20_idx = np.argsort(-test_probs)[:20]
        top20 = test_index.iloc[top20_idx][["csd_uid", "year", "month"]].copy()
        top20["prob"] = test_probs[top20_idx]
        top20["actual"] = y_test.numpy()[top20_idx]
        print("Top-20 predictions (prob, actual):")
        print(top20)

        metrics = evaluate_metrics(y_test.numpy(), test_probs)
        metrics.update({"seed": seed, "val_pr_auc": best_val_ap, **prob_stats})
        seed_results.append(metrics)
        test_probs_seeds.append(test_probs)

    # -------------------------------------------------------------------
    # Aggregate results
    # -------------------------------------------------------------------
    # Ensemble (mean across seeds)
    ensemble_probs = np.mean(np.stack(test_probs_seeds, axis=0), axis=0)
    ensemble_metrics = evaluate_metrics(y_test.numpy(), ensemble_probs)
    ensemble_metrics.update({"model": "GraphSAGE (wild‑bird) Ensemble", "seed": "ensemble"})

    # Save per‑seed and ensemble metrics CSV
    df_metrics = pd.DataFrame(seed_results)
    df_metrics = pd.concat([df_metrics, pd.DataFrame([ensemble_metrics])], ignore_index=True)
    os.makedirs(os.path.dirname(OUT_METRICS), exist_ok=True)
    df_metrics.to_csv(OUT_METRICS, index=False)
    print(f"Saved metrics to {OUT_METRICS}")

    # Save test predictions (with per‑seed columns and ensemble)
    pred_df = test_index.copy()
    for i, seed in enumerate(SEEDS):
        pred_df[f"wildbird_gs_seed_{seed}_prob"] = test_probs_seeds[i]
    pred_df['wildbird_gs_ensemble_prob'] = ensemble_probs
    pred_df.to_csv(OUT_PRED, index=False)
    print(f"Saved test predictions to {OUT_PRED}")

    # -------------------------------------------------------------------
    # Bootstrap comparison against baseline GraphSAGE and wild‑bird LR
    # -------------------------------------------------------------------
    baseline_gs_path = "results/baselines/graphsage_v1_early_stopping_strict_test_predictions.csv"
    baseline_lr_path = "results/baselines/wildbird_lr_strict_test_predictions.csv"  # LR wild‑bird predictions we saved earlier
    df_baseline_gs = pd.read_csv(baseline_gs_path)
    df_baseline_lr = pd.read_csv(baseline_lr_path)
    # Align on csd_uid, year, month
    merge_cols = ["csd_uid", "year", "month"]
    df_eval = pred_df.merge(df_baseline_gs[merge_cols + [c for c in df_baseline_gs.columns if "strict_graphsage" in c]],
                            on=merge_cols, how="left")
    df_eval = df_eval.merge(df_baseline_lr[merge_cols + ["strict_env_lag_logistic_prob"]],
                            on=merge_cols, how="left")
    y = df_eval['actual'].values
    wb_probs = df_eval['wildbird_gs_ensemble_prob'].values
    bench_gs_probs = df_eval[[c for c in df_baseline_gs.columns if "strict_graphsage" in c]].mean(axis=1).values
    bench_lr_probs = df_eval['strict_env_lag_logistic_prob'].values

    # Bootstrap 1,000 resamples (CSD clusters)
    unique_csds = df_eval['csd_uid'].unique()
    csd_groups = df_eval.groupby('csd_uid').indices
    n_csds = len(unique_csds)
    np.random.seed(42)
    N_BOOT = 1000
    diffs_lr = []
    diffs_gs = []
    for _ in range(N_BOOT):
        resampled = np.random.choice(unique_csds, size=n_csds, replace=True)
        idx = np.concatenate([csd_groups[c] for c in resampled])
        y_b = y[idx]
        wb_b = wb_probs[idx]
        bench_lr_b = bench_lr_probs[idx]
        bench_gs_b = bench_gs_probs[idx]
        diffs_lr.append(average_precision_score(y_b, wb_b) - average_precision_score(y_b, bench_lr_b))
        diffs_gs.append(average_precision_score(y_b, wb_b) - average_precision_score(y_b, bench_gs_b))
    ci_lr = np.percentile(diffs_lr, [2.5, 97.5])
    ci_gs = np.percentile(diffs_gs, [2.5, 97.5])
    win_lr = np.mean(np.array(diffs_lr) > 0) * 100.0
    win_gs = np.mean(np.array(diffs_gs) > 0) * 100.0
    dec_lr = (
        "comparable, not proven better" if (ci_lr[0] <= 0 <= ci_lr[1])
        else ("statistically better" if ci_lr[0] > 0 else "statistically worse")
    )
    dec_gs = (
        "comparable, not proven better" if (ci_gs[0] <= 0 <= ci_gs[1])
        else ("statistically better" if ci_gs[0] > 0 else "statistically worse")
    )

    # -------------------------------------------------------------------
    # Markdown summary
    # -------------------------------------------------------------------
    md = f"""# Corrected Wild‑bird kNN‑3 GraphSAGE Model

**Training period**: 2022‑2024 (2024‑Q4 validation)\n**Test period**: 2025\n**Seeds**: {SEEDS}\n**Feature count**: 62\n\n## Per‑seed metrics\n"""
    for r in seed_results:
        md += f"- Seed {int(r['seed'])}: PR‑AUC {r['pr_auc']:.4f}, ROC‑AUC {r['roc_auc']:.4f}, P@50 {r['precision_at_50']:.4f}, R@50 {r['recall_at_50']:.4f}\n"
    md += f"\n## Ensemble metrics\n- PR‑AUC {ensemble_metrics['pr_auc']:.4f}\n- ROC‑AUC {ensemble_metrics['roc_auc']:.4f}\n- P@50 {ensemble_metrics['precision_at_50']:.4f}, R@50 {ensemble_metrics['recall_at_50']:.4f}\n\n## Prediction sanity checks (last seed)\n- Min prob: {prob_stats['min']:.6f}\n- Max prob: {prob_stats['max']:.6f}\n- Mean prob: {prob_stats['mean']:.6f}\n- Std dev: {prob_stats['std']:.6f}\n- Unique values: {prob_stats['unique']}\n\n## Bootstrap comparison vs baseline GraphSAGE (strict)\n- Mean diff: {np.mean(diffs_gs):+.4f}\n- 95 % CI: [{ci_gs[0]:+.4f}, {ci_gs[1]:+.4f}]\n- Resample >0 %: {win_gs:.1f}\n- Decision: **{dec_gs.upper()}**\n\n## Bootstrap comparison vs wild‑bird LR (strict)\n- Mean diff: {np.mean(diffs_lr):+.4f}\n- 95 % CI: [{ci_lr[0]:+.4f}, {ci_lr[1]:+.4f}]\n- Resample >0 %: {win_lr:.1f}\n- Decision: **{dec_lr.upper()}**\n\n## Artifacts\n- Metrics CSV: [{os.path.basename(OUT_METRICS)}](file:///{os.path.abspath(OUT_METRICS).replace('\\', '/')})\n- Test predictions CSV: [{os.path.basename(OUT_PRED)}](file:///{os.path.abspath(OUT_PRED).replace('\\', '/')})\n- Summary markdown (this file): [{os.path.basename(OUT_MD)}](file:///{os.path.abspath(OUT_MD).replace('\\', '/')})\n"""
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Saved markdown summary to {OUT_MD}")

if __name__ == '__main__':
    main()
