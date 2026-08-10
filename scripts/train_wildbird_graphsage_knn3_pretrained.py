#!/usr/bin/env python3
"""
scripts/train_wildbird_graphsage_knn3_pretrained.py

Trains a leakage-safe self-supervised pretrained wild-bird-enhanced kNN-3 GraphSAGE model.

Goal:
Test whether masked node-feature reconstruction pretraining improves the GNN enough
to outperform the wild-bird Logistic Regression model, while keeping the comparison fair.

Input data:
- Data: data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv
- Graph: data/processed/graph/csd_graph_knn_3_v1.csv

Feature rules:
- 62 numeric features (50 strict environment + lag + 12 leakage-safe wild-bird HPAI pressure)
- Excludes identifiers, uids, targets, raw outbreak_current_month, metadata, eBird, GBIF, neighbor pressure.

Splits:
- Train: 2022-01 through 2024-09
- Validation: 2024-10 through 2024-12
- Test: 2025 (evaluated exactly once)

Graph representation:
- One node per CSD over a fixed kNN-3 spatial edge index.
- One snapshot per (year, month): node features are CSD-month values, so each
  month carries its own feature matrix rather than sharing a single collapsed one.

Pretraining Phase:
- Masked node-feature reconstruction (20% masking per epoch).
- GraphSAGE encoder + Linear decoder.
- Reconstruction MSE loss on masked features only.
- Runs over train and validation snapshots only; test months are never encoded.
- Validation reconstruction loss used for early stopping.

Fine-tuning Phase:
- Binary classification head attached to pretrained encoder.
- Fine-tuned with BCEWithLogitsLoss(pos_weight=...).
- Validation PR-AUC used for supervised early stopping.

Outputs (new, never overwritten):
- results/baselines/wildbird_graphsage_knn3_pretrained_multiseed_metrics.csv
- results/baselines/wildbird_graphsage_knn3_pretrained_test_predictions.csv
- results/baselines/wildbird_graphsage_knn3_pretrained_summary.csv
- results/baselines/wildbird_graphsage_knn3_pretrained_bootstrap_comparison.csv
- results/baselines/wildbird_graphsage_knn3_pretrained_summary.md
"""

import os
import random
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Environment configuration (single-threaded to avoid BLAS issues)
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

OUT_METRICS = "results/baselines/wildbird_graphsage_knn3_pretrained_multiseed_metrics.csv"
OUT_PRED = "results/baselines/wildbird_graphsage_knn3_pretrained_test_predictions.csv"
OUT_SUMM_CSV = "results/baselines/wildbird_graphsage_knn3_pretrained_summary.csv"
OUT_BOOTSTRAP = "results/baselines/wildbird_graphsage_knn3_pretrained_bootstrap_comparison.csv"
OUT_MD = "results/baselines/wildbird_graphsage_knn3_pretrained_summary.md"

BASE_CORRECTED_PRED = "results/baselines/wildbird_graphsage_knn3_corrected_test_predictions.csv"
BASE_LR_PRED = "results/baselines/wildbird_lr_strict_test_predictions.csv"

SEEDS = [11, 22, 33]
MASK_RATIO = 0.20

# ---------------------------------------------------------------------------
# Columns to exclude from features
# ---------------------------------------------------------------------------
EXCLUDE_EXPLICIT = {
    "csd_uid", "csd_name", "province_code", "province_name", "ccs_uid",
    "year", "month", "outbreak_next_month",
    "dw_fill_method", "climate_extraction_method",
    "glw2020_chicken_extraction_method", "glw2015_duck_extraction_method",
    "road_data_coverage", "outbreak_current_month",
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

# ---------------------------------------------------------------------------
# Models: Encoder, Decoder, Classifier
# ---------------------------------------------------------------------------
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

class ReconstructionDecoder(nn.Module):
    def __init__(self, hidden=32, out_channels=62):
        super().__init__()
        self.out = nn.Linear(hidden, out_channels)

    def forward(self, x):
        return self.out(x)

class PretrainedGraphSAGEClassifier(nn.Module):
    def __init__(self, encoder, hidden=32):
        super().__init__()
        self.encoder = encoder
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        return self.out(h).squeeze(-1)

def load_graph(edge_csv: str, node_ids: pd.Series):
    edges = pd.read_csv(edge_csv)
    uid_to_idx = {uid: i for i, uid in enumerate(node_ids)}
    src = edges["source_csd_uid"].map(uid_to_idx).dropna().astype(int).values
    tgt = edges["target_csd_uid"].map(uid_to_idx).dropna().astype(int).values
    edge_index = np.stack([src, tgt], axis=0)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index, uid_to_idx

def main():
    print("=== Self-Supervised Pretraining for Wild-Bird kNN-3 GraphSAGE ===")

    # -------------------------------------------------------------------
    # Load data & select features
    # -------------------------------------------------------------------
    df = pd.read_csv(INPUT_MODEL_TABLE)
    print(f"Loaded modeling table: {df.shape[0]:,} rows x {df.shape[1]} cols")

    feature_cols = []
    for c in df.columns:
        if c in EXCLUDE_EXPLICIT:
            continue
        if any(sub in c.lower() for sub in ["ebird", "gbif", "neighbor", "pressure"]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)
    feature_cols = sorted(feature_cols)

    print(f"Selected feature count: {len(feature_cols)}")
    print("Features:", feature_cols)
    assert len(feature_cols) == 62, f"Expected exactly 62 features, got {len(feature_cols)}"

    # -------------------------------------------------------------------
    # Splits (Development: train <= 2024-09, val = 2024-10..12, test = 2025)
    # -------------------------------------------------------------------
    df_train = df[(df['year'] <= 2023) | ((df['year'] == 2024) & (df['month'] <= 9))].copy()
    df_val = df[(df['year'] == 2024) & (df['month'] >= 10)].copy()
    df_test = df[df['year'] == 2025].copy()

    print(f"Split sizes: Train={len(df_train):,}, Val={len(df_val):,}, Test={len(df_test):,}")

    # Build Graph Node Indexing
    node_ids = pd.concat([df_train['csd_uid'], df_val['csd_uid'], df_test['csd_uid']]).unique()
    edge_index, uid_to_idx = load_graph(GRAPH_EDGE_CSV, pd.Series(node_ids))
    n_nodes = len(uid_to_idx)
    print(f"Graph: {n_nodes} nodes, {edge_index.shape[1]} edges")

    # Standardize features (fit on train only)
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

    # Pos weight for fine-tuning
    pos = y_train.sum().item()
    neg = y_train.shape[0] - pos
    pos_weight = torch.tensor([neg / pos], dtype=torch.float)

    seed_results = []
    test_probs_seeds = []
    patience = 20

    for seed in SEEDS:
        print(f"\n==================== Seed {seed} ====================")
        set_seed(seed)

        # ---------------------------------------------------------------
        # Phase 1: Self-Supervised Pretraining
        # ---------------------------------------------------------------
        print("\n--- Phase 1: Pretraining Masked Feature Reconstruction ---")
        encoder = GraphSAGEEncoder(in_channels=len(feature_cols), hidden=32)
        decoder = ReconstructionDecoder(hidden=32, out_channels=len(feature_cols))

        optimizer_pretrain = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=0.01, weight_decay=5e-4
        )

        best_val_loss = float("inf")
        best_encoder_state = None
        best_pretrain_epoch = 0
        no_improve_pretrain = 0

        def reconstruction_loss(snapshot):
            """Mask 20% of observed feature entries and reconstruct them."""
            m = torch.rand_like(snapshot['x']) < MASK_RATIO
            m &= snapshot['mask'].unsqueeze(1)  # never score unobserved nodes
            if m.sum() == 0:
                return None
            x_masked = snapshot['x'].clone()
            x_masked[m] = 0.0
            recon = decoder(encoder(x_masked, edge_index))
            return F.mse_loss(recon[m], snapshot['x'][m])

        for epoch in range(1, 201):
            encoder.train()
            decoder.train()

            # Pretraining sees train and val snapshots only; test months are never
            # encoded, so the self-supervised phase carries no future information.
            train_losses = [reconstruction_loss(s) for s in train_snaps]
            train_losses = [l for l in train_losses if l is not None]
            if not train_losses:
                continue
            loss_train = torch.stack(train_losses).mean()

            optimizer_pretrain.zero_grad()
            loss_train.backward()
            optimizer_pretrain.step()

            # Validation reconstruction loss
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                val_losses = [reconstruction_loss(s) for s in val_snaps]
                val_losses = [l for l in val_losses if l is not None]
                val_loss = (
                    torch.stack(val_losses).mean().item() if val_losses
                    else loss_train.item()
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_encoder_state = copy.deepcopy(encoder.state_dict())
                best_pretrain_epoch = epoch
                no_improve_pretrain = 0
            else:
                no_improve_pretrain += 1
                if no_improve_pretrain >= patience:
                    print(f"Pretraining early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.6f} at epoch {best_pretrain_epoch}")
                    break

        print(f"Pretraining complete. Selected Pretrain Epoch: {best_pretrain_epoch}, Best Val Loss: {best_val_loss:.6f}")

        # ---------------------------------------------------------------
        # Phase 2: Supervised Fine-Tuning
        # ---------------------------------------------------------------
        print("\n--- Phase 2: Supervised Fine-Tuning ---")
        encoder.load_state_dict(best_encoder_state)
        model = PretrainedGraphSAGEClassifier(encoder=encoder, hidden=32)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer_finetune = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        best_val_ap = -1.0
        best_model_state = None
        best_finetune_epoch = 0
        no_improve_finetune = 0

        for epoch in range(1, 201):
            model.train()
            optimizer_finetune.zero_grad()
            # One forward pass per training month; loss averaged over snapshots.
            losses = [
                criterion(model(s['x'], edge_index)[s['mask']], s['y'][s['mask']])
                for s in train_snaps
            ]
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer_finetune.step()

            # Validation PR-AUC
            val_probs = predict(model, val_snaps)
            val_ap = average_precision_score(y_val.numpy(), val_probs)

            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_model_state = copy.deepcopy(model.state_dict())
                best_finetune_epoch = epoch
                no_improve_finetune = 0
            else:
                no_improve_finetune += 1
                if no_improve_finetune >= patience:
                    print(f"Fine-tuning early stopping at epoch {epoch}. Best Val PR-AUC: {best_val_ap:.4f} at epoch {best_finetune_epoch}")
                    break

        # Load best fine-tuned model
        model.load_state_dict(best_model_state)

        # Evaluate on test set (2025)
        test_probs = predict(model, test_snaps)

        # Prediction Sanity Checks
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
        print(f"Seed {seed} Prediction Stats:", prob_stats)
        assert prob_stats["std"] > 1e-6, f"Seed {seed}: Predicted probabilities collapsed! Std is {prob_stats['std']}"
        # A model that ignores the month would emit one score per CSD for the whole
        # test year, which is what the previous collapsed feature matrix produced.
        assert prob_stats["mean_within_csd_std"] > 1e-6, (
            f"Seed {seed}: predictions do not vary across months within a CSD; "
            "node features are not month-specific"
        )

        top20_idx = np.argsort(-test_probs)[:20]
        top20 = test_index.iloc[top20_idx][["csd_uid", "year", "month"]].copy()
        top20["prob"] = test_probs[top20_idx]
        top20["actual"] = y_test.numpy()[top20_idx]
        print(f"Seed {seed} Top-20 predictions:")
        print(top20.to_string(index=False))

        metrics = evaluate_metrics(y_test.numpy(), test_probs)
        metrics.update({
            "seed": seed,
            "pretrain_epoch": best_pretrain_epoch,
            "finetune_epoch": best_finetune_epoch,
            "val_pr_auc": best_val_ap,
            "model": f"GraphSAGE Pretrained Seed {seed}",
            **prob_stats
        })
        seed_results.append(metrics)
        test_probs_seeds.append(test_probs)

    # -------------------------------------------------------------------
    # Aggregate Multi-Seed Ensemble Results
    # -------------------------------------------------------------------
    ensemble_probs = np.mean(np.stack(test_probs_seeds, axis=0), axis=0)
    ensemble_metrics = evaluate_metrics(y_test.numpy(), ensemble_probs)
    ensemble_metrics.update({
        "model": "Wild-Bird kNN-3 GraphSAGE (Pretrained Ensemble)",
        "seed": "ensemble",
        "pretrain_epoch": int(np.mean([r["pretrain_epoch"] for r in seed_results])),
        "finetune_epoch": int(np.mean([r["finetune_epoch"] for r in seed_results])),
        "val_pr_auc": float(np.mean([r["val_pr_auc"] for r in seed_results])),
        "min": float(ensemble_probs.min()),
        "max": float(ensemble_probs.max()),
        "mean": float(ensemble_probs.mean()),
        "std": float(ensemble_probs.std()),
        "unique": int(np.unique(ensemble_probs).shape[0])
    })

    # Save Multi-seed Metrics CSV
    df_metrics = pd.DataFrame(seed_results)
    df_metrics = pd.concat([df_metrics, pd.DataFrame([ensemble_metrics])], ignore_index=True)
    os.makedirs(os.path.dirname(OUT_METRICS), exist_ok=True)
    df_metrics.to_csv(OUT_METRICS, index=False)
    print(f"\nSaved multiseed metrics to {OUT_METRICS}")

    # Save Test Predictions CSV
    pred_df = test_index.copy()
    for i, seed in enumerate(SEEDS):
        pred_df[f"wildbird_gs_pretrained_seed_{seed}_prob"] = test_probs_seeds[i]
    pred_df['wildbird_gs_pretrained_ensemble_prob'] = ensemble_probs
    pred_df.to_csv(OUT_PRED, index=False)
    print(f"Saved test predictions to {OUT_PRED}")

    # -------------------------------------------------------------------
    # Bootstrap Comparison against Non-Pretrained GraphSAGE & Wild-bird LR
    # -------------------------------------------------------------------
    print("\n--- Running 1,000 CSD-Cluster Bootstrap Resamples ---")
    df_corrected = pd.read_csv(BASE_CORRECTED_PRED)
    df_lr = pd.read_csv(BASE_LR_PRED)

    merge_cols = ["csd_uid", "year", "month"]
    df_eval = pred_df.merge(df_corrected[merge_cols + ["wildbird_gs_ensemble_prob"]], on=merge_cols, how="left")
    df_eval = df_eval.merge(df_lr[merge_cols + ["strict_env_lag_logistic_prob"]], on=merge_cols, how="left")

    y_eval = df_eval['actual'].values
    pretrained_probs = df_eval['wildbird_gs_pretrained_ensemble_prob'].values
    corrected_gs_probs = df_eval['wildbird_gs_ensemble_prob'].values
    lr_probs = df_eval['strict_env_lag_logistic_prob'].values

    unique_csds = df_eval['csd_uid'].unique()
    csd_groups = df_eval.groupby('csd_uid').indices
    n_csds = len(unique_csds)

    np.random.seed(42)
    N_BOOT = 1000
    diffs_gs = []
    diffs_lr = []

    for _ in range(N_BOOT):
        resampled_csds = np.random.choice(unique_csds, size=n_csds, replace=True)
        idx = np.concatenate([csd_groups[c] for c in resampled_csds])

        y_b = y_eval[idx]
        pre_b = pretrained_probs[idx]
        gs_b = corrected_gs_probs[idx]
        lr_b = lr_probs[idx]

        score_pre = average_precision_score(y_b, pre_b)
        score_gs = average_precision_score(y_b, gs_b)
        score_lr = average_precision_score(y_b, lr_b)

        diffs_gs.append(score_pre - score_gs)
        diffs_lr.append(score_pre - score_lr)

    def get_bootstrap_summary(diffs, comp_name, comp_score):
        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
        win_rate = np.mean(np.array(diffs) > 0) * 100.0
        mean_diff = np.mean(diffs)

        if ci_low > 0:
            decision = "statistically better"
        elif ci_low <= 0 <= ci_high:
            decision = "comparable, not proven better"
        else:
            decision = "statistically worse"

        return {
            "comparator": comp_name,
            "pretrained_ensemble_pr_auc": ensemble_metrics["pr_auc"],
            "comparator_pr_auc": comp_score,
            "mean_pr_auc_diff": mean_diff,
            "ci_lower_95": ci_low,
            "ci_upper_95": ci_high,
            "win_rate_pct": win_rate,
            "decision": decision
        }

    corr_gs_score = average_precision_score(y_eval, corrected_gs_probs)
    lr_score = average_precision_score(y_eval, lr_probs)

    boot_gs = get_bootstrap_summary(diffs_gs, "Corrected Wild-Bird kNN-3 GraphSAGE (No Pretraining)", corr_gs_score)
    boot_lr = get_bootstrap_summary(diffs_lr, "Wild-Bird Logistic Regression Baseline", lr_score)

    df_boot = pd.DataFrame([boot_gs, boot_lr])
    df_boot.to_csv(OUT_BOOTSTRAP, index=False)
    print(f"Saved bootstrap comparison to {OUT_BOOTSTRAP}")

    # Summary CSV
    df_summ = pd.DataFrame([
        {
            "model": "Wild-Bird Logistic Regression",
            "pr_auc": lr_score,
            "roc_auc": roc_auc_score(y_eval, lr_probs),
            "precision_at_50": precision_recall_at_k(y_eval, lr_probs, 50)[0],
            "recall_at_50": precision_recall_at_k(y_eval, lr_probs, 50)[1]
        },
        {
            "model": "Corrected Wild-Bird kNN-3 GraphSAGE (No Pretraining)",
            "pr_auc": corr_gs_score,
            "roc_auc": roc_auc_score(y_eval, corrected_gs_probs),
            "precision_at_50": precision_recall_at_k(y_eval, corrected_gs_probs, 50)[0],
            "recall_at_50": precision_recall_at_k(y_eval, corrected_gs_probs, 50)[1]
        },
        {
            "model": "Wild-Bird kNN-3 GraphSAGE (With Pretraining)",
            "pr_auc": ensemble_metrics["pr_auc"],
            "roc_auc": ensemble_metrics["roc_auc"],
            "precision_at_50": ensemble_metrics["precision_at_50"],
            "recall_at_50": ensemble_metrics["recall_at_50"]
        }
    ])
    df_summ.to_csv(OUT_SUMM_CSV, index=False)
    print(f"Saved summary CSV to {OUT_SUMM_CSV}")

    # Markdown Summary
    md_content = f"""# Leakage-Safe Self-Supervised Pretrained Wild-Bird kNN-3 GraphSAGE Summary

**Pretraining Task**: Masked Node-Feature Reconstruction (20% masking per pass)  
**Supervised Task**: HPAI Outbreak Next Month Prediction  
**Training Period**: 2022-01 through 2024-09  
**Validation Period**: 2024-10 through 2024-12  
**Test Period**: 2025 (Held-out single evaluation)  
**Feature Count**: 62  

## Multi-Seed Results

| Seed | Pretrain Epoch | Finetune Epoch | Val PR-AUC | Test PR-AUC | Test ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
|---|---|---|---|---|---|---|---|---|---|---|---|
"""
    for r in seed_results:
        md_content += f"| {r['seed']} | {r['pretrain_epoch']} | {r['finetune_epoch']} | {r['val_pr_auc']:.4f} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['precision_at_50']:.4f} | {r['recall_at_50']:.4f} | {r['precision_at_100']:.4f} | {r['recall_at_100']:.4f} | {r['precision_at_200']:.4f} | {r['recall_at_200']:.4f} |\n"

    md_content += f"| **Ensemble** | {ensemble_metrics['pretrain_epoch']} | {ensemble_metrics['finetune_epoch']} | {ensemble_metrics['val_pr_auc']:.4f} | **{ensemble_metrics['pr_auc']:.4f}** | **{ensemble_metrics['roc_auc']:.4f}** | **{ensemble_metrics['precision_at_50']:.4f}** | **{ensemble_metrics['recall_at_50']:.4f}** | **{ensemble_metrics['precision_at_100']:.4f}** | **{ensemble_metrics['recall_at_100']:.4f}** | **{ensemble_metrics['precision_at_200']:.4f}** | **{ensemble_metrics['recall_at_200']:.4f}** |\n\n"

    md_content += f"""## Prediction Sanity Checks (Ensemble)
- Min Prob: {ensemble_metrics['min']:.6f}
- Max Prob: {ensemble_metrics['max']:.6f}
- Mean Prob: {ensemble_metrics['mean']:.6f}
- Std Dev: {ensemble_metrics['std']:.6f}
- Unique Predictions: {ensemble_metrics['unique']}

## Bootstrap Comparison (1,000 Resamples)

### vs. Corrected Wild-Bird kNN-3 GraphSAGE (No Pretraining)
- Baseline PR-AUC: {boot_gs['comparator_pr_auc']:.4f}
- Pretrained PR-AUC: {boot_gs['pretrained_ensemble_pr_auc']:.4f}
- Observed Mean Diff: {boot_gs['mean_pr_auc_diff']:+.4f}
- 95% Percentile CI: [{boot_gs['ci_lower_95']:+.4f}, {boot_gs['ci_upper_95']:+.4f}]
- Pretrained Win Rate: {boot_gs['win_rate_pct']:.1f}%
- Decision: **{boot_gs['decision'].upper()}**

### vs. Wild-Bird Logistic Regression
- Baseline PR-AUC: {boot_lr['comparator_pr_auc']:.4f}
- Pretrained PR-AUC: {boot_lr['pretrained_ensemble_pr_auc']:.4f}
- Observed Mean Diff: {boot_lr['mean_pr_auc_diff']:+.4f}
- 95% Percentile CI: [{boot_lr['ci_lower_95']:+.4f}, {boot_lr['ci_upper_95']:+.4f}]
- Pretrained Win Rate: {boot_lr['win_rate_pct']:.1f}%
- Decision: **{boot_lr['decision'].upper()}**

## Model Comparison

| Model | Test PR-AUC | Test ROC-AUC | P@50 | R@50 |
|---|---|---|---|---|
| Wild-Bird Logistic Regression | {lr_score:.4f} | {roc_auc_score(y_eval, lr_probs):.4f} | {precision_recall_at_k(y_eval, lr_probs, 50)[0]:.4f} | {precision_recall_at_k(y_eval, lr_probs, 50)[1]:.4f} |
| Corrected Wild-Bird kNN-3 GraphSAGE (No Pretraining) | {corr_gs_score:.4f} | {roc_auc_score(y_eval, corrected_gs_probs):.4f} | {precision_recall_at_k(y_eval, corrected_gs_probs, 50)[0]:.4f} | {precision_recall_at_k(y_eval, corrected_gs_probs, 50)[1]:.4f} |
| **Wild-Bird kNN-3 GraphSAGE (With Pretraining)** | **{ensemble_metrics['pr_auc']:.4f}** | **{ensemble_metrics['roc_auc']:.4f}** | **{ensemble_metrics['precision_at_50']:.4f}** | **{ensemble_metrics['recall_at_50']:.4f}** |

"""

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Saved summary markdown to {OUT_MD}")

    print("\n=======================================================================")
    print("FINAL MODEL COMPARISON TABLE")
    print("=======================================================================")
    print(df_summ.to_string(index=False))
    print("=======================================================================")

if __name__ == '__main__':
    main()
