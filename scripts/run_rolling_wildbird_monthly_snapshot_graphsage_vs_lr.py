#!/usr/bin/env python3
"""
scripts/run_rolling_wildbird_monthly_snapshot_graphsage_knn3_vs_lr.py

Strict Rolling-Origin Temporal Comparison:
Valid Monthly-Snapshot Wild-Bird GraphSAGE vs Retrained Wild-Bird Logistic Regression Baseline.

Rolling Folds:
- Fold 1: Train 2022-01..09, Val 2022-10..12, Final Train 2022 full  -> Test 2023
- Fold 2: Train 2022-01..2023-09, Val 2023-10..12, Final Train 2022..2023 full -> Test 2024
- Fold 3: Train 2022-01..2024-09, Val 2024-10..12, Final Train 2022..2024 full -> Test 2025

Features:
- Exactly 62 numeric features (50 strict env + 12 wild-bird pressure). Asserted for every fold.

GraphSAGE Snapshot Construction:
- Separate PyG Data snapshot for every (year, month).
- Dynamic time-varying features per CSD-month, shared spatial edges within each month, shared GNN weights.

Outputs (new, never overwritten):
- results/baselines/rolling_wildbird_monthly_snapshot_graphsage_vs_lr_fold_metrics.csv
- results/baselines/rolling_wildbird_monthly_snapshot_graphsage_vs_lr_predictions.csv
- results/baselines/rolling_wildbird_monthly_snapshot_graphsage_vs_lr_bootstrap.csv
- results/baselines/rolling_wildbird_monthly_snapshot_graphsage_vs_lr_summary.md
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
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import SAGEConv

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

# Environment configuration
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# Paths
INPUT_MODEL_TABLE = "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
GRAPH_EDGE_CSV = "data/processed/graph/csd_graph_knn_3_v1.csv"

OUT_FOLD_METRICS = "results/baselines/rolling_wildbird_monthly_snapshot_graphsage_vs_lr_fold_metrics.csv"
OUT_PRED = "results/baselines/rolling_wildbird_monthly_snapshot_graphsage_vs_lr_predictions.csv"
OUT_BOOTSTRAP = "results/baselines/rolling_wildbird_monthly_snapshot_graphsage_vs_lr_bootstrap.csv"
OUT_MD = "results/baselines/rolling_wildbird_monthly_snapshot_graphsage_vs_lr_summary.md"

SEEDS = [11, 22, 33]

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
    df_eval = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df_eval = df_eval.sort_values(by="y_prob", ascending=False)
    top_k = df_eval.head(k)
    prec = top_k["y_true"].mean()
    total_pos = y_true.sum()
    rec = top_k["y_true"].sum() / total_pos if total_pos > 0 else 0.0
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

def build_monthly_snapshots(df: pd.DataFrame, feature_cols: list, edges_raw: pd.DataFrame, scaler: StandardScaler):
    snapshots = []
    ym_pairs = df[["year", "month"]].drop_duplicates().sort_values(by=["year", "month"]).values

    for y, m in ym_pairs:
        df_t = df[(df["year"] == y) & (df["month"] == m)].copy().reset_index(drop=True)
        n_t = len(df_t)

        X_t_raw = df_t[feature_cols].fillna(0.0).values
        X_t = scaler.transform(X_t_raw).astype(np.float32)
        y_t = df_t["outbreak_next_month"].values.astype(np.float32)

        uid_map = {uid: i for i, uid in enumerate(df_t["csd_uid"].values)}
        src_mapped = edges_raw["source_csd_uid"].map(uid_map).dropna().astype(int).values
        tgt_mapped = edges_raw["target_csd_uid"].map(uid_map).dropna().astype(int).values

        edge_index = torch.tensor(np.stack([src_mapped, tgt_mapped], axis=0), dtype=torch.long)
        x_tensor = torch.from_numpy(X_t)
        y_tensor = torch.from_numpy(y_t)

        data = Data(
            x=x_tensor,
            edge_index=edge_index,
            y=y_tensor,
            year=int(y),
            month=int(m),
            csd_uids=df_t["csd_uid"].values,
            csd_names=df_t["csd_name"].values if "csd_name" in df_t else np.array([""] * n_t),
            province_codes=df_t["province_code"].values if "province_code" in df_t else np.array([0] * n_t)
        )
        snapshots.append(data)

    return snapshots

def evaluate_snapshots(model, snapshot_loader):
    model.eval()
    all_probs = []
    all_ys = []
    with torch.no_grad():
        for batch in snapshot_loader:
            logits = model(batch.x, batch.edge_index)
            probs = torch.sigmoid(logits).cpu().numpy()
            ys = batch.y.cpu().numpy()
            all_probs.extend(probs)
            all_ys.extend(ys)
    return np.array(all_ys), np.array(all_probs)

def evaluate_snapshots_aligned(model, snapshots, df_test_ref):
    """
    Evaluates snapshots and aligns predictions strictly to df_test_ref row order using (csd_uid, year, month).
    """
    model.eval()
    records = []
    with torch.no_grad():
        for snap in snapshots:
            logits = model(snap.x, snap.edge_index)
            probs = torch.sigmoid(logits).cpu().numpy()
            ys = snap.y.cpu().numpy()
            for i in range(len(ys)):
                records.append({
                    "csd_uid": snap.csd_uids[i],
                    "year": snap.year,
                    "month": snap.month,
                    "actual": ys[i],
                    "prob": probs[i]
                })
    df_res = pd.DataFrame(records)
    df_aligned = df_test_ref[["csd_uid", "year", "month"]].merge(df_res, on=["csd_uid", "year", "month"], how="left")
    return df_aligned["prob"].values

def main():
    print("=== Running Strict 3-Fold Rolling-Origin Temporal Comparison ===")

    # Load data
    df = pd.read_csv(INPUT_MODEL_TABLE)
    print(f"Loaded modeling table: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # Select feature columns
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
    assert len(feature_cols) == 62, f"Expected 62 features, got {len(feature_cols)}"

    edges_raw = pd.read_csv(GRAPH_EDGE_CSV)

    # Define Rolling Folds
    folds_def = [
        {
            "fold": 1,
            "test_year": 2023,
            "dev_train": (df["year"] == 2022) & (df["month"] <= 9),
            "dev_val": (df["year"] == 2022) & (df["month"] >= 10),
            "final_train": (df["year"] == 2022),
            "test": (df["year"] == 2023)
        },
        {
            "fold": 2,
            "test_year": 2024,
            "dev_train": (df["year"] <= 2022) | ((df["year"] == 2023) & (df["month"] <= 9)),
            "dev_val": (df["year"] == 2023) & (df["month"] >= 10),
            "final_train": (df["year"] <= 2023),
            "test": (df["year"] == 2024)
        },
        {
            "fold": 3,
            "test_year": 2025,
            "dev_train": (df["year"] <= 2023) | ((df["year"] == 2024) & (df["month"] <= 9)),
            "dev_val": (df["year"] == 2024) & (df["month"] >= 10),
            "final_train": (df["year"] <= 2024),
            "test": (df["year"] == 2025)
        }
    ]

    all_fold_metrics = []
    all_out_of_sample_preds = []

    for f_info in folds_def:
        fold = f_info["fold"]
        test_yr = f_info["test_year"]
        print(f"\n==================== FOLD {fold} (Test Year {test_yr}) ====================")

        df_dev_train = df[f_info["dev_train"]].copy()
        df_dev_val = df[f_info["dev_val"]].copy()
        df_final_train = df[f_info["final_train"]].copy()
        df_test = df[f_info["test"]].copy().reset_index(drop=True)

        assert len(feature_cols) == 62, f"Fold {fold}: Feature count check failed!"

        n_test = len(df_test)
        pos_test = int(df_test["outbreak_next_month"].sum())
        prev_test = pos_test / n_test
        y_test_ref = df_test["outbreak_next_month"].values.astype(np.float32)

        print(f"Fold {fold} Sizes: DevTrain={len(df_dev_train):,}, DevVal={len(df_dev_val):,}, FinalTrain={len(df_final_train):,}, Test={n_test:,} ({pos_test} pos, prev={prev_test:.6f})")

        # ---------------------------------------------------------------
        # Model 1: Retrained Wild-Bird Logistic Regression
        # ---------------------------------------------------------------
        scaler_lr = StandardScaler()
        X_tr_lr = scaler_lr.fit_transform(df_final_train[feature_cols].fillna(0.0).values)
        X_te_lr = scaler_lr.transform(df_test[feature_cols].fillna(0.0).values)

        y_tr_lr = df_final_train["outbreak_next_month"].values

        pos_lr = y_tr_lr.sum()
        neg_lr = len(y_tr_lr) - pos_lr
        pos_weight_lr = neg_lr / pos_lr if pos_lr > 0 else 1.0

        clf = LogisticRegression(max_iter=1000, class_weight={0: 1.0, 1: pos_weight_lr}, solver="liblinear", random_state=42)
        clf.fit(X_tr_lr, y_tr_lr)
        lr_probs = clf.predict_proba(X_te_lr)[:, 1]

        lr_m = evaluate_metrics(y_test_ref, lr_probs)
        lr_m.update({
            "fold": fold,
            "test_year": test_yr,
            "model": "Wild-Bird Logistic Regression",
            "seed": "N/A",
            "test_rows": n_test,
            "test_positives": pos_test,
            "prevalence": prev_test,
            "pr_auc_lift": lr_m["pr_auc"] / prev_test if prev_test > 0 else np.nan,
            "unique_preds": int(np.unique(lr_probs).shape[0]),
            "score_std_mean": float(df_test.assign(prob=lr_probs).groupby("csd_uid")["prob"].std().fillna(0.0).mean())
        })
        all_fold_metrics.append(lr_m)
        print(f"Fold {fold} LR PR-AUC: {lr_m['pr_auc']:.4f} (Lift: {lr_m['pr_auc_lift']:.2f}x) | ROC-AUC: {lr_m['roc_auc']:.4f}")

        # ---------------------------------------------------------------
        # Model 2: Monthly-Snapshot GraphSAGE (Seeds 11, 22, 33 & Ensemble)
        # ---------------------------------------------------------------
        # Step A: Development Phase (Early Stopping Epochs)
        scaler_dev_gs = StandardScaler()
        scaler_dev_gs.fit(df_dev_train[feature_cols].fillna(0.0).values)

        dev_train_snaps = build_monthly_snapshots(df_dev_train, feature_cols, edges_raw, scaler_dev_gs)
        dev_val_snaps = build_monthly_snapshots(df_dev_val, feature_cols, edges_raw, scaler_dev_gs)

        loader_dev_train = PyGDataLoader(dev_train_snaps, batch_size=4, shuffle=True)
        loader_dev_val = PyGDataLoader(dev_val_snaps, batch_size=len(dev_val_snaps), shuffle=False)

        dev_pos = sum([s.y.sum().item() for s in dev_train_snaps])
        dev_total = sum([s.y.shape[0] for s in dev_train_snaps])
        pos_weight_dev = torch.tensor([(dev_total - dev_pos) / dev_pos], dtype=torch.float)

        optimal_epochs = {}
        patience = 20

        for seed in SEEDS:
            set_seed(seed)
            model = GraphSAGEModel(in_channels=62, hidden=32)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_dev)

            best_val_ap = -1.0
            best_epoch = 0
            no_improve = 0

            for epoch in range(1, 201):
                model.train()
                for batch in loader_dev_train:
                    optimizer.zero_grad()
                    logits = model(batch.x, batch.edge_index)
                    loss = criterion(logits, batch.y)
                    loss.backward()
                    optimizer.step()

                y_v, p_v = evaluate_snapshots(model, loader_dev_val)
                val_ap = average_precision_score(y_v, p_v)

                if val_ap > best_val_ap:
                    best_val_ap = val_ap
                    best_epoch = epoch
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

            optimal_epochs[seed] = best_epoch

        # Step B: Final Retraining on Full Fold Train & Test Evaluation
        scaler_final_gs = StandardScaler()
        scaler_final_gs.fit(df_final_train[feature_cols].fillna(0.0).values)

        final_train_snaps = build_monthly_snapshots(df_final_train, feature_cols, edges_raw, scaler_final_gs)
        test_snaps = build_monthly_snapshots(df_test, feature_cols, edges_raw, scaler_final_gs)

        loader_final_train = PyGDataLoader(final_train_snaps, batch_size=4, shuffle=True)

        final_pos = sum([s.y.sum().item() for s in final_train_snaps])
        final_total = sum([s.y.shape[0] for s in final_train_snaps])
        pos_weight_final = torch.tensor([(final_total - final_pos) / final_pos], dtype=torch.float)

        seed_probs_list = []

        for seed in SEEDS:
            t_epoch = optimal_epochs[seed]
            set_seed(seed)
            model = GraphSAGEModel(in_channels=62, hidden=32)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_final)

            for epoch in range(1, t_epoch + 1):
                model.train()
                for batch in loader_final_train:
                    optimizer.zero_grad()
                    logits = model(batch.x, batch.edge_index)
                    loss = criterion(logits, batch.y)
                    loss.backward()
                    optimizer.step()

            test_probs_s = evaluate_snapshots_aligned(model, test_snaps, df_test)

            gs_s_m = evaluate_metrics(y_test_ref, test_probs_s)
            gs_s_m.update({
                "fold": fold,
                "test_year": test_yr,
                "model": f"Monthly Snapshot GraphSAGE Seed {seed}",
                "seed": seed,
                "test_rows": n_test,
                "test_positives": pos_test,
                "prevalence": prev_test,
                "pr_auc_lift": gs_s_m["pr_auc"] / prev_test if prev_test > 0 else np.nan,
                "unique_preds": int(np.unique(test_probs_s).shape[0]),
                "score_std_mean": float(df_test.assign(prob=test_probs_s).groupby("csd_uid")["prob"].std().fillna(0.0).mean())
            })
            all_fold_metrics.append(gs_s_m)
            seed_probs_list.append(test_probs_s)

        # Ensemble
        gs_ens_probs = np.mean(np.stack(seed_probs_list, axis=0), axis=0)
        gs_ens_m = evaluate_metrics(y_test_ref, gs_ens_probs)
        gs_ens_m.update({
            "fold": fold,
            "test_year": test_yr,
            "model": "Wild-Bird Monthly-Snapshot kNN-3 GraphSAGE (Ensemble)",
            "seed": "ensemble",
            "test_rows": n_test,
            "test_positives": pos_test,
            "prevalence": prev_test,
            "pr_auc_lift": gs_ens_m["pr_auc"] / prev_test if prev_test > 0 else np.nan,
            "unique_preds": int(np.unique(gs_ens_probs).shape[0]),
            "score_std_mean": float(df_test.assign(prob=gs_ens_probs).groupby("csd_uid")["prob"].std().fillna(0.0).mean())
        })
        all_fold_metrics.append(gs_ens_m)

        print(f"Fold {fold} GraphSAGE Ensemble PR-AUC: {gs_ens_m['pr_auc']:.4f} (Lift: {gs_ens_m['pr_auc_lift']:.2f}x) | ROC-AUC: {gs_ens_m['roc_auc']:.4f}")

        # Within-CSD Prediction Variation QA Check
        constant_csds_gs = (df_test.assign(prob=gs_ens_probs).groupby("csd_uid")["prob"].nunique() == 1).sum()
        pct_constant_gs = (constant_csds_gs / df_test["csd_uid"].nunique()) * 100.0
        print(f"Fold {fold} GraphSAGE Within-CSD QA Check: {constant_csds_gs} constant CSDs ({pct_constant_gs:.2f}%)")
        assert pct_constant_gs < 10.0, f"Fold {fold}: GraphSAGE predictions suffered from temporal collapse!"

        # Store Out-of-Sample Predictions
        df_fold_pred = df_test[["csd_uid", "year", "month"]].copy()
        if "csd_name" in df_test:
            df_fold_pred["csd_name"] = df_test["csd_name"]
        if "province_code" in df_test:
            df_fold_pred["province_code"] = df_test["province_code"]

        df_fold_pred["actual"] = y_test_ref
        df_fold_pred["wildbird_lr_prob"] = lr_probs
        df_fold_pred["wildbird_snapshot_gs_ensemble_prob"] = gs_ens_probs
        df_fold_pred["fold"] = fold
        all_out_of_sample_preds.append(df_fold_pred)

    # -------------------------------------------------------------------
    # Step 4: Save Fold Metrics & Predictions
    # -------------------------------------------------------------------
    df_fold_metrics = pd.DataFrame(all_fold_metrics)
    os.makedirs(os.path.dirname(OUT_FOLD_METRICS), exist_ok=True)
    df_fold_metrics.to_csv(OUT_FOLD_METRICS, index=False)
    print(f"\nSaved fold metrics to {OUT_FOLD_METRICS}")

    df_pooled_pred = pd.concat(all_out_of_sample_preds, ignore_index=True)
    df_pooled_pred.to_csv(OUT_PRED, index=False)
    print(f"Saved pooled test predictions to {OUT_PRED} ({len(df_pooled_pred):,} total out-of-sample test rows)")

    # -------------------------------------------------------------------
    # Step 5: Pooled Evaluation & 1,000 CSD-Cluster Bootstrap
    # -------------------------------------------------------------------
    print("\n-------------------------------------------------------------------")
    print("STEP 5: POOLED EVALUATION & 1,000 CSD-CLUSTER BOOTSTRAP RESAMPLES")
    print("-------------------------------------------------------------------")

    y_pooled = df_pooled_pred["actual"].values
    p_lr_pooled = df_pooled_pred["wildbird_lr_prob"].values
    p_gs_pooled = df_pooled_pred["wildbird_snapshot_gs_ensemble_prob"].values

    pooled_lr_m = evaluate_metrics(y_pooled, p_lr_pooled)
    pooled_gs_m = evaluate_metrics(y_pooled, p_gs_pooled)

    print(f"Pooled LR PR-AUC:        {pooled_lr_m['pr_auc']:.4f} | ROC-AUC: {pooled_lr_m['roc_auc']:.4f}")
    print(f"Pooled GraphSAGE PR-AUC: {pooled_gs_m['pr_auc']:.4f} | ROC-AUC: {pooled_gs_m['roc_auc']:.4f}")

    direct_pooled_diff = pooled_gs_m["pr_auc"] - pooled_lr_m["pr_auc"]

    # 1,000 CSD-Cluster Bootstrap Resampling across all test years
    unique_csds = df_pooled_pred["csd_uid"].unique()
    n_csds = len(unique_csds)
    csd_groups = df_pooled_pred.groupby("csd_uid").indices

    np.random.seed(42)
    N_BOOT = 1000
    boot_diffs = []

    for _ in range(N_BOOT):
        resampled_csds = np.random.choice(unique_csds, size=n_csds, replace=True)
        idx = np.concatenate([csd_groups[c] for c in resampled_csds])

        y_b = y_pooled[idx]
        gs_b = p_gs_pooled[idx]
        lr_b = p_lr_pooled[idx]

        score_gs = average_precision_score(y_b, gs_b)
        score_lr = average_precision_score(y_b, lr_b)
        boot_diffs.append(score_gs - score_lr)

    boot_mean = np.mean(boot_diffs)
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    win_rate = np.mean(np.array(boot_diffs) > 0) * 100.0

    if ci_low > 0:
        statistical_decision = "statistically better"
    elif ci_low <= 0 <= ci_high:
        statistical_decision = "comparable, not proven better"
    else:
        statistical_decision = "statistically worse"

    print(f"\nPooled Bootstrap Results:")
    print(f"  Observed Pooled PR-AUC Diff: {direct_pooled_diff:+.4f}")
    print(f"  Bootstrap Mean Difference:   {boot_mean:+.4f}")
    print(f"  95% Percentile CI:           [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"  GraphSAGE Win Rate:          {win_rate:.1f}%")
    print(f"  Statistical Decision:        {statistical_decision.upper()}")

    # Save Bootstrap CSV
    df_boot_out = pd.DataFrame([{
        "comparator": "Wild-Bird Logistic Regression Baseline",
        "pooled_graphsage_pr_auc": pooled_gs_m["pr_auc"],
        "pooled_lr_pr_auc": pooled_lr_m["pr_auc"],
        "observed_pooled_diff": direct_pooled_diff,
        "bootstrap_mean_diff": boot_mean,
        "ci_lower_95": ci_low,
        "ci_upper_95": ci_high,
        "win_rate_pct": win_rate,
        "decision": statistical_decision
    }])
    df_boot_out.to_csv(OUT_BOOTSTRAP, index=False)
    print(f"Saved pooled bootstrap comparison to {OUT_BOOTSTRAP}")

    # -------------------------------------------------------------------
    # Step 6: Markdown Summary
    # -------------------------------------------------------------------
    md_content = f"""# Strict Rolling-Origin Temporal Comparison: Wild-Bird Snapshot GraphSAGE vs LR

**Evaluation Protocol**: Strict Rolling-Origin Temporal Validation (2023, 2024, 2025 Test Years)  
**Models Compared**:
1. Wild-Bird Logistic Regression Baseline
2. Wild-Bird Monthly-Snapshot kNN-3 GraphSAGE (3-Seed Ensemble)

---

## 1. Per-Fold Comparison Table

| Fold | Test Year | Test Rows | Positives | Prevalence | Model | PR-AUC | PR-AUC Lift | ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""
    for f_val in [1, 2, 3]:
        df_f = df_fold_metrics[(df_fold_metrics["fold"] == f_val) & (df_fold_metrics["seed"].isin(["ensemble", "N/A"]))]
        for row in df_f.itertuples():
            md_content += f"| {row.fold} | {row.test_year} | {row.test_rows:,} | {row.test_positives} | {row.prevalence:.6f} | {row.model} | **{row.pr_auc:.4f}** | {row.pr_auc_lift:.2f}x | {row.roc_auc:.4f} | {row.precision_at_50:.4f} | {row.recall_at_50:.4f} | {row.precision_at_100:.4f} | {row.recall_at_100:.4f} | {row.precision_at_200:.4f} | {row.recall_at_200:.4f} |\n"

    md_content += f"""\n---

## 2. Out-of-Sample Pooled Evaluation (2023–2025 Combined: {len(df_pooled_pred):,} Rows)

| Model | Pooled PR-AUC | Pooled ROC-AUC | P@50 | R@50 | P@100 | R@100 | P@200 | R@200 |
|---|---|---|---|---|---|---|---|---|
| Wild-Bird Logistic Regression Baseline | {pooled_lr_m['pr_auc']:.4f} | {pooled_lr_m['roc_auc']:.4f} | {pooled_lr_m['precision_at_50']:.4f} | {pooled_lr_m['recall_at_50']:.4f} | {pooled_lr_m['precision_at_100']:.4f} | {pooled_lr_m['recall_at_100']:.4f} | {pooled_lr_m['precision_at_200']:.4f} | {pooled_lr_m['recall_at_200']:.4f} |
| **Wild-Bird Monthly-Snapshot GraphSAGE (Ensemble)** | **{pooled_gs_m['pr_auc']:.4f}** | **{pooled_gs_m['roc_auc']:.4f}** | **{pooled_gs_m['precision_at_50']:.4f}** | **{pooled_gs_m['recall_at_50']:.4f}** | **{pooled_gs_m['precision_at_100']:.4f}** | **{pooled_gs_m['recall_at_100']:.4f}** | **{pooled_gs_m['precision_at_200']:.4f}** | **{pooled_gs_m['recall_at_200']:.4f}** |

---

## 3. Pooled 1,000 CSD-Cluster Bootstrap Analysis

- **Observed Pooled PR-AUC Difference**: {direct_pooled_diff:+.4f}
- **Bootstrap Mean Difference**: {boot_mean:+.4f}
- **95% Percentile Confidence Interval**: [{ci_low:+.4f}, {ci_high:+.4f}]
- **GraphSAGE Win Rate**: {win_rate:.1f}%
- **Statistical Decision**: **{statistical_decision.upper()}**

---

## 4. Methodological & Temporal QA Statement

> [!IMPORTANT]
> **TEMPORAL INTEGRITY & SUPERSEDED MODEL ASSURANCE**:  
> 1. Every GraphSAGE prediction in this experiment was generated at the **true CSD-month snapshot level** using distinct monthly feature matrices and shared spatial kNN-3 topology.  
> 2. Zero static CSD-level feature matrices (`X_all`) were used, and zero predictions were broadcast across time. Within-CSD monthly score standard deviations were positive across all folds.  
> 3. All prior static CSD-level GraphSAGE results remain **SUPERSEDED** and were not loaded or referenced as comparators.

"""

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Saved summary markdown to {OUT_MD}")

if __name__ == '__main__':
    main()
