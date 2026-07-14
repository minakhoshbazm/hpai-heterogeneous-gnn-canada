# scripts/train_time_persistence_baseline.py
"""Train a simple time‑aware persistence baseline for one‑month‑ahead outbreak risk.

This script is lightweight and does not perform any hyper‑parameter tuning. It:
- Loads the prepared modeling table.
- Splits the data into a temporal train (2022‑2024) and test (2025) set.
- Computes three hierarchical risk estimates:
  1️⃣ CSD‑specific historical mean risk.
  2️⃣ Province‑month seasonal mean risk (fallback).
  3️⃣ Global positive rate (final fallback).
- Generates predictions for the 2025 test rows using the hierarchy above.
- Calculates evaluation metrics:
  * Average precision (PR‑AUC)
  * ROC‑AUC
  * Precision@K / Recall@K for K = 50, 100, 200
- Saves metrics and predictions to `results/baselines/`.
- Prints a concise report and the current `git status`.

Usage:
    python scripts/train_time_persistence_baseline.py [input_csv]
If the input path is omitted, the default
`data/processed/modeling/model_table_v1_environment_train_ready.csv` is used.
"""

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int):
    """Return precision and recall for the top‑k predictions.

    Parameters
    ----------
    y_true: array‑like of binary labels (0/1)
    y_score: array‑like of prediction scores
    k: int – number of top predictions to consider
    """
    if k <= 0:
        return 0.0, 0.0
    # sort indices by descending score
    order = np.argsort(-y_score)
    top_k = order[:k]
    true_in_top_k = y_true[top_k].sum()
    precision = true_in_top_k / k
    recall = true_in_top_k / y_true.sum() if y_true.sum() > 0 else 0.0
    return precision, recall


def run_git_status(repo_root: Path):
    """Execute `git status --short` and return its stdout as a string."""
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Git status failed: {e}"

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main():
    # ---------------------------------------------------------------------
    # Paths & arguments
    # ---------------------------------------------------------------------
    default_input = "data/processed/modeling/model_table_v1_environment_train_ready.csv"
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(default_input)
    repo_root = Path(__file__).resolve().parents[2]  # project root (two levels up from scripts)

    # ---------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------
    if not input_path.is_file():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(input_path, dtype={"csd_uid": str, "province_code": str})

    # Expected columns (minimal set)
    required_cols = {"csd_uid", "province_code", "year", "month", "outbreak_next_month"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERROR] Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------------------
    # Temporal split
    # ---------------------------------------------------------------------
    train_years = [2022, 2023, 2024]
    test_year = 2025
    train_df = df[df["year"].isin(train_years)].copy()
    test_df = df[df["year"] == test_year].copy()

    # ---------------------------------------------------------------------
    # Basic counts
    # ---------------------------------------------------------------------
    train_rows = len(train_df)
    test_rows = len(test_df)
    train_pos = int(train_df["outbreak_next_month"].sum())
    test_pos = int(test_df["outbreak_next_month"].sum())

    # ---------------------------------------------------------------------
    # Hierarchical risk estimation
    # ---------------------------------------------------------------------
    # 1. CSD‑level historical mean
    csd_risk = train_df.groupby("csd_uid")["outbreak_next_month"].mean()
    # 2. Province‑month seasonal mean
    prov_month_risk = train_df.groupby(["province_code", "month"])["outbreak_next_month"].mean()
    # 3. Global rate
    global_rate = train_df["outbreak_next_month"].mean()

    # Apply hierarchy to test set
    test_df["pred"] = test_df["csd_uid"].map(csd_risk)
    missing_mask = test_df["pred"].isna()
    if missing_mask.any():
        # map province‑month risk for rows where CSD risk is missing
        prov_month_keys = list(zip(test_df.loc[missing_mask, "province_code"], test_df.loc[missing_mask, "month"]))
        test_df.loc[missing_mask, "pred"] = [prov_month_risk.get(k, np.nan) for k in prov_month_keys]
    # Final fallback to global rate
    test_df["pred"] = test_df["pred"].fillna(global_rate)

    # ---------------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------------
    y_true = test_df["outbreak_next_month"].values
    y_score = test_df["pred"].values

    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    metrics = {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_positives": train_pos,
        "test_positives": test_pos,
        "average_precision": pr_auc,
        "roc_auc": roc_auc,
    }

    for k in (50, 100, 200):
        prec, rec = precision_recall_at_k(y_true, y_score, k)
        metrics[f"precision_at_{k}"] = prec
        metrics[f"recall_at_{k}"] = rec

    # ---------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------
    out_dir = Path("results/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "time_persistence_baseline_metrics.csv"
    pd.Series(metrics).to_csv(metrics_path, header=["value"])

    preds_path = out_dir / "time_persistence_test_predictions.csv"
    # keep a few useful columns for downstream inspection
    test_df[["csd_uid", "province_code", "year", "month", "outbreak_next_month", "pred"]].to_csv(preds_path, index=False)

    # ---------------------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------------------
    print("--- Training / Test Summary ---")
    print(f"Train rows: {train_rows}, positives: {train_pos}")
    print(f"Test rows: {test_rows}, positives: {test_pos}\n")
    print("--- Metrics ---")
    print(f"Average precision (PR-AUC): {pr_auc:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")
    for k in (50, 100, 200):
        print(f"Precision@{k}: {metrics[f'precision_at_{k}']:.6f}, Recall@{k}: {metrics[f'recall_at_{k}']:.6f}")
    print("\nTop 20 predictions (sorted by score descending):")
    top20 = test_df.sort_values("pred", ascending=False).head(20)
    print(top20[["csd_uid", "province_code", "year", "month", "pred", "outbreak_next_month"]].to_string(index=False))

    print("\nMetrics saved to:", metrics_path)
    print("Predictions saved to:", preds_path)

    # ---------------------------------------------------------------------
    # Git status
    # ---------------------------------------------------------------------
    print("\n--- Git status ---")
    print(run_git_status(repo_root))


if __name__ == "__main__":
    main()
