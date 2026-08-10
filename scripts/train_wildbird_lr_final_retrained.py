#!/usr/bin/env python3
"""
scripts/train_wildbird_lr_final_retrained.py

Trains and evaluates the final wild-bird Logistic Regression baseline.

Splits:
- Training period: 2022-01 through 2024-12 (all pre-2025 data)
- Test period: 2025 (held-out single evaluation)

Features:
- Exactly 62 numeric features (50 strict env + 12 wild-bird pressure)

Model:
- StandardScaler fit on 2022-2024 training period
- LogisticRegression with positive class weighting fit on 2022-2024 training period

Outputs (new, never overwritten):
- results/baselines/wildbird_lr_final_retrained_metrics.csv
- results/baselines/wildbird_lr_final_retrained_test_predictions.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

INPUT_MODEL_TABLE = "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"

OUT_METRICS = "results/baselines/wildbird_lr_final_retrained_metrics.csv"
OUT_PRED = "results/baselines/wildbird_lr_final_retrained_test_predictions.csv"

EXCLUDE_EXPLICIT = {
    "csd_uid", "csd_name", "province_code", "province_name", "ccs_uid",
    "year", "month", "outbreak_next_month",
    "dw_fill_method", "climate_extraction_method",
    "glw2020_chicken_extraction_method", "glw2015_duck_extraction_method",
    "road_data_coverage", "outbreak_current_month",
}

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

def main():
    print("=== Training Final Retrained Wild-Bird Logistic Regression Baseline ===")

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

    print(f"Selected {len(feature_cols)} feature columns")
    assert len(feature_cols) == 62, f"Expected 62 features, got {len(feature_cols)}"

    # Final Train (2022-2024) and Test (2025) splits
    df_train = df[df["year"] <= 2024].copy()
    df_test = df[df["year"] == 2025].copy()

    print(f"Train size (2022-2024): {len(df_train):,} rows, Test size (2025): {len(df_test):,} rows")

    # Fit scaler on 2022-2024 train set only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].fillna(0.0))
    X_test = scaler.transform(df_test[feature_cols].fillna(0.0))

    y_train = df_train["outbreak_next_month"].values
    y_test = df_test["outbreak_next_month"].values

    # Class weighting
    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = neg / pos if pos > 0 else 1.0

    print(f"Train positives: {pos}, negatives: {neg}, calculated pos_weight: {pos_weight:.4f}")

    clf = LogisticRegression(max_iter=1000, class_weight={0: 1.0, 1: pos_weight}, solver="liblinear", random_state=42)
    clf.fit(X_train, y_train)

    test_probs = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate_metrics(y_test, test_probs)
    print("\nFinal Retrained LR Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # Save metrics CSV
    df_metrics = pd.DataFrame([{
        "model": "Final Retrained Wild-Bird Logistic Regression",
        **metrics
    }])
    os.makedirs(os.path.dirname(OUT_METRICS), exist_ok=True)
    df_metrics.to_csv(OUT_METRICS, index=False)
    print(f"\nSaved metrics to {OUT_METRICS}")

    # Save predictions CSV
    df_pred = df_test[["csd_uid", "year", "month"]].copy()
    if "csd_name" in df_test:
        df_pred["csd_name"] = df_test["csd_name"]
    if "province_code" in df_test:
        df_pred["province_code"] = df_test["province_code"]
    df_pred["actual"] = y_test
    df_pred["wildbird_lr_final_prob"] = test_probs

    df_pred.to_csv(OUT_PRED, index=False)
    print(f"Saved test predictions to {OUT_PRED}")

if __name__ == "__main__":
    main()
