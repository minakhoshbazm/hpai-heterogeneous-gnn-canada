"""Audit the leakage boundaries and baseline inputs for the GraphSAGE plan."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data/processed/modeling/model_table_v1_environment_with_lags_wildbird.csv"
GRAPH = ROOT / "data/processed/graph/csd_graph_knn_3_v1.csv"
OUT = ROOT / "results/audits/graphsage_improvement_pipeline_audit.json"

ID_COLS = {"csd_uid", "csd_name", "province_code", "province_name", "ccs_uid"}
TARGET_COLS = {"outbreak_next_month"}
TIME_COLS = {"month", "year"}
NON_FEATURE_COLS = ID_COLS | TARGET_COLS | TIME_COLS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    df = pd.read_csv(TABLE)
    graph = pd.read_csv(GRAPH, dtype=str)
    numeric = df.select_dtypes(include="number").columns.tolist()
    features = [c for c in numeric if c not in NON_FEATURE_COLS]
    suspicious = [
        c for c in features
        if any(token in c.lower() for token in ("next", "future", "test", "label"))
    ]
    periods = {
        "train": {"years": [2022, 2023]},
        "validation": {"years": [2024]},
        "locked_test": {"years": [2025]},
    }
    split_counts = {}
    for name, spec in periods.items():
        subset = df[df["year"].isin(spec["years"])]
        split_counts[name] = {
            "rows": int(len(subset)),
            "csds": int(subset["csd_uid"].nunique()),
            "positives": int((subset["outbreak_next_month"] > 0).sum()),
            "negatives": int((subset["outbreak_next_month"] <= 0).sum()),
            "months": sorted(subset["month"].unique().tolist()),
        }
    payload = {
        "dataset": str(TABLE.relative_to(ROOT)),
        "dataset_sha256": sha256(TABLE),
        "graph": str(GRAPH.relative_to(ROOT)),
        "graph_sha256": sha256(GRAPH),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "numeric_feature_count": len(features),
        "numeric_feature_names": features,
        "suspicious_feature_names": suspicious,
        "target": "outbreak_next_month",
        "split_policy": periods,
        "split_counts": split_counts,
        "graph_edges": int(len(graph)),
        "graph_source_columns": graph.columns.tolist(),
        "baseline_reference_files": [
            "results/baselines/graphsage_v1_early_stopping_strict_multiseed_metrics.csv",
            "results/baselines/graphsage_v1_early_stopping_strict_test_predictions.csv",
        ],
        "audit_status": "PASS_WITH_REVIEW" if not suspicious else "REVIEW_REQUIRED",
        "notes": [
            "This audit does not use 2025 labels for fitting or model selection.",
            "Verify scaler fitting and exact baseline hyperparameters from the training script before Group A.",
            "Do not start Groups B-I until suspicious features and baseline artifact lineage are reviewed.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

