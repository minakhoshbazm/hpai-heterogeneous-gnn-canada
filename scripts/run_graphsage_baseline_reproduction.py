"""Five-seed, non-overwriting reproduction of the audited kNN-3 GraphSAGE baseline."""

import train_wildbird_graphsage_knn3_corrected as baseline


baseline.SEEDS = [11, 22, 33, 44, 55]
baseline.OUT_METRICS = (
    "results/baselines/graphsage_knn3_baseline_reproduction_metrics.csv"
)
baseline.OUT_PRED = (
    "results/baselines/graphsage_knn3_baseline_reproduction_test_predictions.csv"
)
baseline.OUT_MD = "results/baselines/graphsage_knn3_baseline_reproduction_summary.md"


if __name__ == "__main__":
    baseline.main()

