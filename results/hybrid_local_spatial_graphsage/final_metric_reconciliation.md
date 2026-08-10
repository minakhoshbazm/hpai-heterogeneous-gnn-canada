# Final Metric Reconciliation & Model Audit Report

## Master Final Metric Reconciliation Table

| Model Display Name | Model Class | Architecture | Layers | Hidden Dims | Dropout | Train Years | Val Year | Eval Year | Seeds | Per-Seed Mean AP | Per-Seed AP Std | Ensemble AP | Ensemble ROC-AUC | Ensemble Macro P@10 | Ensemble Macro R@50 | Brier Skill Score | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Standalone Non-Graph MLP | HarmonizedPyTorchMLP | Linear(62->32) -> ReLU -> Dropout(0.2) -> Linear(32->32) -> ReLU -> Dropout(0.2) -> Linear(32->1) | 2 | 32 | 0.2 | 2022-2023 | 2024 | 2025 (Locked) | 5 | 0.060858 | 0.008132 | 0.07422 | 0.925875 | 0.06360000371932983 | 0.1599 | -0.015394000336527824 | Harmonized non-graph neural baseline |
| Standalone Spatial GraphSAGE | StandaloneGraphSAGE | SAGEConv(62->32) -> ReLU -> Dropout(0.2) -> SAGEConv(32->32) -> ReLU -> Dropout(0.2) -> Linear(32->1) | 2 | 32 | 0.2 | 2022-2023 | 2024 | 2025 (Locked) | 5 | 0.089832 | 0.015266 | 0.090275 | 0.910913 | 0.07270000129938126 | 0.2001 | 0.008081999607384205 | Homogeneous spatial kNN-3 graph GNN |
| Concatenation Hybrid (Variant A) | ConcatenationLocalSpatialGraphSAGE | Local MLP + Spatial GraphSAGE -> Concat(64) -> Linear(32) -> Linear(1) | 2 | 32 | 0.2 | 2022-2023 | 2024 | 2025 (Locked) | 5 | 0.075105 | 0.01042 | 0.047849 | 0.927088 | 0.06360000371932983 | 0.185 | 0.011734000407159328 | Direct concatenation fusion |
| Gated Residual Hybrid (Variant B) | GatedLocalSpatialGraphSAGE | Local MLP + Spatial GraphSAGE -> Gated Sigmoid Gate -> Fusion(32) -> Linear(1) | 2 | 32 | 0.2 | 2022-2023 | 2024 | 2025 (Locked) | 5 | 0.08787 | 0.019626 | 0.083053 | 0.919808 | 0.054499998688697815 | 0.2132 | 0.004031000193208456 | Learnable gated residual fusion |
| Self-Loop / Disconnected Hybrid Control | GatedLocalSpatialGraphSAGE | Local MLP + Spatial GraphSAGE on Self-Loop Edges Only | 2 | 32 | 0.2 | 2022-2023 | 2024 | 2025 (Locked) | 5 | 0.06588 | 0.009025 | 0.078356 | 0.9244 | 0.07270000129938126 | 0.1991 | 0.03209400177001953 | Disconnected graph architecture control |


## Empirical Shuffled-Edge Graph Distribution (10 Realizations)

| Metric | Shuffled Mean | Shuffled Std | 95% CI | Real Edge Hybrid Value | % Shuffled Runs Beating Real Edge |
| --- | --- | --- | --- | --- | --- |
| Average Precision (AP) | 0.069548 | 0.004762 | [0.063258, 0.075498] | 0.083053 | 0.0% |
| Macro Monthly Recall@50 | 0.1644 | 0.0283 | [0.1218, 0.2042] | 0.2132 | 0.0% |


## Paired Month-Block Bootstrap Comparisons (1,000 Iterations)

| Paired Comparison | Observed AP Diff | 95% Confidence Interval | Bootstrap Win Rate | Statistical Conclusion |
| --- | --- | --- | --- | --- |
| Gated Hybrid vs Harmonized MLP | 0.009639 | [-0.0080, +0.0383] | 84.6% | Comparable (CI includes 0) |
| Gated Hybrid vs Standalone GraphSAGE | -0.005192 | [-0.0202, +0.0095] | 27.0% | Comparable (CI includes 0) |
| Gated Hybrid vs Concatenation Hybrid | 0.031963 | [+0.0023, +0.0597] | 99.8% | Statistically Superior (CI > 0) |
| Gated Hybrid vs Shuffled-Edge Hybrid | 0.009952 | [-0.0110, +0.0450] | 75.2% | Comparable (CI includes 0) |
| Gated Hybrid vs Self-Loop Control | 0.006355 | [-0.0223, +0.0469] | 63.3% | Comparable (CI includes 0) |
| Standalone GraphSAGE vs Harmonized MLP | 0.014831 | [-0.0004, +0.0401] | 96.6% | Comparable (CI includes 0) |


