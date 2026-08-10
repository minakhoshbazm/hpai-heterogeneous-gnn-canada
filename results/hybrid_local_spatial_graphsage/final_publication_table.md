# Primary Publication Table: Harmonized Outbreak Risk Forecasting Benchmark

*Evaluated on the locked 2025 retrospective test set (55,594 region-months, 79 positive outbreaks) across 5 independent random seeds. All neural models use identical 62 numeric features, 1:25 negative training ratio, and out-of-fold Platt probability calibration.*

| Model Architecture | Model Type | Per-Seed AP (Mean ± Std) | Ensemble AP (95% CI) | Ensemble ROC-AUC | Macro Monthly P@10 | Macro Monthly R@50 | Macro Monthly R@100 | Calibrated Brier Score | Brier Skill Score (BSS) | Primary Scientific Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Standalone Spatial GraphSAGE** | Spatial GNN | **0.0898 ± 0.0153** | **0.0903** `[0.0752, 0.1084]` | 0.9109 | **0.0727** | 0.2001 | 0.2858 | 0.001408 | **+0.0081** | **Highest AP Risk Ranking** |
| **Gated Local-Spatial GraphSAGE** | Dual-Branch Hybrid | 0.0879 ± 0.0196 | 0.0831 `[0.0684, 0.1012]` | 0.9198 | 0.0545 | **0.2132** | **0.3015** | 0.001413 | +0.0040 | **Highest Monthly Outbreak Retrieval** |
| **Standalone Non-Graph MLP** | Neural Local Baseline | 0.0609 ± 0.0081 | 0.0742 `[0.0581, 0.0945]` | 0.9259 | 0.0636 | 0.1599 | 0.2284 | 0.001441 | -0.0154 | Non-Graph Neural Baseline |
| **Concatenation Hybrid (Variant A)** | Dual-Branch Hybrid | 0.0751 ± 0.0104 | 0.0478 `[0.0381, 0.0620]` | 0.9271 | 0.0636 | 0.1850 | 0.2612 | 0.001402 | +0.0117 | Direct Concatenation Fusion |
| **Self-Loop / Disconnected Control** | Disconnected Graph | 0.0659 ± 0.0090 | 0.0784 `[0.0625, 0.0980]` | 0.9244 | **0.0727** | 0.1991 | 0.2801 | 0.001408 | +0.0321 | Architecture Control |
| **Shuffled-Edge Control (10 Runs)** | Random Edge Control | 0.0696 ± 0.0048 | 0.0696 `[0.0633, 0.0755]` | **0.9171** | 0.0545 | 0.1639 | 0.2458 | 0.001410 | +0.0078 | Topological Graph Control |
