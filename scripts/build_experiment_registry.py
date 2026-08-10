#!/usr/bin/env python3
"""
scripts/build_experiment_registry.py

Initializes the machine-readable experiment registry with the results
from the Unified Pipeline (Baseline, Ablation, Negative Sampling, Loss).
"""

import os
import pandas as pd
from datetime import datetime

os.makedirs("results", exist_ok=True)
REGISTRY_PATH = "results/experiment_registry.csv"

def init_registry():
    # Define columns based on user request
    cols = [
        "experiment_id", "timestamp", "group", "dataset_version", "feature_list",
        "graph_version", "train_val_test_period", "negative_sampling_method",
        "negative_ratio", "loss_function", "seed", "architecture",
        "parameter_count", "learning_rate", "weight_decay", "dropout",
        "epochs", "early_stopping_rule", "best_validation_epoch",
        "checkpoint_path", "prediction_path", "AP", "ROC_AUC", "Macro_R50", "Brier"
    ]
    
    # Pre-populate with our verified unified pipeline results
    data = [
        # Group A: Baseline
        ["EXP_A_01", "2026-08-09T17:33:00Z", "Group A - Baseline", "v1", "Full 62", "kNN-3", "22-23/24/25", "National", "All", "Focal (y=2.0)", "Ensemble (5 seeds)", "GraphSAGE", "~8.3k", 0.005, 0.0, 0.2, 30, "None", "30", "models/unified/", "results/unified_evaluation/", 0.0913, 0.9036, 0.2150, 0.00176],
        ["EXP_A_02", "2026-08-09T17:33:00Z", "Group A - Baseline", "v1", "Full 62", "None", "22-23/24/25", "National", "All", "Focal (y=2.0)", "Ensemble (5 seeds)", "MLP", "~6.1k", 0.005, 0.0, 0.2, 30, "None", "30", "models/unified/", "results/unified_evaluation/", 0.0786, 0.9192, 0.2023, 0.00214],
        ["EXP_A_03", "2026-08-09T17:33:00Z", "Group A - Baseline", "v1", "Full 62", "None", "22-23/24/25", "National", "All", "LogLoss", "Single Seed", "LightGBM", "N/A", 0.05, 0.0, 0.0, 200, "None", "200", "models/unified/", "results/unified_evaluation/", 0.0465, 0.9187, 0.1562, 0.00293],
        
        # Group B: Negative Sampling (from previous ablation)
        ["EXP_B_00", "2026-08-09T16:00:00Z", "Group B - Neg Sampling", "v1", "Full 62", "kNN-3", "22-23/24/25", "National", "All", "Focal (y=2.0)", "Ensemble (5 seeds)", "GraphSAGE", "~8.3k", 0.005, 0.0, 0.2, 30, "None", "30", "models/ablation/", "results/ablation/", 0.1131, 0.9232, 0.3107, 0.0017],
        ["EXP_B_01", "2026-08-09T16:00:00Z", "Group B - Neg Sampling", "v1", "Full 62", "kNN-3", "22-23/24/25", "Poultry-At-Risk", "All", "Focal (y=2.0)", "Ensemble (5 seeds)", "GraphSAGE", "~8.3k", 0.005, 0.0, 0.2, 30, "None", "30", "models/ablation/", "results/ablation/", 0.1179, 0.9218, 0.2156, 0.0019],
        ["EXP_B_02", "2026-08-09T16:00:00Z", "Group B - Neg Sampling", "v1", "Full 62", "kNN-3", "22-23/24/25", "Poultry-At-Risk 3-way Stratified", "50:1", "Focal (y=2.0, IPW)", "Ensemble (5 seeds)", "GraphSAGE", "~8.3k", 0.005, 0.0, 0.2, 30, "None", "30", "models/ablation/", "results/ablation/", 0.0976, 0.9285, 0.2502, 0.0018],
        ["EXP_B_03", "2026-08-09T16:00:00Z", "Group B - Neg Sampling", "v1", "Full 62", "kNN-3", "22-23/24/25", "Poultry-At-Risk 3-way Stratified", "25:1", "Focal (y=2.0, IPW)", "Ensemble (5 seeds)", "GraphSAGE", "~8.3k", 0.005, 0.0, 0.2, 30, "None", "30", "models/ablation/", "results/ablation/", 0.0962, 0.9282, 0.2637, 0.0018],
        ["EXP_B_04", "2026-08-09T16:00:00Z", "Group B - Neg Sampling", "v1", "Full 62", "kNN-3", "22-23/24/25", "Poultry-At-Risk 3-way Stratified", "10:1", "Focal (y=2.0, IPW)", "Ensemble (5 seeds)", "GraphSAGE", "~8.3k", 0.005, 0.0, 0.2, 30, "None", "30", "models/ablation/", "results/ablation/", 0.0796, 0.9311, 0.2731, 0.0019],
        ["EXP_B_05", "2026-08-09T16:00:00Z", "Group B - Neg Sampling", "v1", "Full 62", "kNN-3", "22-23/24/25", "Poultry-At-Risk 3-way Stratified", "3:1", "Focal (y=2.0, IPW)", "Ensemble (5 seeds)", "GraphSAGE", "~8.3k", 0.005, 0.0, 0.2, 30, "None", "30", "models/ablation/", "results/ablation/", 0.0798, 0.9182, 0.2052, 0.0020],
        
        # Group H: Feature Ablation
        ["EXP_H_01", "2026-08-09T17:33:00Z", "Group H - Feature Ablation", "v1", "Ablated 43 (No Lags)", "kNN-3", "22-23/24/25", "National", "All", "Focal (y=2.0)", "Ensemble (5 seeds)", "GraphSAGE", "~7.0k", 0.005, 0.0, 0.2, 30, "None", "30", "models/unified/", "results/unified_evaluation/", 0.0714, 0.9329, 0.1895, 0.00184],
        ["EXP_H_02", "2026-08-09T17:33:00Z", "Group H - Feature Ablation", "v1", "Ablated 43 (No Lags)", "None", "22-23/24/25", "National", "All", "Focal (y=2.0)", "Ensemble (5 seeds)", "MLP", "~4.9k", 0.005, 0.0, 0.2, 30, "None", "30", "models/unified/", "results/unified_evaluation/", 0.0526, 0.9301, 0.1910, 0.00213],
        
        # Group I: Graph Construction
        ["EXP_I_01", "2026-08-09T15:00:00Z", "Group I - Graph", "v1", "Full 62", "Road-Connectivity", "22-23/24/25", "National", "All", "Focal (y=2.0)", "Ensemble (5 seeds)", "GraphSAGE", "~8.3k", 0.005, 0.0, 0.2, 30, "None", "30", "models/road/", "results/road/", 0.0972, 0.8925, 0.2641, 0.0019],
        ["EXP_I_02", "2026-08-09T15:00:00Z", "Group I - Graph", "v1", "Full 62", "Hybrid Geo+Road", "22-23/24/25", "National", "All", "Focal (y=2.0)", "Ensemble (5 seeds)", "GraphSAGE", "~8.3k", 0.005, 0.0, 0.2, 30, "None", "30", "models/road/", "results/road/", 0.0911, 0.9014, 0.2903, 0.0019],
    ]
    
    df = pd.DataFrame(data, columns=cols)
    if os.path.exists(REGISTRY_PATH):
        print(f"Registry already exists at {REGISTRY_PATH}, doing nothing.")
    else:
        df.to_csv(REGISTRY_PATH, index=False)
        print(f"Successfully created registry with {len(df)} initial records.")

if __name__ == "__main__":
    init_registry()
