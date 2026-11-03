import os
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, confusion_matrix, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

# Define paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "modeling", "model_table_v1_environment_with_lags_wildbird.csv")
PREDS_DIR = os.path.join(ROOT_DIR, "results", "final_manuscript", "canonical_predictions")
OUT_DIR = os.path.join(ROOT_DIR, "results", "evaluation_audit")
os.makedirs(OUT_DIR, exist_ok=True)

# Map available canonical prediction models
SOURCES = {
    "LR": os.path.join(PREDS_DIR, "preds_LR.csv"),
    "RF": os.path.join(PREDS_DIR, "preds_RF.csv"),
    "XGBoost": os.path.join(PREDS_DIR, "preds_XGB.csv"),
    "Matched_MLP": os.path.join(ROOT_DIR, "results", "graphsage_stage5", "predictions", "preds_M_baseline.csv"),
    "GraphSAGE_Canonical": os.path.join(ROOT_DIR, "results", "graphsage_stage5", "predictions", "preds_G_baseline.csv"),
    "GraphSAGE_PseudoAbs_10to1": os.path.join(ROOT_DIR, "results", "graphsage_stage2", "predictions", "preds_B5_ratio_10.csv"),
    "GraphSAGE_PseudoAbs_100to1": os.path.join(ROOT_DIR, "results", "graphsage_stage2", "predictions", "preds_B2_ratio_100.csv"),
    "Logistic_PseudoAbs_10to1": os.path.join(ROOT_DIR, "results", "baselines", "pseudo_absence_logistic_test_predictions.csv")
}

def main():
    print("Loading test data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file {DATA_PATH} not found.")
        return
        
    df_all = pd.read_csv(DATA_PATH, dtype={"csd_uid": str})
    
    # Check what is in the 2025 Test Split
    df_test = df_all[df_all["year"] == 2025].copy()
    y_test_true = df_test["outbreak_next_month"].values
    
    # Build Poultry At-Risk Eligible Domain Mask
    farm_cols = ['ccs_broilers_roasters_cornish_farms_reporting', 'ccs_broilers_roasters_cornish_production_kg', 'ccs_turkey_farms_reporting', 'ccs_turkey_production_kg']
    df_all['farm_sum'] = df_all[farm_cols].fillna(0).sum(axis=1)
    at_risk_csds = set(df_all.groupby('csd_uid')['farm_sum'].max().loc[lambda x: x > 0].index)
    eligible_mask = df_test['csd_uid'].isin(at_risk_csds) | (df_test['outbreak_next_month'] == 1)
    
    df_test_eligible = df_test[eligible_mask].copy()
    y_test_eligible = df_test_eligible['outbreak_next_month'].values
    
    print("\n--- Pipeline Data Audit ---")
    print(f"Total 2025 Test Cells (All Landscape): {len(df_test)} (Pos: {int(y_test_true.sum())}, Neg: {int(len(df_test) - y_test_true.sum())})")
    print(f"Eligible 2025 Test Cells (Poultry At-Risk): {len(df_test_eligible)} (Pos: {int(y_test_eligible.sum())}, Neg: {int(len(df_test_eligible) - y_test_eligible.sum())})")
    
    results_all = []
    results_eligible = []
    
    def get_top_k_metrics(y_true, y_prob, k):
        if y_true.sum() == 0: return 0.0, 0.0
        top_k_idx = np.argsort(y_prob)[-k:][::-1]
        top_k_true = y_true[top_k_idx]
        return top_k_true.sum() / k, top_k_true.sum() / y_true.sum()

    for model_name, path in SOURCES.items():
        if not os.path.exists(path):
            continue
            
        df_preds = pd.read_csv(path, dtype={"csd_uid": str})
        prob_col = "prob_ratio_10" if model_name == "Logistic_PseudoAbs_10to1" else "pred_prob_ensemble"
        
        # 1. All Cells Domain
        df_merged_all = df_test.merge(df_preds[["csd_uid", "year", "month", prob_col]], on=["csd_uid", "year", "month"], how="left")
        y_prob_all = df_merged_all[prob_col].fillna(0.0).values
        p10_all, r10_all = get_top_k_metrics(y_test_true, y_prob_all, 10)
        p50_all, r50_all = get_top_k_metrics(y_test_true, y_prob_all, 50)
        results_all.append({
            "Model": model_name,
            "Domain": "All Canadian Cells",
            "Positives": int(y_test_true.sum()),
            "Negatives": int(len(df_test) - y_test_true.sum()),
            "PR-AUC": average_precision_score(y_test_true, y_prob_all),
            "ROC-AUC": roc_auc_score(y_test_true, y_prob_all),
            "Brier": brier_score_loss(y_test_true, y_prob_all),
            "P@10": p10_all,
            "P@50": p50_all,
            "R@50": r50_all
        })
        
        # 2. Eligible Poultry-at-Risk Domain
        df_merged_el = df_test_eligible.merge(df_preds[["csd_uid", "year", "month", prob_col]], on=["csd_uid", "year", "month"], how="left")
        y_prob_el = df_merged_el[prob_col].fillna(0.0).values
        p10_el, r10_el = get_top_k_metrics(y_test_eligible, y_prob_el, 10)
        p50_el, r50_el = get_top_k_metrics(y_test_eligible, y_prob_el, 50)
        results_eligible.append({
            "Model": model_name,
            "Domain": "Eligible Poultry Cells",
            "Positives": int(y_test_eligible.sum()),
            "Negatives": int(len(df_test_eligible) - y_test_eligible.sum()),
            "PR-AUC": average_precision_score(y_test_eligible, y_prob_el),
            "ROC-AUC": roc_auc_score(y_test_eligible, y_prob_el),
            "Brier": brier_score_loss(y_test_eligible, y_prob_el),
            "P@10": p10_el,
            "P@50": p50_el,
            "R@50": r50_el
        })

    # Save outputs
    pd.DataFrame(results_all).to_csv(os.path.join(OUT_DIR, "evaluation_all_cells.csv"), index=False)
    pd.DataFrame(results_eligible).to_csv(os.path.join(OUT_DIR, "evaluation_eligible_cells.csv"), index=False)
    print("Saved evaluation_all_cells.csv and evaluation_eligible_cells.csv")

if __name__ == "__main__":
    main()
