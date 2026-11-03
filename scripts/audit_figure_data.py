import os
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "results/manuscript_draft/tables/final"
OUT_FILE = ROOT / "results/manuscript_draft/figure_data_audit.csv"

def audit():
    audit_results = []

    # FIGURE 3 - Primary Model Comparison
    try:
        t2 = pd.read_csv(TABLES_DIR / "Table2_primary_comparison.csv")
        expected_aps = {
            'GraphSAGE': 0.1107,
            'Matched MLP': 0.0704,
            'LR': 0.0560,
            'XGBoost': 0.0502,
            'LightGBM': 0.0385,
            'RF': 0.0329
        }
        for model, exp in expected_aps.items():
            rendered = float(t2[t2['Model'] == model]['AP'].values[0])
            status = 'PASS' if abs(rendered - exp) < 0.0001 else 'FAIL'
            audit_results.append({
                'Figure': 'Figure 3', 'Panel': 'All', 'Metric/value shown': f'{model} AP',
                'Source CSV/file': 'Table2_primary_comparison.csv', 'Source column': 'AP',
                'Expected value': exp, 'Rendered value': rendered, 'PASS/FAIL': status
            })
    except Exception as e:
        print(f"Error auditing Figure 3: {e}")

    # FIGURE 4 - Core Spatial Ablation
    try:
        t3 = pd.read_csv(TABLES_DIR / "Table3_bootstrap_comparisons.csv")
        # Full features
        full = t3[t3['Features'] == '62 (full)'].iloc[0]
        # Expected (authoritative frozen bootstrap values)
        exp_full = {'Mean_Delta_AP': 0.0350, 'CI_95_Lower': 0.0040, 'CI_95_Upper': 0.0720}
        for metric, exp in exp_full.items():
            rendered = float(full[metric])
            status = 'PASS' if abs(rendered - exp) < 0.0001 else 'FAIL'
            audit_results.append({
                'Figure': 'Figure 4', 'Panel': 'A (Full)', 'Metric/value shown': metric,
                'Source CSV/file': 'Table3_bootstrap_comparisons.csv', 'Source column': metric,
                'Expected value': exp, 'Rendered value': rendered, 'PASS/FAIL': status
            })
            
        # No lag features
        nolag = t3[t3['Features'] == '43 (no lags)'].iloc[0]
        exp_nolag = {'Mean_Delta_AP': 0.0320, 'CI_95_Lower': 0.0060, 'CI_95_Upper': 0.0640}
        for metric, exp in exp_nolag.items():
            rendered = float(nolag[metric])
            status = 'PASS' if abs(rendered - exp) < 0.0001 else 'FAIL'
            audit_results.append({
                'Figure': 'Figure 4', 'Panel': 'B (No Lag)', 'Metric/value shown': metric,
                'Source CSV/file': 'Table3_bootstrap_comparisons.csv', 'Source column': metric,
                'Expected value': exp, 'Rendered value': rendered, 'PASS/FAIL': status
            })
    except Exception as e:
        print(f"Error auditing Figure 4: {e}")

    # FIGURE 5 - Ablation Landscape
    try:
        t5 = pd.read_csv(TABLES_DIR / "Table5_imbalance_loss_ablation.csv")
        # Just check one as a proxy: B5 Ratio 10:1 AP = 0.1110 -> Delta = +0.0003
        rendered = float(t5[t5['Configuration'] == 'B5: Ratio 10:1']['Ensemble_Test_AP'].values[0])
        exp_delta = 0.1110 - 0.1107
        rend_delta = rendered - 0.1107
        status = 'PASS' if abs(rend_delta - exp_delta) < 0.0001 else 'FAIL'
        audit_results.append({
            'Figure': 'Figure 5', 'Panel': 'All', 'Metric/value shown': 'B5 Ratio 10:1 Delta AP',
            'Source CSV/file': 'Table5_imbalance_loss_ablation.csv', 'Source column': 'Ensemble_Test_AP',
            'Expected value': exp_delta, 'Rendered value': rend_delta, 'PASS/FAIL': status
        })
    except Exception as e:
        print(f"Error auditing Figure 5: {e}")

    # FIGURE 7 - Generalization
    try:
        t_diag = pd.read_csv(ROOT / "results/manuscript_draft/canonical_split_diagnostics.csv")
        gs_train = float(t_diag[(t_diag['Model'] == 'GraphSAGE') & (t_diag['Split'] == 'Train')]['AP'].values[0])
        xgb_train = float(t_diag[(t_diag['Model'] == 'XGBoost') & (t_diag['Split'] == 'Train')]['AP'].values[0])
        
        status_gs = 'PASS' if abs(gs_train - 0.0889) < 0.0001 else 'FAIL'
        status_xgb = 'PASS' if abs(xgb_train - 1.0) < 0.0001 else 'FAIL'
        
        audit_results.append({
            'Figure': 'Figure 7', 'Panel': 'A (Generalization)', 'Metric/value shown': 'GraphSAGE Train AP',
            'Source CSV/file': 'canonical_split_diagnostics.csv', 'Source column': 'AP',
            'Expected value': 0.0889, 'Rendered value': gs_train, 'PASS/FAIL': status_gs
        })
        audit_results.append({
            'Figure': 'Figure 7', 'Panel': 'A (Generalization)', 'Metric/value shown': 'XGBoost Train AP',
            'Source CSV/file': 'canonical_split_diagnostics.csv', 'Source column': 'AP',
            'Expected value': 1.0, 'Rendered value': xgb_train, 'PASS/FAIL': status_xgb
        })
    except Exception as e:
        print(f"Error auditing Figure 7: {e}")

    df_audit = pd.DataFrame(audit_results)
    df_audit.to_csv(OUT_FILE, index=False)
    print(f"Figure data audit saved to {OUT_FILE}")

if __name__ == "__main__":
    audit()
