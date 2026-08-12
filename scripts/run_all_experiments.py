#!/usr/bin/env python3
"""
Master script to run all GraphSAGE improvement experiments (Groups A to I),
aggregate results, perform bootstrapping, and output the final registry and markdown report.
"""

from __future__ import annotations
import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "results/experiment_registry.csv"

def run_group_script(script_name: str):
    print("\n" + "=" * 80)
    print(f"  RUNNING SCRIPT: {script_name}")
    print("=" * 80 + "\n")
    script_path = ROOT / "scripts" / script_name
    subprocess.run(["py", str(script_path)], check=True)

def perform_csd_bootstrap(
    y_test: np.ndarray,
    baseline_probs: np.ndarray,
    candidate_probs: np.ndarray,
    test_csds: np.ndarray,
    n_boot: int = 1000
) -> tuple[float, float, float, float]:
    
    unique_csds = np.unique(test_csds)
    n_csds = len(unique_csds)
    csd_indices = {csd: np.where(test_csds == csd)[0] for csd in unique_csds}
    
    np.random.seed(42)
    diffs = []
    
    for _ in range(n_boot):
        boot_csds = np.random.choice(unique_csds, size=n_csds, replace=True)
        boot_idx = np.concatenate([csd_indices[c] for c in boot_csds])
        
        y_b = y_test[boot_idx]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue
            
        ap_base = average_precision_score(y_b, baseline_probs[boot_idx])
        ap_cand = average_precision_score(y_b, candidate_probs[boot_idx])
        diffs.append(ap_cand - ap_base)
        
    observed_diff = average_precision_score(y_test, candidate_probs) - average_precision_score(y_test, baseline_probs)
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    win_rate = np.mean(np.array(diffs) > 0)
    
    return observed_diff, ci_lower, ci_upper, win_rate

def main():
    print("Starting HPAI GraphSAGE Improvement Master Pipeline...")
    
    # List of scripts to run (Groups B to I; Group A baseline was run first)
    scripts = [
        "run_group_b_negative_sampling.py",
        "run_group_c_loss.py",
        "run_group_d_ssl.py",
        "run_group_e_transfer.py",
        "run_group_f_combined.py",
        "run_group_g_temporal.py",
        "run_group_h_ablation.py",
        "run_group_i_graphs.py"
    ]
    
    for s in scripts:
        try:
            run_group_script(s)
        except Exception as e:
            print(f"Error executing {s}: {e}")

    # Now load and aggregate predictions for CSD cluster bootstrapping
    print("\n--- Performing CSD-level Cluster Bootstrapping vs. Baseline ---")
    
    base_pred_path = ROOT / "results/group_a_baseline_graphsage_knn3_2022_2025/test_predictions.csv"
    if not base_pred_path.exists():
        print(f"Baseline predictions not found at {base_pred_path}. Bootstrapping aborted.")
        return
        
    df_base = pd.read_csv(base_pred_path)
    y_test = df_base["actual"].values
    baseline_probs = df_base["ensemble_prob"].values
    test_csds = df_base["csd_uid"].values
    
    candidate_paths = {
        "B1_all_poultry": ROOT / "results/group_b_negative_sampling/B1_all_poultry_test_predictions.csv",
        "B4_ratio_25": ROOT / "results/group_b_negative_sampling/B4_ratio_25_test_predictions.csv",
        "G2_gru_6months": ROOT / "results/group_g_temporal/temporal_summary_results.csv"  # will use its prediction file if written
    }
    
    # We will compute bootstrap CIs for all available test prediction files
    bootstrap_results = []
    
    # Scan for any test prediction CSV files in results directories
    results_dir = ROOT / "results"
    for r_dir in results_dir.glob("group_*"):
        for pred_file in r_dir.glob("*test_predictions.csv"):
            if "group_a" in str(r_dir):
                continue
            config_name = pred_file.stem.replace("_test_predictions", "")
            try:
                df_cand = pd.read_csv(pred_file)
                cand_probs = df_cand["ensemble_prob"].values
                
                obs_diff, ci_low, ci_high, win = perform_csd_bootstrap(y_test, baseline_probs, cand_probs, test_csds)
                
                bootstrap_results.append({
                    "Config": config_name,
                    "AP Diff": obs_diff,
                    "95% CI Lower": ci_low,
                    "95% CI Upper": ci_high,
                    "Win Rate": win,
                    "Significant": "Yes" if (ci_low > 0 or ci_high < 0) else "No"
                })
                print(f"  {config_name} vs Baseline: diff={obs_diff:+.4f} | 95% CI [{ci_low:+.4f}, {ci_high:+.4f}] | win={win:.1%} | Sig: {bootstrap_results[-1]['Significant']}")
            except Exception as e:
                # Prediction file might be structured differently or not finished
                pass

    if bootstrap_results:
        df_boot = pd.DataFrame(bootstrap_results)
        df_boot.to_csv(results_dir / "bootstrap_comparisons.csv", index=False)
        print("\n=== Bootstrap Results ===")
        print(df_boot.to_markdown(index=False))

    print("\nMaster Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
