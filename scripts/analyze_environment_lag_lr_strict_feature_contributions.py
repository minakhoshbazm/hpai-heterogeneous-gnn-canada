"""
Analyze Feature Contributions & Grouped Permutation Importance for Strict Environment + Lag Logistic Regression.

Purpose:
Performs feature contribution analysis and monthly-stratified grouped permutation importance for the strict Environment + Lag LR model:
  - Base-model training: 2022-2024 (181,944 rows)
  - Evaluation: Untouched 2025 Test Set (55,594 rows, 79 positives)
  - Features: Exactly the 50 strict ENV_LAG_FEATURES (43 environment features + 7 approved lag/history features).
  - Strictly excludes ccs_uid, all uid fields, raw outbreak_current_month, identifiers, target, and metadata.

Outputs:
  results/baselines/environment_lag_lr_strict_coefficients.csv
  results/baselines/environment_lag_lr_strict_group_permutation_importance.csv
  results/baselines/environment_lag_lr_strict_feature_contribution_summary.md
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

INPUT_PATH = "data/processed/modeling/model_table_v1_environment_with_lags.csv"
COEFF_OUT = "results/baselines/environment_lag_lr_strict_coefficients.csv"
IMPORTANCE_OUT = "results/baselines/environment_lag_lr_strict_group_permutation_importance.csv"
SUMMARY_MD_OUT = "results/baselines/environment_lag_lr_strict_feature_contribution_summary.md"

EXCLUDE_EXPLICIT_ENV_LAG = {
    'csd_uid', 'csd_name', 'province_code', 'province_name', 'ccs_uid',
    'year', 'month', 'outbreak_next_month',
    'dw_fill_method', 'climate_extraction_method', 'glw2020_chicken_extraction_method',
    'glw2015_duck_extraction_method', 'road_data_coverage',
    'outbreak_current_month'
}

APPROVED_LAG_FEATURES = [
    'same_csd_outbreak_current_month',
    'same_csd_outbreak_prev_1_month',
    'same_csd_outbreak_count_past_3_months',
    'same_csd_outbreak_count_past_6_months',
    'same_csd_outbreak_count_past_12_months',
    'province_outbreak_count_current_month',
    'province_outbreak_count_past_3_months'
]

def assign_group(f):
    if f.startswith('same_csd_outbreak_') or f.startswith('province_outbreak_'):
        return 'Outbreak history'
    elif 'missing_flag' in f or 'fallback' in f or 'pixel_count' in f or 'image_count' in f:
        return 'Missingness/data-quality flags'
    elif f.startswith('climate_'):
        return 'Climate'
    elif f.startswith('dw_'):
        return 'Dynamic World land cover'
    elif f.startswith('ccs_') or f.startswith('glw'):
        return 'Poultry/agriculture'
    elif 'road' in f or 'highway' in f or 'distance' in f or f == 'csd_area_km2' or f == 'small_area_flag':
        return 'Roads/geographic structure'
    else:
        return 'Other'

def format_table(df):
    lines = []
    cols = df.columns.tolist()
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        row_str = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                row_str.append(f"{val:.6f}")
            else:
                row_str.append(str(val))
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)

def main():
    print("==========================================================================================")
    print("  Feature Contribution & Grouped Permutation Importance (Strict Env + Lag LR)")
    print("==========================================================================================")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, dtype={'csd_uid': str})
    print(f"Loaded dataset with shape: {df.shape}")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_EXPLICIT_ENV_LAG and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = sorted(feature_cols)

    # ------------------ STRICT FEATURE AUDIT ASSERTS ------------------
    print("\nExecuting Strict Feature Audit Assertions...")
    print(f"Selected feature count: {len(feature_cols)} (Expected: 50)")
    assert len(feature_cols) == 50, f"ERROR: Feature count is {len(feature_cols)}, expected exactly 50!"
    assert 'ccs_uid' not in feature_cols, "ERROR: ccs_uid is present in feature list!"
    uid_features = [c for c in feature_cols if 'uid' in c.lower()]
    assert len(uid_features) == 0, f"ERROR: Features containing 'uid' detected: {uid_features}"
    assert 'outbreak_current_month' not in feature_cols, "ERROR: Raw outbreak_current_month detected in feature list!"

    for lag in APPROVED_LAG_FEATURES:
        assert lag in feature_cols, f"ERROR: Approved lag feature {lag} missing from feature list!"

    print("SUCCESS: Feature list strictly audited (50 features: 43 env + 7 lags).")

    # Define splits
    train_df = df[df['year'].isin([2022, 2023, 2024])].copy()
    test_df = df[df['year'] == 2025].copy()

    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df['outbreak_next_month'].values.astype(int)
    X_test_orig = test_df[feature_cols].values.astype(np.float64)
    y_test = test_df['outbreak_next_month'].values.astype(int)
    months_test = test_df['month'].values

    # Fit scaler on Train features only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled_orig = scaler.transform(X_test_orig)

    # Fit weighted Logistic Regression
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # ---------------- 1. STANDARDIZED COEFFICIENTS ----------------
    print("\nCalculating Standardized Coefficients...")
    coeffs = clf.coef_[0]
    
    coeff_records = []
    for f_name, coef in zip(feature_cols, coeffs):
        grp = assign_group(f_name)
        odds_ratio = np.exp(coef)
        direction = "higher-risk" if coef > 0 else "lower-risk"
        coeff_records.append({
            'feature_name': f_name,
            'feature_group': grp,
            'standardized_coefficient': coef,
            'odds_ratio': odds_ratio,
            'risk_association': direction
        })

    coeff_df = pd.DataFrame(coeff_records).sort_values(by='standardized_coefficient', ascending=False)
    
    # ---------------- 2. GROUPED PERMUTATION IMPORTANCE ----------------
    print("\nRunning Monthly-Stratified Grouped Permutation Importance (30 repetitions)...")
    baseline_probs = clf.predict_proba(X_test_scaled_orig)[:, 1]
    baseline_pr_auc = average_precision_score(y_test, baseline_probs)
    print(f"Baseline 2025 Test PR-AUC: {baseline_pr_auc:.6f}")

    feat_to_idx = {f: i for i, f in enumerate(feature_cols)}
    groups_dict = {}
    for f in feature_cols:
        g = assign_group(f)
        groups_dict.setdefault(g, []).append(feat_to_idx[f])

    n_repeats = 30
    importance_records = []

    for group_name, col_indices in sorted(groups_dict.items()):
        print(f"  - Permuting group: {group_name} ({len(col_indices)} features)...")
        pr_auc_drops = []
        
        for rep in range(n_repeats):
            rng = np.random.RandomState(seed=42 + rep)
            X_test_perm = X_test_orig.copy()
            
            for m in np.unique(months_test):
                m_mask = (months_test == m)
                m_indices = np.where(m_mask)[0]
                permuted_m_indices = rng.permutation(m_indices)
                
                for c_idx in col_indices:
                    X_test_perm[m_indices, c_idx] = X_test_orig[permuted_m_indices, c_idx]
            
            X_test_perm_scaled = scaler.transform(X_test_perm)
            perm_probs = clf.predict_proba(X_test_perm_scaled)[:, 1]
            perm_pr_auc = average_precision_score(y_test, perm_probs)
            drop = baseline_pr_auc - perm_pr_auc
            pr_auc_drops.append(drop)
            
        mean_drop = np.mean(pr_auc_drops)
        std_drop = np.std(pr_auc_drops)
        importance_records.append({
            'feature_group': group_name,
            'feature_count': len(col_indices),
            'baseline_pr_auc': baseline_pr_auc,
            'mean_pr_auc_drop': mean_drop,
            'std_pr_auc_drop': std_drop,
            'relative_importance_pct': (mean_drop / baseline_pr_auc) * 100.0 if baseline_pr_auc > 0 else 0.0
        })

    importance_df = pd.DataFrame(importance_records).sort_values(by='mean_pr_auc_drop', ascending=False)

    top20_positive = coeff_df.head(20)
    top20_negative = coeff_df.tail(20).iloc[::-1]

    # Save CSV outputs
    os.makedirs(os.path.dirname(COEFF_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(IMPORTANCE_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_MD_OUT), exist_ok=True)

    coeff_df.to_csv(COEFF_OUT, index=False)
    importance_df.to_csv(IMPORTANCE_OUT, index=False)

    # Save Markdown Summary using custom format_table
    md_content = f"""# Strict Environment + Lag Logistic Regression Feature Contribution Analysis

## Executive Summary
This document reports the standardized feature coefficients and monthly-stratified grouped permutation importance for the strict 50-feature Environment + Lag Logistic Regression model evaluated on the untouched 2025 test set.

> [!IMPORTANT]
> **CAUSAL DISCLAIMER**: These results represent **statistical predictive associations** identified within observational data. They do **not** constitute proof of biological or epidemiological causation.

---

## 1. Grouped Permutation Importance (2025 Test Set)
Features were permuted within each calendar month across 30 random seeds to preserve monthly seasonality.

{format_table(importance_df)}

---

## 2. Top 20 Positive Feature Coefficients (Higher Risk Association)

{format_table(top20_positive[['feature_name', 'feature_group', 'standardized_coefficient', 'odds_ratio']])}

---

## 3. Top 20 Negative Feature Coefficients (Lower Risk Association)

{format_table(top20_negative[['feature_name', 'feature_group', 'standardized_coefficient', 'odds_ratio']])}
"""

    with open(SUMMARY_MD_OUT, "w") as f:
        f.write(md_content)

    print("\n==========================================================================================")
    print("                    GROUPED PERMUTATION IMPORTANCE (2025 TEST SET)")
    print("==========================================================================================")
    print(importance_df.to_string(index=False))

    print("\n==========================================================================================")
    print("                   TOP 10 POSITIVE STANDARDIZED COEFFICIENTS")
    print("==========================================================================================")
    print(top20_positive.head(10)[['feature_name', 'feature_group', 'standardized_coefficient', 'odds_ratio']].to_string(index=False))

    print("\n==========================================================================================")
    print("                   TOP 10 NEGATIVE STANDARDIZED COEFFICIENTS")
    print("==========================================================================================")
    print(top20_negative.head(10)[['feature_name', 'feature_group', 'standardized_coefficient', 'odds_ratio']].to_string(index=False))

    print(f"\nSaved coefficients table to: {COEFF_OUT}")
    print(f"Saved grouped importance table to: {IMPORTANCE_OUT}")
    print(f"Saved summary markdown report to: {SUMMARY_MD_OUT}")

if __name__ == '__main__':
    main()
