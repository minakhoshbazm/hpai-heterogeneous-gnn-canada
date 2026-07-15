#!/usr/bin/env python3
"""
scripts/create_pseudo_absence_training_sets.py

Implements stratified and exposure-weighted pseudo-absence sampling
for the 2022-2024 training splits. Uses numpy choice to support weighted sampling with replace=False.
"""

import os
import numpy as np
import pandas as pd

def main():
    model_table_path = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
    
    print("Loading modeling table...")
    df_all = pd.read_csv(model_table_path)
    
    # 1. Filter to training years 2022, 2023, 2024
    df_train = df_all[df_all['year'].isin([2022, 2023, 2024])].copy()
    print(f"Training set row count: {len(df_train)}")
    
    # Separating positives and negatives
    df_pos = df_train[df_train['outbreak_next_month'] == 1].copy()
    df_neg = df_train[df_train['outbreak_next_month'] == 0].copy()
    
    pos_count = len(df_pos)
    print(f"Positive training outbreaks: {pos_count}")
    
    # 6. Define poultry exposure proxy columns to construct sampling weights
    exposure_cols = [
        'ccs_broilers_roasters_cornish_production_kg',
        'ccs_broilers_roasters_cornish_farms_reporting',
        'ccs_turkey_production_kg',
        'ccs_turkey_farms_reporting'
    ]
    
    # Check which exposure columns exist in dataframe
    existing_exp_cols = [c for c in exposure_cols if c in df_neg.columns]
    print(f"Found {len(existing_exp_cols)} poultry exposure columns for weights calculation: {existing_exp_cols}")
    
    # Compute sampling weight: log1p of positive exposure proxies + small base weight
    if existing_exp_cols:
        # Sum non-negative values of existing exposure metrics
        exposure_sum = df_neg[existing_exp_cols].fillna(0.0).clip(lower=0.0).sum(axis=1)
        raw_weights = np.log1p(exposure_sum)
        # Scale to avoid huge ranges, then add base weight
        base_weight = 0.05
        weights = raw_weights / (raw_weights.max() if raw_weights.max() > 0 else 1.0) + base_weight
    else:
        # Fallback to uniform base weights
        print("Warning: No exposure columns found. Falling back to uniform weights.")
        weights = pd.Series(1.0, index=df_neg.index)
        
    df_neg['sampling_weight'] = weights

    # Ratios to process
    ratios = [5, 10, 20]
    np.random.seed(42)  # Fixed random seed for reproducibility

    for r in ratios:
        print(f"\n--- Processing Ratio {r}:1 ---")
        target_neg_count = pos_count * r
        
        # Sampling stratified by Month and Province where possible
        sampled_neg_indices = set()
        
        # Group positives by province and month
        pos_strata = df_pos.groupby(['province_code', 'month']).size().to_dict()
        
        for (prov, m), count in pos_strata.items():
            needed = count * r
            # Find eligible negatives matching this stratum
            eligible = df_neg[
                (df_neg['province_code'] == prov) & 
                (df_neg['month'] == m)
            ]
            
            # If not enough, fallback to same month across all provinces
            if len(eligible) < needed:
                eligible = df_neg[df_neg['month'] == m]
                
            # If still not enough, fallback to any training negative
            if len(eligible) < needed:
                eligible = df_neg
                
            # Filter out already sampled indices to prevent duplicates
            eligible = eligible[~eligible.index.isin(sampled_neg_indices)]
            
            if len(eligible) == 0:
                continue
                
            # Sample using exposure weights via numpy random choice to avoid pandas sample errors
            sample_size = min(needed, len(eligible))
            
            # Normalize weights for this subset
            w = eligible['sampling_weight'].values
            probs = w / w.sum()
            
            chosen_idx = np.random.choice(eligible.index, size=sample_size, replace=False, p=probs)
            sampled_neg_indices.update(chosen_idx)
            
        # If we are slightly short of total target negatives due to stratum exhaustions, top-up
        if len(sampled_neg_indices) < target_neg_count:
            rem_needed = target_neg_count - len(sampled_neg_indices)
            eligible = df_neg[~df_neg.index.isin(sampled_neg_indices)]
            if len(eligible) >= rem_needed:
                w = eligible['sampling_weight'].values
                probs = w / w.sum()
                chosen_idx = np.random.choice(eligible.index, size=rem_needed, replace=False, p=probs)
                sampled_neg_indices.update(chosen_idx)
                
        # Build training set
        df_sampled_neg = df_neg.loc[list(sampled_neg_indices)].drop(columns=['sampling_weight'])
        df_out = pd.concat([df_pos, df_sampled_neg], axis=0).sample(frac=1.0).reset_index(drop=True)
        
        # Output file path
        out_path = f"data/processed/modeling/pseudo_absence_train_ratio_{r}.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print(f"Saved training dataset to: {out_path}")
        
        # 8. Print QA metrics
        print(f"  Positive Row Count:            {len(df_pos)}")
        print(f"  Sampled Negative Row Count:    {len(df_sampled_neg)}")
        print(f"  Final Total Row Count:         {len(df_out)}")
        print(f"  Positive Rate:                 {len(df_pos) / len(df_out):.4%}")
        print(f"  Unique CSDs Represented:       {df_out['csd_uid'].nunique()}")
        print(f"  Provinces Represented:         {df_out['province_code'].nunique()}")
        
        # Month distribution
        month_dist = df_out['month'].value_counts().sort_index().to_dict()
        print(f"  Month Distribution:            {month_dist}")
        
        # Check duplicate rows key
        dups = df_out.duplicated(subset=['csd_uid', 'year', 'month']).sum()
        print(f"  Duplicate key rows exist:      {dups > 0} (count: {dups})")

if __name__ == '__main__':
    main()
