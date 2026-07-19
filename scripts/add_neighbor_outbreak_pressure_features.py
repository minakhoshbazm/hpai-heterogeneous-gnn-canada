#!/usr/bin/env python3
"""
scripts/add_neighbor_outbreak_pressure_features.py

Constructs spatial outbreak-pressure features from neighboring CSDs.
"""

import os
import numpy as np
import pandas as pd

def main():
    model_table_path = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
    edges_path = 'data/processed/graph/csd_adjacency_edges_v1_model_scope.csv'
    out_table_path = 'data/processed/modeling/model_table_v1_environment_with_lags_and_neighbor_pressure.csv'

    print("Loading datasets...")
    df_model = pd.read_csv(model_table_path)
    df_edges = pd.read_csv(edges_path)
    
    input_row_count = len(df_model)
    print(f"Input model table rows: {input_row_count}")

    # Reconstruct or find outbreak_current_month
    # Rules: check if present, otherwise outbreak_current_month(t) = outbreak_next_month(t-1)
    df_model = df_model.sort_values(by=['csd_uid', 'year', 'month']).reset_index(drop=True)
    
    if 'outbreak_current_month' not in df_model.columns:
        print("Reconstructing 'outbreak_current_month' using lag of 'outbreak_next_month'...")
        # Since we sorted by CSD and chronologically, we can shift outbreak_next_month per CSD
        df_model['outbreak_current_month'] = df_model.groupby('csd_uid')['outbreak_next_month'].shift(1).fillna(0.0)
    else:
        print("Using existing 'outbreak_current_month' column.")

    # Reconstruct past histories per CSD: t-1, t-2, etc. to compute running sums
    # We need current month, past 3 months (t, t-1, t-2), past 6 months (t to t-5)
    # Use shift to get lagged values
    g = df_model.groupby('csd_uid')
    
    o_t = df_model['outbreak_current_month'].values
    o_t1 = g['outbreak_current_month'].shift(1).fillna(0.0).values
    o_t2 = g['outbreak_current_month'].shift(2).fillna(0.0).values
    o_t3 = g['outbreak_current_month'].shift(3).fillna(0.0).values
    o_t4 = g['outbreak_current_month'].shift(4).fillna(0.0).values
    o_t5 = g['outbreak_current_month'].shift(5).fillna(0.0).values
    
    # Calculate rolling metrics per CSD-month
    df_model['outbreaks_current'] = o_t
    df_model['outbreaks_past3'] = o_t + o_t1 + o_t2
    df_model['outbreaks_past6'] = o_t + o_t1 + o_t2 + o_t3 + o_t4 + o_t5

    # Create adjacency mapping: neighbor lookup
    # Edges are undirected
    adj = {}
    for _, row in df_edges.iterrows():
        u = int(row['source_csd_uid'])
        v = int(row['target_csd_uid'])
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    print("Pivot feature values to compute spatial pressures...")
    # Pivot outbreak variables to easily map: index: CSD_UID, columns: (year, month)
    # This allows fast matrix vector multiplication or mapping
    # Create unique period keys
    df_model['period'] = df_model['year'].astype(str) + "_" + df_model['month'].astype(str)
    
    pivot_current = df_model.pivot(index='csd_uid', columns='period', values='outbreaks_current').fillna(0.0)
    pivot_past3 = df_model.pivot(index='csd_uid', columns='period', values='outbreaks_past3').fillna(0.0)
    pivot_past6 = df_model.pivot(index='csd_uid', columns='period', values='outbreaks_past6').fillna(0.0)
    
    all_csds = pivot_current.index.tolist()
    csd_set = set(all_csds)
    periods = pivot_current.columns.tolist()

    # Pre-build neighbor arrays
    neighbor_current_sum = pd.DataFrame(0.0, index=all_csds, columns=periods)
    neighbor_past3_sum = pd.DataFrame(0.0, index=all_csds, columns=periods)
    neighbor_past6_sum = pd.DataFrame(0.0, index=all_csds, columns=periods)
    
    neighbor_counts = {}

    for csd in all_csds:
        ns = adj.get(csd, [])
        # Only count neighbors that exist in the modeling table CSD scope
        ns_valid = [n for n in ns if n in csd_set]
        neighbor_counts[csd] = len(ns_valid)
        
        if len(ns_valid) > 0:
            # sum neighbors along periods axis
            neighbor_current_sum.loc[csd] = pivot_current.loc[ns_valid].sum(axis=0)
            neighbor_past3_sum.loc[csd] = pivot_past3.loc[ns_valid].sum(axis=0)
            neighbor_past6_sum.loc[csd] = pivot_past6.loc[ns_valid].sum(axis=0)

    # Melt pivoted metrics back to match original rows
    def melt_df(df_piv, name):
        df_piv.index.name = 'csd_uid'
        df_piv.columns.name = 'period'
        df_melt = df_piv.reset_index().melt(id_vars='csd_uid', var_name='period', value_name=name)
        df_melt[['year', 'month']] = df_melt['period'].str.split('_', expand=True).astype(int)
        df_melt = df_melt.drop(columns=['period'])
        return df_melt

    df_n_current = melt_df(neighbor_current_sum, 'neighbor_outbreak_count_current_month')
    df_n_past3 = melt_df(neighbor_past3_sum, 'neighbor_outbreak_count_past_3_months')
    df_n_past6 = melt_df(neighbor_past6_sum, 'neighbor_outbreak_count_past_6_months')
    
    # Merge neighbor outbreak sums
    df_n = df_n_current.merge(df_n_past3, on=['csd_uid', 'year', 'month'])
    df_n = df_n.merge(df_n_past6, on=['csd_uid', 'year', 'month'])
    
    # Map neighbor counts
    df_n['neighbor_count'] = df_n['csd_uid'].map(neighbor_counts)
    
    # Calculate rates
    df_n['neighbor_outbreak_rate_current_month'] = np.where(
        df_n['neighbor_count'] > 0,
        df_n['neighbor_outbreak_count_current_month'] / df_n['neighbor_count'],
        0.0
    )
    df_n['neighbor_outbreak_rate_past_3_months'] = np.where(
        df_n['neighbor_count'] > 0,
        df_n['neighbor_outbreak_count_past_3_months'] / (df_n['neighbor_count'] * 3.0),
        0.0
    )

    # Final Merge with model table
    df_final = df_model.merge(df_n, on=['csd_uid', 'year', 'month'])
    
    # Cleanup helper columns
    df_final = df_final.drop(columns=['period', 'outbreaks_current', 'outbreaks_past3', 'outbreaks_past6'])
    
    # Save the output CSV
    os.makedirs(os.path.dirname(out_table_path), exist_ok=True)
    df_final.to_csv(out_table_path, index=False)
    print(f"Saved enriched modeling table to: {out_table_path}")

    # QA Report
    print("\n=======================================================")
    print("                     QA REPORT")
    print("=======================================================")
    print(f"Input Row Count:                      {input_row_count}")
    print(f"Output Row Count:                     {len(df_final)}")
    print(f"Number of Unique CSDs:                {df_final['csd_uid'].nunique()}")
    print(f"Number of Adjacency Edges:            {len(df_edges)}")
    
    dups = df_final.duplicated(subset=['csd_uid', 'year', 'month']).sum()
    print(f"Duplicate Key (csd/year/month) Count: {dups}")
    
    new_features = [
        'neighbor_outbreak_count_current_month',
        'neighbor_outbreak_count_past_3_months',
        'neighbor_outbreak_count_past_6_months',
        'neighbor_outbreak_rate_current_month',
        'neighbor_outbreak_rate_past_3_months',
        'neighbor_count'
    ]
    
    print("\nSummary Statistics of New Features:")
    for f in new_features:
        f_min = df_final[f].min()
        f_max = df_final[f].max()
        f_mean = df_final[f].mean()
        corr = df_final[f].corr(df_final['outbreak_next_month'])
        print(f"  {f:40} | Min: {f_min:.4f} | Max: {f_max:.4f} | Mean: {f_mean:.4f} | Corr with Target: {corr:.4f}")
    print("=======================================================")

if __name__ == '__main__':
    main()
