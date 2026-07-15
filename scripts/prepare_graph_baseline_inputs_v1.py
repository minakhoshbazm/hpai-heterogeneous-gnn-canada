#!/usr/bin/env python3
"""
scripts/prepare_graph_baseline_inputs_v1.py

Validates and prepares graph baseline inputs:
1. Filters the CSD adjacency edges to keep only nodes that exist in the modeling table.
2. Checks and removes self-loops.
3. Dedupes undirected edges (ensures canonical ordering).
4. Saves a clean filtered edge list.
"""

import os
import pandas as pd
import numpy as np

def main():
    modeling_table_path = 'data/processed/modeling/model_table_v1_environment_with_lags.csv'
    edges_path = 'data/processed/graph/csd_adjacency_edges.csv'
    out_edges_path = 'data/processed/graph/csd_adjacency_edges_v1_model_scope.csv'

    print("1. Loading datasets...")
    if not os.path.exists(modeling_table_path):
        raise FileNotFoundError(f"Modeling table not found at: {modeling_table_path}")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Edges file not found at: {edges_path}")

    df_model = pd.read_csv(modeling_table_path)
    df_edges = pd.read_csv(edges_path)

    # 2. Confirm columns in modeling table
    required_model_cols = ['csd_uid', 'year', 'month', 'outbreak_next_month']
    for col in required_model_cols:
        if col not in df_model.columns:
            raise KeyError(f"Required column '{col}' missing from modeling table.")
    print("Modeling table columns validated successfully.")

    # 3. Confirm columns in edge list
    required_edge_cols = ['source_csd_uid', 'target_csd_uid']
    for col in required_edge_cols:
        if col not in df_edges.columns:
            raise KeyError(f"Required column '{col}' missing from edge list.")
    print("Edge list columns validated successfully.")

    # Get sets of CSD IDs
    model_csd_set = set(df_model['csd_uid'].unique())
    orig_edge_csd_set = set(df_edges['source_csd_uid']).union(set(df_edges['target_csd_uid']))

    # 4. Filter edge list
    orig_edge_count = len(df_edges)
    
    # Check self loops in original
    orig_self_loops = (df_edges['source_csd_uid'] == df_edges['target_csd_uid']).sum()
    print(f"Original self-loops: {orig_self_loops}")

    # Remove self-loops and filter to modeling table scope
    df_filtered = df_edges[
        (df_edges['source_csd_uid'] != df_edges['target_csd_uid']) &
        (df_edges['source_csd_uid'].isin(model_csd_set)) &
        (df_edges['target_csd_uid'].isin(model_csd_set))
    ].copy()

    # Deduplicate undirected edges: sort each row's source and target to enforce canonical order
    # source < target
    srcs = df_filtered['source_csd_uid'].values
    dsts = df_filtered['target_csd_uid'].values
    canon_srcs = np.minimum(srcs, dsts)
    canon_dsts = np.maximum(srcs, dsts)
    
    df_filtered['canon_src'] = canon_srcs
    df_filtered['canon_dst'] = canon_dsts
    
    df_filtered = df_filtered.drop_duplicates(subset=['canon_src', 'canon_dst'])
    
    # Replace columns with canonical and drop helper columns
    df_final_edges = pd.DataFrame({
        'source_csd_uid': df_filtered['canon_src'],
        'target_csd_uid': df_filtered['canon_dst']
    }).reset_index(drop=True)

    filtered_edge_count = len(df_final_edges)
    filtered_edge_csd_set = set(df_final_edges['source_csd_uid']).union(set(df_final_edges['target_csd_uid']))

    # 5. Print metrics
    dropped_edges = orig_edge_count - filtered_edge_count
    dropped_csds = len(orig_edge_csd_set - filtered_edge_csd_set)

    print("\n--- Summary Metrics ---")
    print(f"Original Edge Count:                  {orig_edge_count}")
    print(f"Filtered Edge Count (Undirected):     {filtered_edge_count}")
    print(f"Dropped Edges:                        {dropped_edges}")
    print(f"Original Unique Edge CSD Count:       {len(orig_edge_csd_set)}")
    print(f"Filtered Unique Edge CSD Count:       {len(filtered_edge_csd_set)}")
    print(f"Dropped CSD IDs:                      {dropped_csds}")
    print(f"Modeling Table Unique CSD Count:      {len(model_csd_set)}")

    # 7. Verification: check self-loops and duplicate undirected edges
    final_self_loops = (df_final_edges['source_csd_uid'] == df_final_edges['target_csd_uid']).sum()
    
    # Check duplicate undirected pairs
    dups_check = df_final_edges.duplicated(subset=['source_csd_uid', 'target_csd_uid']).sum()

    print("\n--- Output Verification ---")
    print(f"Final self-loops:                     {final_self_loops}")
    print(f"Final duplicate undirected edges:     {dups_check}")
    
    assert final_self_loops == 0, "Error: Self-loops found in final output!"
    assert dups_check == 0, "Error: Duplicate undirected edges found in final output!"

    # 6. Save clean filtered file
    os.makedirs(os.path.dirname(out_edges_path), exist_ok=True)
    df_final_edges.to_csv(out_edges_path, index=False)
    print(f"\nSaved filtered edge file to: {out_edges_path}")

if __name__ == '__main__':
    main()
