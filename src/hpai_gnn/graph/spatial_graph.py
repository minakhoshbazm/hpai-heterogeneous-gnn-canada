"""
Spatial graph construction utilities for HPAI GNN forecasting.
"""

import os
import geopandas as gpd
import numpy as np
import pandas as pd
import torch

def build_spatial_graph(gpkg_path, sorted_cd_uids=None, inverse_distance_weights=False):
    """
    Builds a static region-region spatial graph of Canada's Census Divisions using queen contiguity.
    
    Parameters:
    ----------
    gpkg_path : str
        Path to the geopackage boundary file (census_divisions_epsg3347.gpkg).
    sorted_cd_uids : list or np.ndarray, optional
        Desired node ordering of CD UIDs. If None, nodes are sorted alphabetically by cd_uid.
    inverse_distance_weights : bool, default False
        If True, computes edge weights based on inverse centroid-to-centroid distances.
        If False, edge weights are set to 1.0.
        
    Returns:
    -------
    edge_index : torch.Tensor
        LongTensor of shape [2, E] representing graph edges.
    edge_attr : torch.Tensor
        FloatTensor of shape [E, 1] representing edge weights.
    node_to_idx : dict
        Mapping from cd_uid (str) to node index (int).
    """
    if not os.path.exists(gpkg_path):
        raise FileNotFoundError(f"Boundary geopackage not found at: {gpkg_path}")
        
    gdf = gpd.read_file(gpkg_path)
    gdf['cd_uid'] = gdf['cd_uid'].astype(str)
    
    # Ensure consistent node ordering
    if sorted_cd_uids is not None:
        sorted_cd_uids = [str(uid) for uid in sorted_cd_uids]
        gdf = gdf.set_index('cd_uid').reindex(sorted_cd_uids).reset_index()
    else:
        gdf = gdf.sort_values(by='cd_uid').reset_index(drop=True)
        
    node_to_idx = {row['cd_uid']: idx for idx, row in gdf.iterrows()}
    
    # Create temporary column for source and target matching
    gdf_temp = gdf.copy()
    gdf_temp['node_idx'] = gdf_temp.index
    
    # Perform a spatial join using the 'touches' predicate (Queen contiguity)
    # Queen contiguity means sharing a boundary point or segment without interior overlap.
    joined = gpd.sjoin(gdf_temp, gdf_temp, predicate='touches', how='inner')
    
    edge_list = []
    edge_weights = []
    
    # Extract edge indices (left = source, right = target)
    for _, row in joined.iterrows():
        u = int(row['node_idx_left'])
        v = int(row['node_idx_right'])
        
        # Exclude self-loops (touches should not return self-loops, but we enforce it)
        if u == v:
            continue
            
        edge_list.append((u, v))
        
    # Optional inverse distance weights using centroids in EPSG:3347 (meters)
    if inverse_distance_weights and len(edge_list) > 0:
        centroids = gdf.geometry.centroid
        for u, v in edge_list:
            dist = centroids.iloc[u].distance(centroids.iloc[v])
            # Avoid division by zero
            weight = 1.0 / (dist + 1.0)
            edge_weights.append(weight)
            
        edge_attr_np = np.array(edge_weights, dtype=np.float32)
        # Normalize weights so that the maximum weight is 1.0
        max_w = edge_attr_np.max()
        if max_w > 0:
            edge_attr_np = edge_attr_np / max_w
        edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32).unsqueeze(1)
    else:
        edge_attr = torch.ones((len(edge_list), 1), dtype=torch.float32)
        
    # Convert edge index list to PyTorch tensor
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
        
    return edge_index, edge_attr, node_to_idx
