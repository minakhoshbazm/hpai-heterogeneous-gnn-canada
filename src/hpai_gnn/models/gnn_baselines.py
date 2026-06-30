"""
Homogeneous Spatial GNN model architectures (GCN, GraphSAGE, GAT).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class SpatialGNN(nn.Module):
    """
    A configurable Homogeneous Spatial GNN model.
    Supports GCN, GraphSAGE, and GAT layer types.
    """
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=2, 
                 model_type='GCN', dropout=0.5, heads=1):
        super(SpatialGNN, self).__init__()
        
        self.model_type = model_type.upper()
        self.num_layers = num_layers
        self.dropout = dropout
        
        if self.model_type not in ['GCN', 'SAGE', 'GAT']:
            raise ValueError(f"Unknown GNN model type: {model_type}. Choose from 'GCN', 'SAGE', 'GAT'.")
            
        self.convs = nn.ModuleList()
        
        # Build conv layers
        for i in range(num_layers):
            # Input dimension calculation
            if i == 0:
                in_dim = in_channels
            else:
                if self.model_type == 'GAT':
                    in_dim = hidden_channels * heads
                else:
                    in_dim = hidden_channels
                    
            # Output dimension calculation
            out_dim = hidden_channels
            
            # Layer instantiation
            if self.model_type == 'GCN':
                conv = GCNConv(in_dim, out_dim)
            elif self.model_type == 'SAGE':
                conv = SAGEConv(in_dim, out_dim)
            elif self.model_type == 'GAT':
                # Concat heads for intermediate layers, average (concat=False) for the final conv layer
                is_last_conv = (i == num_layers - 1)
                conv = GATConv(in_dim, out_dim, heads=heads, concat=not is_last_conv)
                
            self.convs.append(conv)
            
        # Final linear layer to project node features to outbreak logits
        # For GAT, if the last conv has concat=False, output dimension is hidden_channels.
        # If the last conv has concat=True, output dimension is hidden_channels * heads.
        # In our setup, the last conv has concat=False, so its output is hidden_channels.
        self.post_project = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass of the spatial GNN.
        
        Parameters:
        ----------
        x : torch.Tensor
            Node features of shape [N, in_channels]
        edge_index : torch.Tensor
            Graph edge indices of shape [2, E]
        edge_weight : torch.Tensor, optional
            Edge weights of shape [E, 1]
            
        Returns:
        -------
        logits : torch.Tensor
            Outbreak logits of shape [N, out_channels]
        """
        for i, conv in enumerate(self.convs):
            if self.model_type == 'GCN':
                x = conv(x, edge_index, edge_weight=edge_weight)
            elif self.model_type == 'GAT':
                # GAT Conv takes edge_attr (e.g. edge weights)
                edge_attr = edge_weight.squeeze(-1) if edge_weight is not None else None
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:  # SAGE
                x = conv(x, edge_index)
                
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        # Project output to target logit dimension
        logits = self.post_project(x)
        return logits
