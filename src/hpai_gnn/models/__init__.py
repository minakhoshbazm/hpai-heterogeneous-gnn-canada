"""Model architectures including tabular baselines, GNNs, and spatiotemporal networks."""

from hpai_gnn.models.baselines import get_baseline_model
from hpai_gnn.models.gnn_baselines import SpatialGNN

__all__ = ["get_baseline_model", "SpatialGNN"]

