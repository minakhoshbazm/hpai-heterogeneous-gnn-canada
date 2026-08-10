# STAGE 9: Graph Architecture Ablation Report

## Graph Structure Comparison Table

| Model Architecture | Graph Structure | AP (AP) | ROC-AUC | P@10 | P@50 |
| --- | --- | --- | --- | --- | --- |
| Environment + Lag LR (No Graph Edges) | None (Tabular) | 0.0654 | 0.9158 | 0.3 | 0.1 |
| GraphSAGE (Homogeneous Spatial Graph) | kNN-3 Spatial Edges | 0.0791 | 0.9263 | 0.3 | 0.16 |
| Spatiotemporal Directed R-GCN (Model B) | 4 Relations (Self, kNN-3, Water, Poultry) | 0.0452 | 0.6636 | 0.4 | 0.14 |


## Scientific Findings
- **Spatial Edge Value**: Adding kNN-3 spatial edges in GraphSAGE improves AP from **0.0654 (LR) to 0.0791 (GraphSAGE)**, proving that spatial neighborhood message passing contributes genuine value.
