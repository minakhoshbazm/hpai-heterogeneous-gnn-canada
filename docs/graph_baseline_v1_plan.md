# Graph Baseline V1 Plan

## Goal
Train a lightweight graph-based baseline model to predict **one-month-ahead HPAI outbreak risk** (target `outbreak_next_month`). The model will use the existing **CSD adjacency graph** to share information between neighboring CSDs while leveraging the **environment + lag features** already prepared.

## Data Inputs
- **Feature table**: `data/processed/modeling/model_table_v1_environment_with_lags.csv`
  - Contains one row per CSD-month with environmental variables and lagged outbreak/history features.
- **Adjacency edge list**: Use the existing processed CSD adjacency edge list. Before implementation, locate the exact edge file path under `data/processed/` and verify it contains `source_csd_uid` and `target_csd_uid` or equivalent source/target columns.
- **Target**: `outbreak_next_month` column in the feature table.

## Modeling Approach
### Graph Architecture
- **Model type**: start with a simple **GraphSAGE** (or GCN) layer followed by a linear read-out.
- **Node features**: concatenate all environment and lag features for each CSD-month. No additional temporal edges – the same static spatial adjacency is used for every month.
- **Edge index**: undirected CSD adjacency from the edge list.
- **Temporal handling**: create a separate graph snapshot for each month (i.e., batch of node feature rows with the same edge index). The model will be trained on all snapshots jointly, treating the month dimension as independent samples.

### Train / Test Split
- **Train**: all months from **2022-01** to **2024-12**.
- **Test**: all available 2025 rows in the train-ready table; December 2025 may be absent if its next-month target is unavailable.
- **Validation**: optional hold-out of the last 3 months of the training period (2024-10 to 2024-12) for early-stopping.

### Loss & Class Imbalance
- Use **binary cross-entropy** with **positional weighting** (inverse class frequency) or **focal loss** to mitigate the severe class imbalance.

### Evaluation Metrics
- **PR-AUC** (primary, due to imbalance)
- **ROC-AUC**
- **Precision@50**, **Recall@50**, **Precision@100**, **Recall@100**, **Precision@200**, **Recall@200**
- Compare against three baselines:
  1. Environment-only logistic regression (no lags, no graph).
  2. Time-persistence baseline (predict outbreak if it occurred in the previous month).
  3. Environment + lag logistic regression (no graph).

## Implementation Steps (lightweight, no training yet)
1. **Load data**
   - Read `model_table_v1_environment_with_lags.csv`.
   - Load the adjacency edge list (`csd_adj_edges.csv`).
2. **Construct PyTorch Geometric data objects**
   - For each month, create a `Data(x=node_features, edge_index=adjacency, y=target)` object.
   - Stack all month snapshots into a `Dataset`.
3. **Define simple GraphSAGE model**
   ```python
   class SimpleGraphSAGE(torch.nn.Module):
       def __init__(self, in_dim, hidden_dim=64):
           super().__init__()
           self.conv1 = SAGEConv(in_dim, hidden_dim)
           self.conv2 = SAGEConv(hidden_dim, hidden_dim)
           self.lin = torch.nn.Linear(hidden_dim, 1)
       def forward(self, x, edge_index):
           x = F.relu(self.conv1(x, edge_index))
           x = F.relu(self.conv2(x, edge_index))
           return torch.sigmoid(self.lin(x)).squeeze(-1)
   ```
4. **Create train/val/test splits** based on month indices.
5. **Write a small training script placeholder** (`scripts/train_graph_baseline_v1.py`) that loads the data, instantiates the model, and runs a single epoch (no actual training now).
6. **Add README snippet** in the script explaining how to run the full training later.

## Cautions & Gotchas
- **No future leakage**: ensure that lag features never use data from the target month or later.
- **Static spatial edges only**: temporal relationships are encoded solely via node features; do not add edges between month-copies of the same CSD.
- **Class imbalance**: remember to apply class weighting during training.
- **Feature scaling**: standardize/normalize numeric features before feeding them to the GNN.
- **Graph size**: the CSD graph is relatively small (few hundred nodes), so memory should not be an issue.

## Next Steps (once plan approved)
- Implement the data loading pipeline.
- Finalize the placeholder training script.
- Run a quick sanity-check training run on a subset of months.
- Evaluate against the three baselines and record metrics.

---
*This document is intentionally concise and focuses on the essential steps required to get the first graph-based baseline up and running.*
