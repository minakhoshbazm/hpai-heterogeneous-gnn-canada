# Neighbor Outbreak Pressure Modeling Result

## Purpose

Neighbor outbreak pressure features were added to test whether recent outbreaks in adjacent CSDs improve prediction. These features represent spatial outbreak activity in surrounding CSD regions.

## Features Added

The following spatial variables were constructed and added to the dataset:
- `neighbor_outbreak_count_current_month`
- `neighbor_outbreak_count_past_3_months`
- `neighbor_outbreak_count_past_6_months`
- `neighbor_outbreak_rate_current_month`
- `neighbor_outbreak_rate_past_3_months`
- `neighbor_count`

## Tabular Model Result

Adding neighbor outbreak pressure features to the baseline models produced the following evaluations:
- **Logistic Regression**: Improved slightly, raising the PR-AUC from 0.0635 to 0.0653.
- **Random Forest**: Improved more clearly, raising the PR-AUC from 0.0308 to 0.0490.
- **Precision@50**: Random Forest Precision@50 improved significantly from 0.060 to 0.180.

### Interpretation
Neighbor-pressure features are useful for tabular models because they provide spatial context (adjacent outbreak status) that tabular classifiers cannot otherwise learn directly.

## Graph Model Result

- **GraphSAGE Baseline (No neighbor pressure)**: Had a mean PR-AUC of 0.0735.
- **GraphSAGE Enriched (With neighbor pressure)**: Had a mean PR-AUC of 0.0742.
- **Comparison**: This represents a marginal improvement (+0.0007 mean PR-AUC).

### Interpretation
Neighbor-pressure features add only marginal value to GraphSAGE, likely because GNNs inherently capture and propagate this neighborhood information through spatial message passing across the edges.

## Decision

- Keep neighbor-pressure features as useful engineered spatial features in the data pipeline.
- They are especially useful for non-graph (tabular) models to provide baseline comparison parity.
- For GNN models, they are optional due to the redundant spatial information already captured via message passing.
- The next major model improvement likely requires stronger ecological features, particularly wild bird abundance or migration flyway datasets.
