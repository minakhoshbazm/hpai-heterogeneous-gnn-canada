# Bird-Enhanced Modeling Plan

## Purpose

The current spatial graph model using environmental, poultry, geography, and outbreak-history features provides a modest improvement in outbreak prediction. Our next hypothesis is that adding wild bird activity or migration-related features will strengthen the GNN's predictive capability. This is because bird movements, habitat interactions, and transmission risks are spatially coupled across neighboring Census Subdivisions (CSDs), which a graph neural network may be well suited to model.

## Why Bird Features Matter

Wild bird features are critical for modeling HPAI outbreaks for several reasons:
- **Important Reservoirs and Vectors**: HPAI risk is strongly linked to wild waterfowl and migratory birds, which can carry and spread the virus.
- **Trans-boundary Movement**: Wild birds do not respect administrative CSD boundaries; their flight paths and feeding territories naturally bridge neighboring nodes.
- **Buffer Effects**: Wetlands, water bodies, and migration flyways in one CSD can influence nearby poultry production facilities in neighboring CSDs.
- **Relational Modeling**: GNNs can propagate these bird presence and habitat signals across adjacency edges, allowing the model to learn spatial risk buffers better than non-graph models.

## Candidate Bird-Related Features

The following list of features will be evaluated for incorporation into the node feature matrix:
- `gbif_waterfowl_count`: Total observations of waterfowl.
- `gbif_duck_count`: Total observations of duck species.
- `gbif_goose_count`: Total observations of goose species.
- `gbif_total_bird_count`: Total wild bird records.
- `gbif_waterfowl_share`: Ratio of waterfowl to total bird observations.
- `gbif_reporting_intensity`: Total records to control for observer bias.
- `spring_migration_month`: Indicator for seasonal migration peaks.
- `autumn_migration_month`: Indicator for autumn migration peaks.
- `waterfowl_activity_x_water`: Interaction term of waterfowl counts and water area.
- `waterfowl_activity_x_seasonal_pond_proxy`: Interaction term of waterfowl and dynamic water proxies.
- `neighbor_waterfowl_activity_mean`: Mean waterfowl activity in neighboring CSDs.
- `neighbor_waterfowl_activity_sum`: Sum of waterfowl activity in neighboring CSDs.

## Data Source Options

We compare three primary data sources to capture wild bird distributions:
1. **eBird Status and Trends**:
   - *Pros*: Provides modeled relative abundance, which corrects for observer reporting bias.
   - *Cons*: Restrictive API access and requires pre-processing of large raster products.
2. **GBIF (Global Biodiversity Information Facility)**:
   - *Pros*: Readily accessible, open-access, and simple to extract counts.
   - *Cons*: Heavily biased by reporting effort (cities/parks have more entries), meaning it serves as an activity proxy rather than true abundance.
3. **Dynamic World Water/Pond Proxy**:
   - *Pros*: Already processed, objective, and serves as an indirect habitat suitability indicator.
   - *Cons*: Does not reflect actual bird presence or abundance.

## Modeling Hypothesis

- **Logistic Regression**: Adding bird features will improve the performance of non-graph models slightly, but they will be constrained to the local node's values.
- **Graph Neural Networks**: The GNN baseline should show a larger improvement because message-passing allows it to incorporate bird habitat and presence signals from adjacent CSDs. This is particularly valuable when a high-risk habitat (e.g., a wetland) is located next to a poultry-producing CSD.

## Planned Experiment

We will conduct experiments comparing:
1. Current environment + lag logistic regression.
2. Environment + lag + bird logistic regression.
3. Stabilized GraphSAGE baseline (with early stopping).
4. GraphSAGE model with bird-enhanced features.

All evaluations will be conducted on the same, unchanged 2025 test set.

## Cautions

- **Observation Bias**: GBIF coordinates show where birdwatchers go, not the complete species distribution. We must normalize counts or control for reporting intensity.
- **Temporal Leakage**: Features representing month M must only use bird records collected in month M or earlier. No future information can be used.
- **Quality Control**: GBIF taxonomies must be filtered strictly to target wild waterfowl species known to carry HPAI to avoid diluting the feature set.
