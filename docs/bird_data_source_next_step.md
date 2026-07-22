# Bird Data Source Next Step

## Context

Our current models incorporate environmental features (climate, land cover), poultry/static features, outbreak history, spatial adjacency, and neighbor outbreak pressure. The next major ecological component missing from the feature space is wild bird activity or abundance. Incorporating wild bird metrics may improve the model by better representing the ecological vector dynamics of HPAI spread.

## Preferred Source: eBird Status and Trends

- **Research-Quality**: eBird Status and Trends is the preferred data source for final bird features.
- **Model-Derived**: It provides weekly relative abundance and range products derived from statistical models that correct for observer-reporting bias.
- **Spatio-Temporal Aggregation**: These modeled weekly outputs can be cleanly aggregated to the CSD-month level, preserving seasonality and spatial boundaries.
- **Access Pipeline**: Because access requires request approvals and downloading large raster datasets, it may take longer to set up.

## Backup/Pilot Source: GBIF

- **Low Friction**: GBIF is easy, fast, and open-access via API queries.
- **Proven Utility**: It served as the primary data source for the Ontario waterfowl feature pilot.
- **Activity Proxy**: Due to bias in observer effort (reporting is denser near towns and parks), GBIF counts should be treated as wild bird activity/reporting proxies rather than true abundance.
- **Fallback Capability**: If eBird access is delayed, GBIF remains our primary option to pilot the bird feature pipeline.

## Candidate Species or Groups

The modeling will focus on wild bird species known to be key reservoirs or vectors for HPAI:
- Canada Goose
- Snow Goose
- Mallard
- Northern Pintail
- American Wigeon
- Lesser Scaup / Greater Scaup (if available)
- Broader waterfowl group (if species-specific products are incomplete)

## Planned Bird Features

We plan to construct the following nodes features for each CSD-month:
- Monthly waterfowl abundance/activity
- Duck abundance/activity
- Goose abundance/activity
- Spring migration indicator
- Autumn migration indicator
- Bird activity x water area interaction
- Bird activity x seasonal pond proxy interaction
- Neighbor bird activity mean
- Neighbor bird activity sum

## Modeling Hypothesis

Integrating bird features should improve both tabular and GNN classifiers. We hypothesize that GNNs will show a larger improvement because they can propagate bird habitat risk signals from adjacent CSDs (e.g. wetlands) to neighboring poultry production nodes, capturing trans-boundary vector risk.

## Decision

The next practical step is to verify eBird Status and Trends access and download feasibility. If the download pipeline is successfully established, eBird will be used for final bird features. If eBird access is delayed, we will proceed with GBIF counts as our pilot/proxy source to avoid stalling implementation.
