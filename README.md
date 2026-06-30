# HPAI Heterogeneous GNN Canada

A Heterogeneous Spatiotemporal Graph Neural Network Integrating Wild Bird Migrations for Avian Influenza Risk in Canada.

This repository implements a machine learning pipeline using heterogeneous spatiotemporal graph neural networks to predict one-month-ahead Highly Pathogenic Avian Influenza (HPAI) poultry outbreak risk across Canadian regions.

## Project Overview

* **Goal:** Build an automated, reproducible ML pipeline for poultry HPAI outbreak prediction in Canada.
* **Target:** Binary prediction: Will at least one poultry HPAI outbreak occur in region $r$ during month $t+1$, using only data available up to month $t$?
* **Spatial Resolution:** Canadian Census Divisions (CD) or Census Agricultural Regions (CAR) as the primary unit, with provinces as a fallback.
* **Temporal Resolution:** Monthly prediction steps.

## Graph Concept

The model leverages a staged multi-layer heterogeneous spatiotemporal graph containing several layers:
1. **Region Layer:** Spatial nodes (Census Divisions/CARs) representing geographic units. Features include land cover, environmental anomalies, and road density.
2. **Species Layer:** Nodes representing key migratory waterfowl and avian influenza reservoir species (e.g., Mallard, Canada Goose).
3. **Region-Species Edges:** Captures species abundance and migration intensity per region per month.
4. **Transport Layer:** Road networks integrated via node features or region-region spatial connectivity edges.

## Staged Roadmap

* **Stage 1 (Tabular Baselines):** Logistic Regression, Random Forest, XGBoost/LightGBM.
* **Stage 2 (Spatial Graph Baselines):** Homogeneous region GNNs (GCN, GraphSAGE, GAT).
* **Stage 3 (Heterogeneous Graph):** Region-species bipartite graphs using R-GCN, HeteroGraphSAGE, or HeteroGAT.
* **Stage 4 (Temporal Model):** Combining heterogeneous spatial embeddings with GRU/LSTM/Attention encoders.
* **Stage 5 (Advanced Model):** Heterogeneous Graph Transformers, multi-task learning (predicting both poultry outbreaks and wild bird detections), and transfer learning.

## Workflow

* **Local Machine:** Planning, lightweight preprocessing, package management, and running local smoke tests.
* **Google Earth Engine (GEE):** Scale extraction of dynamic environmental covariates (NDVI, NDWI, Dynamic World land cover, surface water).
* **Compute Canada (Alliance Canada):** GPU cluster environment for large-scale data ingestion, heterogeneous graph construction, hyperparameter optimization, and temporal GNN model training.

## Current Project Status

> [!NOTE]
> This project is currently in the **Scaffold & Configuration** phase. All package directories, configuration templates, basic cluster Slurm templates, and documentation are set up. No real-world datasets have been downloaded, and no model training has started yet.