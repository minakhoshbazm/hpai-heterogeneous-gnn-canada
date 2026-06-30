# Literature-Informed GNN Design for HPAI

Our design is grounded in recent advancements in spatial epidemiology, network ecology, and deep graph representation learning. This document outlines the key literary pillars inspiring the architecture.

---

## 1. Heterogeneous Graph Representation Learning
Traditional Graph Neural Networks (GCN, GraphSAGE) operate on homogeneous graphs (where all nodes are of the same type). However, HPAI transmission involves distinct entities: geographic administrative regions, diverse wild bird species, and outbreaks.
* **Inspiration:** Modeling spatial units and ecological species as separate node classes with relational edges ($R \to R$, $S \to R$) allows relational message passing (e.g., using **R-GCN** or **Heterogeneous Graph Transformer (HGT)**). This prevents loss of identity that occurs when flattening species covariates directly into regional tabular features.
* **Key Paper Inspiration:** *Heterogeneous Graph Attention Networks* (Wang et al., 2019) and *Relational Graph Convolutional Networks* (Schlichtkrull et al., 2018).

---

## 2. eBird Citizen-Science and Migration Networks
eBird status and trends provide high-resolution, model-derived abundance metrics for migratory waterfowl across North America.
* **Inspiration:** By integrating eBird grid predictions, we capture the dynamic monthly fluctuations of specific avian host populations inside Canadian regions. This represents an improvement over static bird distribution maps, allowing the graph edges ($S \to R$) to update dynamically over time to reflect migration corridors and seasonal stopovers.
* **Key Paper Inspiration:** *eBird: Quantifying bird distribution and abundance at global scales* (Fink et al., 2020).

---

## 3. Remote Sensing Ecological Predictors
Waterfowl aggregation is heavily dictated by surface water availability, temperature, and vegetation health.
* **Inspiration:** Sentinel-2, ERA5-Land, and JRC Global Surface Water data extracted via Google Earth Engine represent the environmental context. Standard features include NDVI (Normalized Difference Vegetation Index) anomalies to track drought or vegetation flushes, and NDWI (Normalized Difference Water Index) to capture soil wetness and open surface water.
* **Key Paper Inspiration:** *Remote sensing of environmental covariates for spatial modeling of infectious diseases* (Rizzoli et al., 2015).

---

## 4. Spatiotemporal GNNs (ST-GNN)
Epidemiological risk is spatiotemporally auto-correlated. Outbreaks are influenced by past states of neighboring regions and the temporal momentum of bird migrations.
* **Inspiration:** We employ a two-stage encoder-decoder. First, a spatial Heterogeneous GNN aggregates local attributes and wild bird associations for each month $t$ to produce a spatial embedding:
  $$H_t = \text{HeteroGNN}(\mathcal{G}_t)$$
  Second, a sequential network (e.g., **LSTM/GRU** or **Temporal Attention**) aggregates these embeddings over time to forecast the next-month risk:
  $$\hat{Y}_{t+1} = \text{TemporalModel}(H_1, H_2, \dots, H_t)$$
* **Key Paper Inspiration:** *Spatiotemporal Multi-Graph Convolutional Networks for State-level Influenza Forecasting* (Cola et al., 2020).

---

## 5. Animal Movement and Telemetry Priors
GPX/telemetry networks from Movebank and Motus provide real-world animal tracking tracks.
* **Inspiration:** Although telemetry data is sparse, it contains high-precision information about flight paths, stopover durations, and flyway connectivity. We utilize these patterns to construct region-to-region migratory connectivity weights (prior probability matrices of bird movements between Census Divisions).

---

## 6. Genomic-Informed Graph Models (Optional V3)
Recent computational epidemiology models use viral genomic similarity to construct disease transmission graphs.
* **Inspiration:** If virus sequence data becomes available, we can model viral clades and phylogenetic distances as a graph network layer. This allows the model to learn representations of genetic drift, reassortment, and transmission path clusters alongside geographical routes.
* **Key Paper Inspiration:** *Genomic-informed GNNs for viral transmission tracking* (various recent bioinformatics preprints).
