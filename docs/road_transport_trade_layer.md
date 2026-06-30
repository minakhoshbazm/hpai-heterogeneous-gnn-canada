# Road, Transport, and Poultry Trade Layers

This document details the design and integration of transport networks and poultry trade/movement data into the Canadian HPAI spatiotemporal graph model.

---

## 1. Why Roads Matter for HPAI Risk
Epidemiological studies indicate that the spread of HPAI between poultry farms is not purely driven by wild bird vectors. Anthropogenic factors play a critical role:
* **Fecal-Oral Transmission via Vehicles:** Feed trucks, slaughterhouse transport vehicles, and catching crews moving between farms can mechanical transfer virus-laden organic matter.
* **Road Proximity:** Farms located close to major highways or regional trade corridors have a higher probability of exposure to vehicle-borne dust or shared service logistics.

---

## 2. Integrating Road Networks into the GNN
We can integrate road data from sources like OpenStreetMap (OSM) or Statistics Canada's Road Network Files (RNF) in three distinct ways:

> [!IMPORTANT]
> **CRS Requirement:** All metrics involving road length, area calculations, distances to highways, and centroid proximities must be calculated in the project's conformal projected CRS (**`EPSG:3347`**, NAD83 / Statistics Canada Lambert). Calculating distance or density using geographic coordinate degrees (`EPSG:4326`) is prohibited.

### A. As Region Node Features ($X_R$)
* **Road Density:** Total length of roads (e.g., in kilometers) divided by the total area of the Census Division ($km / km^2$). Can be split into major highways vs. local roads.
* **Transport Centroid Proximity:** Shortest distance from the centroid of a Census Division to the nearest major highway or transit node.

### B. As Region-Region Edge Weights ($W_{E}$)
* **Road Distance Graph:** Instead of using simple geographic Euclidean distance between region centroids, we can compute the shortest driving path distance (in kilometers) or travel time (in minutes) over the road network.
* **Connectivity Metric:** The edge weight between region $i$ and region $j$ can be defined as:
  $$W_{ij} = \frac{1}{\text{TravelTime}(i, j)}$$

---

## 3. Poultry Movement & Trade Data

### The Ideal Dataset
The gold standard for representing mechanical transmission vectors is **live animal permitted movement records**:
* **Granularity:** Farm-to-farm, farm-to-processing-facility, or farm-to-slaughterhouse transactions.
* **Temporal scale:** Daily or weekly logs indicating how many birds were moved, their origin, destination, and purpose.

### Movement-Proxy Features (When Ideal Data is Unavailable)
Since detailed farm-to-farm trade transactions are highly confidential and rarely accessible, we build **proxies** using accessible datasets:
1. **Gravity Model Connectivity:** Standard spatial gravity models estimate the likelihood of movement between region $i$ and region $j$ based on farm density and distance:
   $$T_{ij} \propto \frac{\text{PoultryCount}_i \times \text{PoultryCount}_j}{\text{TravelTime}(i, j)^2}$$
2. **Density Cross-Product:** Edge weights between regions can be defined as a function of their road connectivity and relative poultry density, capturing the probability of shared transport services.
3. **Distance to Processing Facilities:** Node feature representing the travel distance to the nearest regional poultry slaughterhouse or packaging plant.

### Why Trade/Genomic Data is Kept Optional
* **Access Restraints:** Live poultry trade registries are protected under strict federal/provincial privacy laws. 
* **Sparsity:** Detailed movements may not be available for all provinces or time steps.
* **Staged Inclusion:** Keeping trade and movement data optional ensures that our core V1 model remains fully operational and testable using publicly accessible ecological and geographical datasets.
