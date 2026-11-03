# Spatial Graph Learning for Predicting Avian Influenza Outbreak Risk in Canadian Poultry

This repository provides the official source code for **Spatial Graph Learning for Predicting Avian Influenza Outbreak Risk in Canadian Poultry**.

The framework implements a spatiotemporal Graph Neural Network based on **GraphSAGE** to forecast one-month-ahead poultry Highly Pathogenic Avian Influenza (HPAI) outbreak risk across 5,054 Canadian Census Subdivisions (CSDs). The pipeline incorporates 62 standardized predictors spanning domestic poultry host demographics, satellite-derived land cover and surface hydrology, meteorological reanalysis, road transportation infrastructure, and wild-bird H5N1 biosurveillance.

---

## Repository Structure

```text
├── src/
│   └── hpai_gnn/
│       ├── data/          # Feature engineering, boundary processing, and dataset assembly
│       ├── models/        # GraphSAGE architectures, GNN variants, and tabular baselines
│       ├── graph/         # Spatial proximity graph construction (kNN-3, contiguity, radius)
│       ├── training/      # Training loops, loss functions, and optimization routines
│       └── evaluation/    # Metric computation (AP, ROC-AUC, PR curves, Brier score)
├── scripts/               # Model execution, evaluation benchmarks, and figure generation
│   ├── run_graphsage_stage2_pipeline.py  # Primary GraphSAGE training pipeline
│   ├── evaluate_table3_1to5.py           # Controlled 1:5 pseudo-absence test benchmark
│   ├── run_stage5_bootstrap.py           # 1,000-iteration bootstrap confidence intervals
│   ├── train_eval_flyway_dynamic_graphsage.py # Seasonal flyway dynamic graph ablation
│   └── plot_*.py                         # Reproduction scripts for all manuscript figures
├── configs/               # Hyperparameter and environment configurations
├── pyproject.toml         # Package setup and build specification
├── requirements.txt       # Core Python dependencies
├── LICENSE                # MIT License
└── README.md
```

---

## Methodology & Architecture

1. **Spatial Proximity Graph:** Connects 5,054 Canadian CSD centroids using a symmetrized $k$-nearest neighbors ($k\text{NN}-3$) formulation under geodesic distances (9,655 undirected edges; mean degree 3.82, maximum degree 8), guaranteeing continuous nationwide connectivity across island and coastal agricultural regions.
2. **Spatiotemporal Feature Manifest:** 62 standardized features across 7 functional domains:
   * Host poultry demographics (12 features)
   * Land cover & surface hydrology (13 features; Dynamic World 10-m)
   * Weather & climate conditions (6 features; ERA5-Land reanalysis)
   * Road transportation infrastructure (10 features; National Road Network)
   * Geographic extent & urbanization (2 features)
   * Historical poultry outbreak lags (7 features; CFIA declarations)
   * Wild-bird HPAI biosurveillance (12 features; CWHC detections)
3. **Prospective Forward-Time Evaluation:**
   * **Development/Training:** January 2022 – December 2023 (121,296 CSD-months)
   * **Prospective Validation:** January 2024 – December 2024 (60,648 CSD-months)
   * **Locked Prospective Test Benchmark:** January 2025 – November 2025 (55,594 CSD-months; evaluated over 50 Monte Carlo resamples at 1:5 case-to-pseudo-absence ratio)

---

## Installation

### Prerequisites
* Python $\ge$ 3.10
* PyTorch $\ge$ 2.0
* PyTorch Geometric (PyG) $\ge$ 2.3

### Environment Setup

```bash
# Clone repository
git clone https://github.com/minakhoshbazm/hpai-heterogeneous-gnn-canada.git
cd hpai-heterogeneous-gnn-canada

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install package and dependencies
pip install -r requirements.txt
pip install -e .
```

---

## Quickstart & Replication

### 1. Train Primary GraphSAGE Model
Train the canonical 2-layer GraphSAGE architecture with mean neighborhood aggregation and unweighted cross-entropy loss:
```bash
python scripts/run_graphsage_stage2_pipeline.py
```

### 2. Benchmark Against Baselines
Evaluate GraphSAGE against matched MLP, Random Forest, XGBoost, and Logistic Regression on the locked prospective test set:
```bash
python scripts/evaluate_table3_1to5.py
```

### 3. Run Bootstrap Resampling & Significance Tests
Compute non-parametric 1,000-iteration bootstrap distributions and confidence intervals:
```bash
python scripts/run_stage5_bootstrap.py
```


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
