# Compute Canada / Alliance Canada Setup Guide

This document outlines how to configure, deploy, and execute the HPAI Heterogeneous GNN project on Digital Research Alliance of Canada (formerly Compute Canada) high-performance computing (HPC) clusters.

---

## 1. Environment Setup

Alliance Canada clusters (e.g., `<CLUSTER_NAME>` such as *Beluga*, *Cedar*, *Graham*, or *Narval*) utilize a hierarchical module system to manage software versions.

### Recommended Modules & Versions
To set up a stable environment for PyTorch and PyTorch Geometric, load the following modules:
```bash
# Clean modules
module purge

# Load basic development tools
module load stdenv/<STDENV_VERSION>       # e.g., stdenv/2023
module load python/<PYTHON_VERSION>       # e.g., python/3.10.13
module load gcc/<GCC_VERSION>             # e.g., gcc/11.3.0
module load cuda/<CUDA_VERSION>           # e.g., cuda/12.2

# Load spatial libraries for GIS feature preparation (Geopandas, Fiona, etc.)
module load gdal/<GDAL_VERSION>           # e.g., gdal/3.7.2
module load proj/<PROJ_VERSION>           # e.g., proj/9.2.1
```

### Virtual Environment Configuration
Do not install packages in your global module space. Create a virtual environment on the cluster's scratch or project storage:
```bash
# Create virtual environment
python3 -m venv ~/envs/hpai_env

# Activate virtual environment
source ~/envs/hpai_env/bin/activate

# Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# Install PyTorch and PyG (wheels optimized for Compute Canada)
pip install torch --index-url https://download.pytorch.org/whl/cu122
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.2.0+cu122.html

# Install package in editable mode
cd /project/<ACCOUNT_NAME>/mkhoshba/hpai-heterogeneous-gnn-canada
pip install -e .
```

---

## 2. Cluster Storage Guidelines

Use the appropriate directory depending on file access frequency and size:
* **`/home/mkhoshba/`**: Personal configurations, small scripts, code files. (Quota: 50GB, backed up).
* **`/project/def-mkhoshba/`**: Long-term research storage, project outputs, final model weights. (Shared allocation, backed up).
* **`/scratch/mkhoshba/`**: Heavy temporary storage, raw spatial datasets (e.g., eBird raster geotiffs, Sentinel/ERA5 environmental parquet files). (Quota: 20TB, **purged regularly**, not backed up).

---

## 3. Slurm Job Resource Specifications

Jobs must be scheduled using the Slurm workload manager. Do not run heavy processing tasks on login nodes.

### Standard Resource Template

Below is a reference for configuring job resources. Use these placeholders when setting up script scripts.

* **Account allocation:** `--account=<ACCOUNT_NAME>` (e.g., `def-mkhoshba` or `rrg-mkhoshba`)
* **Time limit:** `--time=<WALL_TIME>` (e.g., `03:00:00` for 3 hours, or `24:00:00` for 1 day)
* **CPUs:** `--cpus-per-task=<NUM_CPUS>` (e.g., `8` for preprocessing, `4` for GNN training)
* **Memory:** `--mem=<MEMORY>` (e.g., `32G` or `64G` depending on graph scale)
* **GPUs:** `--gres=gpu:<GPU_TYPE>:<GPU_COUNT>` (e.g., `gpu:v100:1` or `gpu:a100:1` or simply `gpu:1` for generic node)

### Example Interactive Dev Session
To request an interactive session for debugging:
```bash
salloc --account=<ACCOUNT_NAME> --time=01:00:00 --cpus-per-task=4 --mem=16G --gres=gpu:1
```
Once resources are allocated, activate your virtual environment:
```bash
source ~/envs/hpai_env/bin/activate
```
