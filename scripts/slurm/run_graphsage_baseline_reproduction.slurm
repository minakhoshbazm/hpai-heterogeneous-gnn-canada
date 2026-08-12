#!/bin/bash
#SBATCH --account=def-drozita_gpu
#SBATCH --job-name=gs-baseline-5seed
#SBATCH --gres=gpu:h100:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/graphsage_baseline_%j.out
#SBATCH --error=logs/graphsage_baseline_%j.err

set -euo pipefail
module load python/3.10.13
source /home/mkhoshba/projects/def-drozita/mkhoshba/venvs/hpai_gnn/bin/activate
cd /home/mkhoshba/projects/def-drozita/mkhoshba/hpai-heterogeneous-gnn-canada
mkdir -p logs results/baselines
python scripts/run_group_a_graphsage_baseline.py
