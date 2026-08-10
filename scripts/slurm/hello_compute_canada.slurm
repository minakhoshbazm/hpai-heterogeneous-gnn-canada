#!/bin/bash
#SBATCH --account=def-drozita_gpu
#SBATCH --job-name=hello_hpai_gnn
#SBATCH --output=logs/hello_%j.out
#SBATCH --error=logs/hello_%j.err
#SBATCH --time=00:10:00                # Limit job execution to 10 minutes (placeholder)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1              # Request 1 CPU (placeholder)
#SBATCH --mem=4G                       # Request 4GB memory (placeholder)
#SBATCH --gres=gpu:1                   # Request 1 GPU (placeholder)

# Exit immediately if a command exits with a non-zero status
set -e

module load python/3.10.13
source /home/mkhoshba/projects/def-drozita/mkhoshba/venvs/hpai_gnn/bin/activate

echo "=========================================================="
echo "      COMPUTE CANADA / ALLIANCE CANADA DIAGNOSTIC REPORT"
echo "=========================================================="
echo "Date/Time:        $(date)"
echo "Running Hostname: $(hostname)"
echo "Working Dir:      $(pwd)"
echo "Submit Dir:       $SLURM_SUBMIT_DIR"
echo "=========================================================="

echo "1. Checking Python environment..."
if command -v python3 &>/dev/null; then
    echo "Python path: $(which python3)"
    python3 --version
else
    echo "python3 could not be found. Loading default python module..."
    module load python/3.10.13
    python3 --version
fi

echo "=========================================================="
echo "2. Checking CUDA & GPU availability..."
if command -v nvidia-smi &>/dev/null; then
    echo "nvidia-smi is available. GPU Status:"
    nvidia-smi
    
    echo "Checking PyTorch CUDA recognition..."
    python3 -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
else
    echo "nvidia-smi is NOT available (running on CPU-only node or drivers missing)."
    python3 -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())"
fi

echo "=========================================================="
echo "3. Importing hpai_gnn package test..."
python3 -c "import hpai_gnn; print('Successfully imported hpai_gnn version:', hpai_gnn.__version__)"
echo "=========================================================="
echo "Diagnostic script completed."
