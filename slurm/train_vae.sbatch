#!/bin/bash
#SBATCH --job-name=vae_train
#SBATCH --partition=nvidia-A4000-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# =============================================================================
# VAE Training SLURM Script
# =============================================================================
# Usage:
#   sbatch slurm/train_vae.sbatch                    # Default: vanilla VAE, 50k cells
#   sbatch slurm/train_vae.sbatch vanilla 100000     # Vanilla VAE, 100k cells
#   sbatch slurm/train_vae.sbatch batch-aware 50000  # Batch-aware VAE
#   sbatch slurm/train_vae.sbatch conditional 50000  # Conditional VAE
#
# GPU Partitions (change #SBATCH --partition above):
#   nvidia-A4000-20   - A4000 (default)
#   nvidia-t4-20      - T4
#   nvidia-A6000-20   - A6000 (7 day limit)
#   nvidia-L40S-20    - L40S (7 day limit)
#   nvidia-A100-20    - A100 (7 day limit)
#
# Monitor:
#   squeue -u $USER
#   tail -f slurm/logs/vae_train_<jobid>.out
# =============================================================================

# Parse arguments (with defaults)
MODEL_TYPE=${1:-vanilla}
N_CELLS=${2:-50000}
EPOCHS=${3:-100}
BETA=${4:-1.0}

# Use personal wandb account (not team)
export WANDB_ENTITY=md3498

# Create log directory
mkdir -p slurm/logs

# Print job info
echo "=============================================="
echo "VAE Training Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Parameters:"
echo "  Model: $MODEL_TYPE"
echo "  N cells: $N_CELLS"
echo "  Epochs: $EPOCHS"
echo "  Beta: $BETA"
echo "=============================================="
echo ""

# Change to project directory
cd /lab/barcheese01/mdiberna/scPepato

# Activate conda environment
source /lab/barcheese01/mdiberna/miniconda3/etc/profile.d/conda.sh
conda activate scpepato

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run training (unbuffered output for real-time logs)
python -u scripts/train_vae.py \
    --model "$MODEL_TYPE" \
    --n-cells "$N_CELLS" \
    --epochs "$EPOCHS" \
    --beta "$BETA" \
    --wandb-project scpepato-vae \
    --wandb-tags slurm "$MODEL_TYPE" \
    --name "${MODEL_TYPE}_vae_${N_CELLS}"

echo ""
echo "=============================================="
echo "Job finished: $(date)"
echo "=============================================="
