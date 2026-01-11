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
#   sbatch slurm/train_vae.sbatch                                    # Default: vanilla, multi, 20k cells
#   sbatch slurm/train_vae.sbatch vanilla multi 20000                # Vanilla VAE, multi-well, 20k cells
#   sbatch slurm/train_vae.sbatch batch-aware single 20000           # Batch-aware, single-well
#   sbatch slurm/train_vae.sbatch conditional multi 20000 100 1.0 20 # Conditional, 20 epoch warmup
#
# Arguments: MODEL_TYPE DATA_MODE N_CELLS EPOCHS BETA KL_WARMUP
#   MODEL_TYPE: vanilla | batch-aware | conditional (default: vanilla)
#   DATA_MODE:  single | multi (default: multi)
#               single = single well (P-1_W-A1), multi = all 6 wells
#   N_CELLS:    Number of cells to train on (default: 20000)
#   EPOCHS:     Max training epochs (default: 100)
#   BETA:       KL weight (default: 1.0)
#   KL_WARMUP:  Epochs to anneal beta from 0 to target (default: 10)
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
DATA_MODE=${2:-multi}
N_CELLS=${3:-20000}
EPOCHS=${4:-100}
BETA=${5:-0.1}
KL_WARMUP=${6:-10}

# Set pattern based on data mode
if [ "$DATA_MODE" == "single" ]; then
    PATTERN="*P-1_W-A1*filtered.parquet"
else
    PATTERN="*CeCl-all*GLYCORNA__filtered.parquet"
fi

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
echo "  Data mode: $DATA_MODE"
echo "  Pattern: $PATTERN"
echo "  N cells: $N_CELLS"
echo "  Epochs: $EPOCHS"
echo "  Beta: $BETA"
echo "  KL warmup: $KL_WARMUP epochs"
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
    --pattern "$PATTERN" \
    --n-cells "$N_CELLS" \
    --epochs "$EPOCHS" \
    --beta "$BETA" \
    --kl-warmup-epochs "$KL_WARMUP" \
    --wandb-project scpepato-vae \
    --wandb-tags slurm "$MODEL_TYPE" "$DATA_MODE" \
    --name "${MODEL_TYPE}_${DATA_MODE}_${N_CELLS}"

echo ""
echo "=============================================="
echo "Job finished: $(date)"
echo "=============================================="
