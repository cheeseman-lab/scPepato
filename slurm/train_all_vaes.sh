#!/bin/bash
# =============================================================================
# Submit all VAE variants for training comparison
# =============================================================================
# Usage:
#   ./slurm/train_all_vaes.sh           # Default: 20k cells, 100 epochs
#   ./slurm/train_all_vaes.sh 50000     # 50k cells
#   ./slurm/train_all_vaes.sh 20000 50  # 20k cells, 50 epochs
#
# Submits 6 jobs total:
#   - 3 model types (vanilla, batch-aware, conditional)
#   - 2 data modes (single-well, multi-well)
# =============================================================================

N_CELLS=${1:-20000}
EPOCHS=${2:-100}

echo "=============================================="
echo "Submitting VAE training jobs"
echo "=============================================="
echo "N cells: $N_CELLS"
echo "Epochs: $EPOCHS"
echo ""

# Create log directory
mkdir -p slurm/logs

# Model types
MODELS=("vanilla" "batch-aware" "conditional")
DATA_MODES=("single" "multi")

echo "Submitting 6 jobs (3 models × 2 data modes)..."
echo ""

for model in "${MODELS[@]}"; do
    for data_mode in "${DATA_MODES[@]}"; do
        JOB_ID=$(sbatch --parsable slurm/train_vae.sbatch "$model" "$data_mode" "$N_CELLS" "$EPOCHS")
        echo "  ${model} (${data_mode}): Job $JOB_ID"
    done
done

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f slurm/logs/vae_train_<jobid>.out"
echo ""
echo "Results will be saved to:"
echo "  outputs/vae/<model>_<data_mode>_${N_CELLS}/"
