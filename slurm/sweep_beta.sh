#!/bin/bash
# =============================================================================
# Beta hyperparameter sweep for VAE
# =============================================================================
# Usage:
#   ./slurm/sweep_beta.sh              # Default: vanilla VAE
#   ./slurm/sweep_beta.sh batch-aware  # Batch-aware VAE
# =============================================================================

MODEL_TYPE=${1:-vanilla}
N_CELLS=${2:-50000}

echo "Submitting beta sweep for $MODEL_TYPE VAE ($N_CELLS cells)..."
echo ""

# Create log directory
mkdir -p slurm/logs

# Beta values to test
BETAS=(0.1 0.5 1.0 2.0 5.0)

for BETA in "${BETAS[@]}"; do
    JOB=$(sbatch --parsable --job-name="vae_b${BETA}" slurm/train_vae.sbatch "$MODEL_TYPE" "$N_CELLS" 100 "$BETA")
    echo "Beta=$BETA: Job $JOB"
done

echo ""
echo "Monitor with: squeue -u $USER"
