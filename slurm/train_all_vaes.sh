#!/bin/bash
# =============================================================================
# Submit all VAE variants for training comparison
# =============================================================================
# Usage:
#   ./slurm/train_all_vaes.sh           # Default: 50k cells
#   ./slurm/train_all_vaes.sh 100000    # 100k cells
# =============================================================================

N_CELLS=${1:-50000}

echo "Submitting VAE training jobs for $N_CELLS cells..."
echo ""

# Create log directory
mkdir -p slurm/logs

# Submit all three variants
JOB1=$(sbatch --parsable slurm/train_vae.sbatch vanilla $N_CELLS)
echo "Vanilla VAE:     Job $JOB1"

JOB2=$(sbatch --parsable slurm/train_vae.sbatch batch-aware $N_CELLS)
echo "Batch-Aware VAE: Job $JOB2"

JOB3=$(sbatch --parsable slurm/train_vae.sbatch conditional $N_CELLS)
echo "Conditional VAE: Job $JOB3"

echo ""
echo "Monitor with: squeue -u $USER"
echo "View logs:    tail -f slurm/logs/vae_train_<jobid>.out"
