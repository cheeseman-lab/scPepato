# scPepato

Single-Cell Perturbation Analysis Toolkit for Optical Pooled Screening.

A Python toolkit for computing and comparing embeddings from single-cell morphological data, with support for batch correction and perturbation analysis.

## Installation

```bash
conda create -n scpepato -c conda-forge python=3.11 uv pip -y
conda activate scpepato
uv pip install -e .
```

## Quick Start

The pipeline has three main steps:

```
1. Data (filtered parquets)  →  2. Embeddings (PCA/VAE)  →  3. Analysis
```

### Step 1: Data

Input data are filtered parquet files from Brieflow containing raw CellProfiler features (~865 dimensions per cell).

Default data location:
```
/lab/ops_analysis/cheeseman/{screen}-analysis/analysis/brieflow_output/aggregate/parquets/
```

### Step 2: Compute Embeddings

Choose from PCA or VAE methods, with or without batch correction:

| | No batch correction | With batch correction |
|---|---|---|
| **PCA** | `--method vanilla` | `--method batch_corrected` |
| **VAE** | `--model vanilla` | `--model batch-aware` |

**PCA embeddings:**
```bash
# Vanilla PCA (global normalization)
python scripts/compute_pca.py --method vanilla --n-cells 20000

# Batch-corrected PCA (per-batch scaling + TVN)
python scripts/compute_pca.py --method batch_corrected --n-cells 20000
```

**VAE embeddings (GPU recommended):**
```bash
# Vanilla VAE
python scripts/train_vae.py --model vanilla --n-cells 50000

# Batch-aware VAE
python scripts/train_vae.py --model batch-aware --n-cells 50000

# Conditional VAE (perturbation-aware)
python scripts/train_vae.py --model conditional --n-cells 50000
```

For large runs, use SLURM:
```bash
srun -p gpu --gpus=1 --mem=64gb --time=04:00:00 \
    conda run -n scpepato python scripts/train_vae.py --n-cells 100000
```

### Step 3: Compare Embedding Spaces

Analyze and compare embeddings with distance metrics, batch effects, and response heterogeneity:

```bash
python scripts/compare_spaces.py \
    outputs/embeddings/pca/mayon_20000 \
    outputs/embeddings/pca_batch_corrected/mayon_20000 \
    outputs/embeddings/vanilla_vae/mayon_50000 \
    --output results/comparison
```

This produces:
- `distance_results.csv` - Per-perturbation distance metrics
- `batch_metrics.csv` - Batch mixing scores and variance ratios
- `heterogeneity_results.csv` - Response heterogeneity analysis
- Visualization plots (PNG)

## Output Format

All embedding methods produce standardized outputs:

```
outputs/embeddings/{method}/{screen}_{n_cells}/
├── embeddings.npy        # (n_cells, n_dims) float32
├── metadata.parquet      # Cell metadata (plate, well, gene, etc.)
├── manifest.json         # Configuration and file paths
├── model.pkl or .pt      # Fitted model (PCA or VAE)
└── normalizer.npz        # Feature normalization stats
```

### Loading Embeddings

```python
from scpepato.embeddings import load_embedding

# Load any embedding type
emb = load_embedding("outputs/embeddings/pca/mayon_20000")
print(emb.embeddings.shape)  # (20000, 50)
print(emb.method)            # "pca"
print(emb.metadata.head())   # Cell metadata
```

## Project Structure

```
src/scpepato/
├── embeddings/     # Embedding computation and loading
│   ├── base.py     # EmbeddingOutput, load_embedding()
│   ├── pca.py      # PCA with batch correction (TVN)
│   └── vae.py      # VAE loading utilities
├── models/         # VAE model architectures
│   ├── vae.py      # VanillaVAE
│   ├── batch_vae.py      # BatchAwareVAE
│   ├── conditional_vae.py # ConditionalVAE
│   └── training.py # Training loop with W&B
├── data/           # Data loading utilities
├── analysis/       # Downstream analysis
│   ├── batch_effects.py  # Batch mixing metrics
│   └── responders.py     # Heterogeneity detection
└── distances/      # Distance metrics (E-distance, MMD)

scripts/
├── compute_pca.py      # PCA embedding script
├── train_vae.py        # VAE training script
├── compare_spaces.py   # Embedding comparison
└── run_visualizer.py   # Interactive visualization
```

## Embedding Methods

### PCA (vanilla)
Standard PCA with global feature normalization. Fast baseline.

### PCA (batch_corrected)
Three-step batch correction:
1. **Batch center-scaling**: StandardScaler per batch
2. **PCA**: Dimensionality reduction
3. **TVN**: Typical Variation Normalization using control samples with CORAL-style covariance alignment

### VAE (vanilla)
Standard variational autoencoder. Learns nonlinear embeddings.

### VAE (batch-aware)
VAE with batch embeddings in the decoder. Encourages batch-invariant latent space.

### VAE (conditional)
VAE conditioned on perturbation labels. Useful for counterfactual analysis.

## Documentation

- `data/screens.csv` - Available screens with paths
