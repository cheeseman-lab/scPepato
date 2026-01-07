#!/usr/bin/env python3
"""Prepare embedding data for the interactive visualizer.

Loads aligned parquet data, samples cells, computes embeddings (UMAP, PHATE, t-SNE),
and saves the result for use with the Dash visualizer.

Usage:
    python scripts/prepare_visualizer_data.py

Output:
    outputs/visualizer/mayon_DAPI_GLYCORNA.parquet
"""

import gc
import time
from pathlib import Path

import numpy as np

from scpepato.data import load_single_cell_data
from scpepato.embedding import run_phate, run_tsne, run_umap


def log(msg: str):
    """Print with immediate flush for real-time output."""
    print(msg, flush=True)


# Configuration
SCREEN = "mayon"
CHANNELS = "DAPI_GLYCORNA"
N_CELLS = 20000  # Sample size for visualization
N_PCS = 50  # Number of PCs to use for embedding
RANDOM_STATE = 42

# Paths
ALIGNED_PARQUET = Path(
    "/lab/ops_analysis/cheeseman/mayon-analysis/analysis/brieflow_output/"
    "aggregate/parquets/CeCl-all_ChCo-DAPI_GLYCORNA__aligned.parquet"
)
OUTPUT_DIR = Path("outputs/visualizer")


def main():
    log("=" * 70)
    log("PREPARE VISUALIZER DATA")
    log("=" * 70)
    log(f"Screen: {SCREEN}")
    log(f"Channels: {CHANNELS}")
    log(f"Sample size: {N_CELLS:,} cells")
    log(f"PCs for embedding: {N_PCS}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define columns to load
    # Metadata for identification
    metadata_cols = [
        "plate",
        "well",
        "tile",
        "gene_symbol_0",
        "sgRNA_0",  # Barcode/sgRNA identifier
    ]

    # Bounds for image extraction
    bounds_cols = [
        "cell_bounds_0",
        "cell_bounds_1",
        "cell_bounds_2",
        "cell_bounds_3",
    ]

    # PC features for embedding
    pc_cols = [f"PC_{i}" for i in range(N_PCS)]

    columns = metadata_cols + bounds_cols + pc_cols

    # Load data with sampling
    log(f"\nLoading data from: {ALIGNED_PARQUET.name}")
    t0 = time.time()

    df = load_single_cell_data(
        ALIGNED_PARQUET,
        columns=columns,
        n_rows=N_CELLS,
        random_state=RANDOM_STATE,
    )

    log(f"  Loaded {len(df):,} cells in {time.time() - t0:.1f}s")

    # Extract PC features for embedding
    X = df[pc_cols].values.astype(np.float32)
    log(f"  Feature matrix: {X.shape}")

    # Check for NaN/Inf
    if np.any(np.isnan(X)):
        log("  WARNING: NaN values found, filling with 0")
        X = np.nan_to_num(X, nan=0.0)
    if np.any(np.isinf(X)):
        log("  WARNING: Inf values found, clipping")
        X = np.clip(X, -1e6, 1e6)

    # Compute embeddings
    log("\nComputing embeddings...")

    # UMAP
    log("\n  Running UMAP...")
    t0 = time.time()
    umap_emb = run_umap(X, n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
    log(f"    Done in {time.time() - t0:.1f}s")

    # PHATE
    log("\n  Running PHATE...")
    t0 = time.time()
    phate_emb = run_phate(X, knn=5, decay=40, random_state=RANDOM_STATE)
    log(f"    Done in {time.time() - t0:.1f}s")

    # t-SNE
    log("\n  Running t-SNE...")
    t0 = time.time()
    tsne_emb = run_tsne(X, perplexity=30, random_state=RANDOM_STATE)
    log(f"    Done in {time.time() - t0:.1f}s")

    # Free memory
    del X
    gc.collect()

    # Add embeddings to dataframe
    log("\nCreating output dataframe...")
    output_df = df[metadata_cols + bounds_cols].copy()
    output_df["umap_1"] = umap_emb[:, 0]
    output_df["umap_2"] = umap_emb[:, 1]
    output_df["phate_1"] = phate_emb[:, 0]
    output_df["phate_2"] = phate_emb[:, 1]
    output_df["tsne_1"] = tsne_emb[:, 0]
    output_df["tsne_2"] = tsne_emb[:, 1]

    log(f"  Output shape: {output_df.shape}")
    log(f"  Columns: {list(output_df.columns)}")

    # Save
    output_path = OUTPUT_DIR / f"{SCREEN}_{CHANNELS}.parquet"
    output_df.to_parquet(output_path, index=False)
    log(f"\nSaved to: {output_path}")
    log(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")

    # Summary stats
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"Total cells: {len(output_df):,}")
    log(f"Unique genes: {output_df['gene_symbol_0'].nunique()}")
    log(f"Unique wells: {output_df['well'].nunique()}")

    # Top genes by count
    log("\nTop 10 genes by cell count:")
    gene_counts = output_df["gene_symbol_0"].value_counts().head(10)
    for gene, count in gene_counts.items():
        log(f"  {gene}: {count:,}")

    log("\nDone! Run the visualizer with:")
    log("  python scripts/run_visualizer.py")


if __name__ == "__main__":
    main()
