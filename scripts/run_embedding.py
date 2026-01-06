#!/usr/bin/env python3
"""Run single-cell embedding on GlycoRNA screen data.

This script loads brieflow single-cell data, computes embeddings (PCA→UMAP, PCA→PHATE),
and generates visualizations to validate whether known regulators separate at single-cell level.

Usage:
    python scripts/run_embedding.py
    python scripts/run_embedding.py --n-cells 50000 --output-dir outputs/test
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scpepato.data import (
    load_aggregated_data,
    load_single_cell_data,
    split_metadata_features,
    subsample_cells,
)
from scpepato.embedding import run_phate, run_umap
from scpepato.viz import compare_embeddings, highlight_perturbations, plot_embedding

# === Configuration ===
BRIEFLOW_ROOT = Path("/lab/ops_analysis/cheeseman/mayon-analysis/analysis/brieflow_output")
SINGLE_CELL_PATH = BRIEFLOW_ROOT / "aggregate/parquets/CeCl-all_ChCo-DAPI_GLYCORNA__aligned.parquet"
AGGREGATED_PATH = BRIEFLOW_ROOT / "aggregate/tsvs/CeCl-all_ChCo-DAPI_GLYCORNA__aggregated.tsv"


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-cell embedding analysis")
    parser.add_argument("--n-cells", type=int, default=100_000, help="Number of cells to subsample")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--pca-components", type=int, default=50, help="Number of PCA components")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def extract_genes_of_interest(agg_df: pd.DataFrame) -> list[str]:
    """Extract genes from non-nontargeting-dominated clusters."""
    cluster_cols = [c for c in agg_df.columns if "cluster" in c.lower() or "leiden" in c.lower()]

    if not cluster_cols:
        print("  No cluster columns found in aggregated data")
        return []

    cluster_col = cluster_cols[0]
    gene_col = "gene_symbol_0"

    if gene_col not in agg_df.columns:
        print(f"  Gene column '{gene_col}' not found")
        return []

    genes_of_interest = []

    for cluster_id in agg_df[cluster_col].unique():
        cluster_mask = agg_df[cluster_col] == cluster_id
        cluster_genes = agg_df.loc[cluster_mask, gene_col].tolist()

        # Skip clusters dominated by nontargeting
        nt_fraction = sum(1 for g in cluster_genes if g == "nontargeting") / len(cluster_genes)

        if nt_fraction < 0.5:
            non_nt_genes = [g for g in cluster_genes if g != "nontargeting"]
            genes_of_interest.extend(non_nt_genes)
            print(f"  Cluster {cluster_id}: {len(non_nt_genes)} genes (nt_frac={nt_fraction:.1%})")

    return list(set(genes_of_interest))


def main():
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # === 1. Load aggregated data to identify genes of interest ===
    print("\n=== Loading aggregated data ===")
    t0 = time.time()
    agg_df = load_aggregated_data(AGGREGATED_PATH)
    print(f"  Shape: {agg_df.shape} ({time.time() - t0:.1f}s)")

    print("\n=== Extracting genes of interest ===")
    genes_of_interest = extract_genes_of_interest(agg_df)
    print(f"  Total genes of interest: {len(genes_of_interest)}")

    # === 2. Load single-cell data ===
    print("\n=== Loading single-cell data ===")
    t0 = time.time()
    sc_df = load_single_cell_data(SINGLE_CELL_PATH)
    print(f"  Full dataset: {sc_df.shape} ({time.time() - t0:.1f}s)")
    print(f"  Memory: {sc_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    # === 3. Subsample ===
    print(f"\n=== Subsampling to {args.n_cells:,} cells ===")
    t0 = time.time()
    stratify_col = "gene_symbol_0" if "gene_symbol_0" in sc_df.columns else None
    sc_sub = subsample_cells(
        sc_df, n=args.n_cells, stratify_col=stratify_col, random_state=args.seed
    )
    print(f"  Subsampled: {sc_sub.shape} ({time.time() - t0:.1f}s)")

    # Free memory
    del sc_df

    # === 4. Split metadata and features ===
    print("\n=== Splitting metadata/features ===")
    metadata, features, feature_names = split_metadata_features(sc_sub)
    print(f"  Metadata: {metadata.shape}")
    print(f"  Features: {features.shape}")
    print(f"  Feature columns: {len(feature_names)} (PC_0 - PC_{len(feature_names) - 1})")

    perturbations = metadata["gene_symbol_0"] if "gene_symbol_0" in metadata.columns else None
    if perturbations is not None:
        print(f"  Unique perturbations: {perturbations.nunique()}")
        print(f"  Top 5: {perturbations.value_counts().head().to_dict()}")

    # === 5. Use existing PCs (data is already PCA-transformed) ===
    print(f"\n=== Using existing PCs (first {args.pca_components}) ===")
    # Data already contains PC_0 to PC_430, just use the first N
    X_pca = features[:, : args.pca_components]
    print(f"  Shape: {X_pca.shape}")

    # === 6. Run UMAP ===
    print("\n=== Running UMAP ===")
    t0 = time.time()
    X_umap = run_umap(X_pca, n_neighbors=15, min_dist=0.1, random_state=args.seed)
    print(f"  Shape: {X_umap.shape} ({time.time() - t0:.1f}s)")

    # === 7. Run PHATE ===
    print("\n=== Running PHATE ===")
    t0 = time.time()
    X_phate = run_phate(X_pca, knn=5, decay=40, random_state=args.seed)
    print(f"  Shape: {X_phate.shape} ({time.time() - t0:.1f}s)")

    # === 8. Visualizations ===
    print("\n=== Generating visualizations ===")

    # Basic embedding comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_embedding(X_umap, ax=axes[0], title="PCA → UMAP", alpha=0.3, s=1)
    plot_embedding(X_phate, ax=axes[1], title="PCA → PHATE", alpha=0.3, s=1)
    plt.tight_layout()
    fig.savefig(args.output_dir / "embeddings_basic.png", dpi=150, bbox_inches="tight")
    print("  Saved: embeddings_basic.png")
    plt.close(fig)

    # Highlight genes of interest
    if perturbations is not None and genes_of_interest:
        present_goi = [g for g in genes_of_interest if g in perturbations.values]
        print(f"  Genes of interest in data: {len(present_goi)}")

        if present_goi:
            # UMAP with highlights
            fig = highlight_perturbations(X_umap, perturbations, present_goi[:10], figsize=(18, 5))
            plt.suptitle("UMAP: Genes of Interest vs Nontargeting", y=1.02)
            fig.savefig(args.output_dir / "umap_highlights.png", dpi=150, bbox_inches="tight")
            print("  Saved: umap_highlights.png")
            plt.close(fig)

            # PHATE with highlights
            fig = highlight_perturbations(X_phate, perturbations, present_goi[:10], figsize=(18, 5))
            plt.suptitle("PHATE: Genes of Interest vs Nontargeting", y=1.02)
            fig.savefig(args.output_dir / "phate_highlights.png", dpi=150, bbox_inches="tight")
            print("  Saved: phate_highlights.png")
            plt.close(fig)

            # Side-by-side comparison
            fig = compare_embeddings(
                {"UMAP": X_umap, "PHATE": X_phate},
                perturbations=perturbations,
                highlight_genes=present_goi[:5],
                figsize_per_plot=(8, 8),
                background_alpha=0.1,
                highlight_alpha=0.5,
            )
            plt.suptitle("Embedding Comparison", y=1.02)
            fig.savefig(args.output_dir / "embedding_comparison.png", dpi=150, bbox_inches="tight")
            print("  Saved: embedding_comparison.png")
            plt.close(fig)

    # === 9. Save results ===
    print("\n=== Saving results ===")
    results = pd.DataFrame(
        {
            "umap_1": X_umap[:, 0],
            "umap_2": X_umap[:, 1],
            "phate_1": X_phate[:, 0],
            "phate_2": X_phate[:, 1],
        }
    )
    results = pd.concat([metadata.reset_index(drop=True), results], axis=1)

    results_path = args.output_dir / "embeddings.parquet"
    results.to_parquet(results_path)
    print(f"  Saved: {results_path}")
    print(f"  Shape: {results.shape}")

    # Save genes of interest
    if genes_of_interest:
        goi_path = args.output_dir / "genes_of_interest.txt"
        with open(goi_path, "w") as f:
            f.write("\n".join(sorted(genes_of_interest)))
        print(f"  Saved: {goi_path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
