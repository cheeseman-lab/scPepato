#!/usr/bin/env python3
"""Compute and save PCA embeddings in standardized format.

Two methods available:
- vanilla: Standard PCA with global normalization
- batch_corrected: Batch center-scaling + PCA + TVN (Typical Variation Normalization)

Creates standardized outputs:
    outputs/embeddings/pca/mayon_20000/           # vanilla
    outputs/embeddings/pca_batch_corrected/mayon_20000/  # batch_corrected
    ├── embeddings.npy        # (n_cells, n_dims) PCA embeddings
    ├── metadata.parquet      # Cell metadata
    ├── manifest.json         # Configuration
    ├── model.pkl             # Fitted PCA model
    └── normalizer.npz        # Feature normalization stats

Usage:
    # Vanilla PCA
    python scripts/compute_pca.py --n-cells 20000

    # Batch-corrected PCA (requires batch and perturbation columns)
    python scripts/compute_pca.py --method batch_corrected --n-cells 20000

    # Quick test
    python scripts/compute_pca.py --n-cells 5000 --method vanilla
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from scpepato.data.features import FeatureNormalizer, load_multiple_filtered_parquets
from scpepato.embeddings.pca import centerscale_by_batch, tvn_on_controls

# Default paths
DEFAULT_PARQUET_DIR = Path(
    "/lab/ops_analysis/cheeseman/mayon-analysis/analysis/brieflow_output/aggregate/parquets"
)
DEFAULT_PATTERN = "P-*_W-*_CeCl-all_ChCo-DAPI_GLYCORNA__filtered.parquet"
OUTPUT_DIR = Path("outputs/embeddings")


def main():
    parser = argparse.ArgumentParser(
        description="Compute PCA embeddings with optional batch correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["vanilla", "batch_corrected"],
        default="vanilla",
        help="PCA method: vanilla (global scaling) or batch_corrected (per-batch + TVN)",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help="Directory containing filtered parquet files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help="Glob pattern for parquet files",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=20000,
        help="Number of cells to sample",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=50,
        help="Number of PCA components",
    )
    parser.add_argument(
        "--batch-col",
        type=str,
        default="plate_well",
        help="Column for batch labels (use 'plate_well' to combine plate+well)",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default=None,
        help="Column for perturbation labels (auto-detected if not specified)",
    )
    parser.add_argument(
        "--control-label",
        type=str,
        default="nontargeting",
        help="Label prefix for control samples",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Base output directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: mayon_{n_cells})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Determine method-specific output directory
    if args.method == "vanilla":
        method_dir = "pca"
    else:
        method_dir = "pca_batch_corrected"

    exp_name = args.name or f"mayon_{args.n_cells}"
    output_dir = args.output_dir / method_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"PCA EMBEDDING: {args.method.upper()}")
    print("=" * 70)
    print(f"Method: {args.method}")
    print(f"Data: {args.parquet_dir.name}/{args.pattern}")
    print(f"Cells: {args.n_cells:,}")
    print(f"Components: {args.n_components}")
    if args.method == "batch_corrected":
        print(f"Batch col: {args.batch_col}")
        print(f"Control label: {args.control_label}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    print("Loading data...")
    features_df, metadata_df = load_multiple_filtered_parquets(
        args.parquet_dir,
        pattern=args.pattern,
        n_total_rows=args.n_cells,
        random_state=args.seed,
    )

    print(f"  Loaded {len(features_df):,} cells")
    print(f"  Features: {features_df.shape[1]}")

    # Drop rows with NA values
    na_mask = features_df.isna().any(axis=1)
    if na_mask.sum() > 0:
        print(f"  Dropping {na_mask.sum()} rows with NA values")
        features_df = features_df[~na_mask].reset_index(drop=True)
        metadata_df = metadata_df[~na_mask].reset_index(drop=True)

    # Create batch column if needed
    if args.method == "batch_corrected" and args.batch_col == "plate_well":
        if "plate" in metadata_df.columns and "well" in metadata_df.columns:
            metadata_df["plate_well"] = (
                metadata_df["plate"].astype(str) + "_" + metadata_df["well"].astype(str)
            )
        elif "well" in metadata_df.columns:
            metadata_df["plate_well"] = metadata_df["well"].astype(str)

    # Auto-detect perturbation column
    pert_col = args.pert_col
    if pert_col is None:
        if "gene_symbol_0" in metadata_df.columns:
            pert_col = "gene_symbol_0"
        elif "gene" in metadata_df.columns:
            pert_col = "gene"

    # Aggregate nontargeting labels
    if pert_col and pert_col in metadata_df.columns:
        pert_labels = metadata_df[pert_col].values.astype(str)
        nontargeting_mask = np.array([str(p).startswith("nontargeting") for p in pert_labels])
        metadata_df = metadata_df.copy()
        metadata_df.loc[nontargeting_mask, pert_col] = "nontargeting"
        n_controls = nontargeting_mask.sum()
        n_perturbations = len(metadata_df[pert_col].unique())
        print(f"  Control cells (nontargeting): {n_controls}")
        print(f"  Unique perturbations: {n_perturbations}")

    # Get raw features
    features = features_df.values.astype(np.float32)

    # Compute PCA
    print(f"\nComputing PCA ({args.method})...")

    if args.method == "vanilla":
        # Standard approach: global normalization + PCA
        normalizer = FeatureNormalizer()
        normalizer.fit(features_df)
        features_normalized = normalizer.transform(features_df)

        pca = PCA(n_components=args.n_components, random_state=args.seed)
        embeddings = pca.fit_transform(features_normalized)

        config = {
            "method": "pca",
            "variant": "vanilla",
            "n_components": args.n_components,
            "n_features": features.shape[1],
            "n_cells": len(embeddings),
            "variance_explained": float(pca.explained_variance_ratio_.sum()),
            "parquet_dir": str(args.parquet_dir),
            "pattern": args.pattern,
            "seed": args.seed,
        }

    else:  # batch_corrected
        if pert_col is None:
            raise ValueError("Perturbation column required for batch_corrected method")

        # Step 1: Batch center-scaling
        print("  Step 1: Batch center-scaling...")
        features_scaled = centerscale_by_batch(features, metadata_df, args.batch_col)

        # Step 2: PCA
        print("  Step 2: PCA transformation...")
        pca = PCA(n_components=args.n_components, random_state=args.seed)
        embeddings = pca.fit_transform(features_scaled)

        # Step 3: TVN normalization
        print("  Step 3: TVN normalization on controls...")
        embeddings = tvn_on_controls(
            embeddings, metadata_df, pert_col, args.control_label, args.batch_col
        )

        # No normalizer saved for batch_corrected (batch-specific scaling)
        normalizer = None

        config = {
            "method": "pca_batch_corrected",
            "variant": "batch_corrected",
            "n_components": args.n_components,
            "n_features": features.shape[1],
            "n_cells": len(embeddings),
            "variance_explained": float(pca.explained_variance_ratio_.sum()),
            "batch_col": args.batch_col,
            "pert_col": pert_col,
            "control_label": args.control_label,
            "parquet_dir": str(args.parquet_dir),
            "pattern": args.pattern,
            "seed": args.seed,
        }

    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.1%}")

    # Save outputs
    print("\nSaving outputs...")

    # 1. Embeddings
    np.save(output_dir / "embeddings.npy", embeddings)
    print(f"  embeddings.npy: {embeddings.shape}")

    # 2. Metadata
    metadata_df.to_parquet(output_dir / "metadata.parquet")
    print(f"  metadata.parquet: {len(metadata_df)} cells")

    # 3. PCA model
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(pca, f)
    print(f"  model.pkl: PCA({args.n_components})")

    # 4. Normalizer (only for vanilla)
    if normalizer is not None:
        normalizer.save(output_dir / "normalizer.npz")
        print("  normalizer.npz: feature stats")

    # 5. Manifest
    config["files"] = {
        "embeddings": "embeddings.npy",
        "metadata": "metadata.parquet",
        "model": "model.pkl",
    }
    if normalizer is not None:
        config["files"]["normalizer"] = "normalizer.npz"

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(config, f, indent=2)
    print("  manifest.json")

    print("\nDone!")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
