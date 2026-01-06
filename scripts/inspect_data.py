#!/usr/bin/env python3
"""Inspect single-cell data from brieflow output.

This script loads data, validates the metadata/feature split, and prints statistics.
Use it to verify the data loader works correctly before running embeddings.
"""

import argparse
from pathlib import Path

import pandas as pd

from scpepato.data import (
    load_single_cell_data,
    split_metadata_features,
    subsample_cells,
    validate_features,
)


def main():
    parser = argparse.ArgumentParser(description="Inspect brieflow single-cell data")
    parser.add_argument(
        "--parquet",
        type=str,
        default="/lab/ops_analysis/cheeseman/mayon-analysis/analysis/brieflow_output/aggregate/parquets/CeCl-all_ChCo-DAPI_GLYCORNA__aligned.parquet",
        help="Path to aligned parquet file",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=10000,
        help="Number of cells to sample for inspection",
    )
    parser.add_argument(
        "--save-sample",
        type=str,
        default=None,
        help="Path to save sample data (optional)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("INSPECTING BRIEFLOW DATA")
    print("=" * 80)

    # Load data
    print(f"\nLoading: {args.parquet}")
    df = load_single_cell_data(args.parquet)
    print(f"Total cells: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    # Subsample for inspection
    if args.n_cells < len(df):
        print(f"\nSubsampling to {args.n_cells:,} cells...")
        df = subsample_cells(df, args.n_cells, stratify_col="well")
        print(f"Sampled cells: {len(df):,}")

    # Split metadata and features
    print("\n" + "=" * 80)
    print("METADATA / FEATURE SPLIT")
    print("=" * 80)

    metadata, features, feature_names = split_metadata_features(df)

    print(f"\nMetadata columns: {len(metadata.columns)}")
    print(f"Feature columns: {len(feature_names)}")
    print(f"Feature array shape: {features.shape}")
    print(f"Feature array dtype: {features.dtype}")

    # Validate features
    print("\n" + "=" * 80)
    print("FEATURE VALIDATION")
    print("=" * 80)

    stats = validate_features(features)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Check metadata columns
    print("\n" + "=" * 80)
    print("METADATA COLUMNS")
    print("=" * 80)

    for col in metadata.columns:
        dtype = metadata[col].dtype
        n_unique = metadata[col].nunique()
        sample = metadata[col].dropna().iloc[:3].tolist() if len(metadata[col].dropna()) > 0 else []
        print(f"\n{col}:")
        print(f"  dtype: {dtype}, unique: {n_unique}")
        print(f"  sample: {sample}")

    # Perturbation statistics
    print("\n" + "=" * 80)
    print("PERTURBATION STATISTICS")
    print("=" * 80)

    if "gene_symbol_0" in metadata.columns:
        gene_counts = metadata["gene_symbol_0"].value_counts()
        print(f"\nUnique genes: {len(gene_counts)}")
        print(
            f"Cells per gene: min={gene_counts.min()}, max={gene_counts.max()}, median={gene_counts.median():.0f}"
        )
        print("\nTop 10 genes by cell count:")
        for gene, count in gene_counts.head(10).items():
            print(f"  {gene}: {count}")

    # Cell cycle statistics
    print("\n" + "=" * 80)
    print("CELL CYCLE STATISTICS")
    print("=" * 80)

    if "class" in metadata.columns:
        class_counts = metadata["class"].value_counts()
        print("\nCell cycle classes:")
        for cls, count in class_counts.items():
            pct = count / len(metadata) * 100
            print(f"  {cls}: {count} ({pct:.1f}%)")

    # Well statistics
    print("\n" + "=" * 80)
    print("WELL STATISTICS")
    print("=" * 80)

    if "well" in metadata.columns:
        well_counts = metadata["well"].value_counts()
        print("\nCells per well:")
        for well, count in sorted(well_counts.items()):
            print(f"  {well}: {count}")

    # Save sample if requested
    if args.save_sample:
        print("\n" + "=" * 80)
        print("SAVING SAMPLE")
        print("=" * 80)

        sample_path = Path(args.save_sample)
        sample_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as parquet with metadata + features
        sample_df = pd.concat([metadata, pd.DataFrame(features, columns=feature_names)], axis=1)
        sample_df.to_parquet(sample_path)
        print(f"Saved sample to: {sample_path}")

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
