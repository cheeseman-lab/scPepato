#!/usr/bin/env python3
"""Compare embedding spaces with distance metrics, batch effects, and heterogeneity analysis.

This script merges the functionality from day6_distances.py and day7_batch_heterogeneity.py
into a single unified comparison tool that works with standardized embedding outputs.

Usage:
    # Compare all embedding spaces in a directory
    python scripts/compare_spaces.py outputs/embeddings/*/mayon_20000

    # Compare specific embedding spaces
    python scripts/compare_spaces.py \
        outputs/embeddings/pca/mayon_20000 \
        outputs/embeddings/pca_batch_corrected/mayon_20000 \
        outputs/embeddings/vanilla_vae/mayon_20000

    # Run specific analyses
    python scripts/compare_spaces.py outputs/embeddings/*/mayon_20000 \
        --distances --batch-effects  # skip heterogeneity

    # Quick test with fewer permutations
    python scripts/compare_spaces.py outputs/embeddings/*/mayon_20000 \
        --n-permutations 100
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scpepato.embeddings import load_embedding


def load_spaces(embedding_dirs: list[Path]) -> dict:
    """Load multiple embedding spaces from standardized directories.

    Parameters
    ----------
    embedding_dirs : list[Path]
        List of directories containing embeddings

    Returns
    -------
    dict
        Mapping from method name to dict with 'embeddings' and 'metadata'
    """
    spaces = {}

    for path in embedding_dirs:
        path = Path(path)
        if not path.exists():
            print(f"  Skipping {path} (not found)")
            continue

        try:
            emb_output = load_embedding(path)
            method = emb_output.method
            spaces[method] = {
                "embeddings": emb_output.embeddings,
                "metadata": emb_output.metadata,
                "config": emb_output.config,
                "path": path,
            }
            print(f"  Loaded {method}: {emb_output.embeddings.shape}")
        except Exception as e:
            print(f"  Failed to load {path}: {e}")

    return spaces


def run_distance_analysis(
    spaces: dict,
    perturbation_labels: np.ndarray,
    control_label: str,
    n_permutations: int,
    min_cells: int,
    output_dir: Path,
):
    """Run distance metrics comparison across embedding spaces."""
    from scpepato.distances import compare_spaces

    print("\n" + "=" * 60)
    print("DISTANCE METRICS ANALYSIS")
    print("=" * 60)

    # Extract embeddings dict
    embeddings_dict = {name: data["embeddings"] for name, data in spaces.items()}

    # Compare spaces
    comparison = compare_spaces(
        embeddings_dict=embeddings_dict,
        perturbation_labels=perturbation_labels,
        control_label=control_label,
        metrics=["e_distance", "mse", "mmd_linear"],
        n_permutations=n_permutations,
        min_cells=min_cells,
    )

    # Check for results
    if len(comparison.results_df) == 0:
        print("\nWarning: No perturbations passed the min_cells filter!")
        print(f"  Try reducing --min-cells (currently {min_cells})")
        return

    # Save results
    comparison.results_df.to_csv(output_dir / "distance_results.csv", index=False)

    # Generate summary
    summary = comparison.summary_by_space()
    summary.to_csv(output_dir / "distance_summary.csv")
    print("\nSummary by space:")
    print(summary)

    # Best space per metric
    best = comparison.best_space_per_metric()
    best.to_csv(output_dir / "best_space_per_metric.csv", index=False)
    print("\nBest space per metric:")
    print(best)

    # Generate figures
    try:
        fig1 = comparison.plot_distance_distributions(metric="e_distance")
        fig1.savefig(output_dir / "distance_distributions.png", dpi=150, bbox_inches="tight")
        plt.close(fig1)

        fig2 = comparison.plot_significance_comparison()
        fig2.savefig(output_dir / "significance_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
    except Exception as e:
        print(f"  Warning: Could not generate distance plots: {e}")


def run_batch_analysis(
    spaces: dict,
    batch_labels: np.ndarray,
    perturbation_labels: np.ndarray,
    output_dir: Path,
):
    """Run batch effects analysis across embedding spaces."""
    from scpepato.analysis.batch_effects import compute_batch_metrics

    print("\n" + "=" * 60)
    print("BATCH EFFECTS ANALYSIS")
    print("=" * 60)

    # Extract embeddings dict
    embeddings_dict = {name: data["embeddings"] for name, data in spaces.items()}

    batch_metrics = compute_batch_metrics(
        embeddings_dict=embeddings_dict,
        batch_labels=batch_labels,
        perturbation_labels=perturbation_labels,
    )

    batch_metrics.to_csv(output_dir / "batch_metrics.csv", index=False)
    print("\nBatch metrics:")
    print(batch_metrics.to_string(index=False))

    # Plot batch metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Batch mixing score
    axes[0].bar(batch_metrics["space"], batch_metrics["batch_mixing_score"])
    axes[0].set_ylabel("Batch Mixing Score\n(higher = better)")
    axes[0].set_title("Batch Mixing")
    axes[0].tick_params(axis="x", rotation=45)

    # Variance ratio
    axes[1].bar(batch_metrics["space"], batch_metrics["variance_ratio"])
    axes[1].set_ylabel("Within/Between Variance Ratio\n(lower = less batch effect)")
    axes[1].set_title("Variance Ratio")
    axes[1].tick_params(axis="x", rotation=45)

    # Cross-batch retrieval
    axes[2].bar(batch_metrics["space"], batch_metrics["cross_batch_retrieval"])
    axes[2].set_ylabel("Cross-Batch Retrieval Score\n(higher = better)")
    axes[2].set_title("Perturbation Retrieval Across Batches")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(output_dir / "batch_metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_heterogeneity_analysis(
    spaces: dict,
    perturbation_labels: np.ndarray,
    control_label: str,
    min_cells: int,
    output_dir: Path,
):
    """Run response heterogeneity analysis across embedding spaces."""
    from scpepato.analysis.responders import compute_responder_fractions

    print("\n" + "=" * 60)
    print("RESPONSE HETEROGENEITY ANALYSIS")
    print("=" * 60)

    # Extract embeddings dict
    embeddings_dict = {name: data["embeddings"] for name, data in spaces.items()}

    heterogeneity_results = compute_responder_fractions(
        embeddings_dict=embeddings_dict,
        perturbation_labels=perturbation_labels,
        control_label=control_label,
        min_cells=min_cells,
    )

    heterogeneity_results.to_csv(output_dir / "heterogeneity_results.csv", index=False)

    # Check if we have results
    if len(heterogeneity_results) == 0:
        print("\nWarning: No perturbations passed the min_cells filter for heterogeneity!")
        print(f"  Try reducing --min-cells (currently {min_cells})")
        return

    # Summary statistics
    het_summary = (
        heterogeneity_results.groupby("space")
        .agg(
            {
                "is_bimodal": ["sum", "mean"],
                "responder_fraction": ["mean", "std"],
                "perturbation": "count",
            }
        )
        .round(4)
    )

    print("\nHeterogeneity summary:")
    print(het_summary)
    het_summary.to_csv(output_dir / "heterogeneity_summary.csv")

    # Find highly heterogeneous perturbations (30-70% responders)
    bimodal = heterogeneity_results[heterogeneity_results["is_bimodal"]]
    if len(bimodal) > 0:
        high_het = bimodal[
            (bimodal["responder_fraction"] > 0.3) & (bimodal["responder_fraction"] < 0.7)
        ]
        high_het.to_csv(output_dir / "high_heterogeneity_perturbations.csv", index=False)
        print(f"\nFound {len(high_het)} highly heterogeneous perturbations (30-70% responders)")

    # Plot heterogeneity comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Fraction bimodal per space
    bimodal_frac = heterogeneity_results.groupby("space")["is_bimodal"].mean()
    axes[0].bar(bimodal_frac.index, bimodal_frac.values)
    axes[0].set_ylabel("Fraction Bimodal")
    axes[0].set_title("Bimodal Response Detection by Space")
    axes[0].tick_params(axis="x", rotation=45)

    # Responder fraction distribution for bimodal perturbations
    for space in embeddings_dict.keys():
        space_bimodal = heterogeneity_results[
            (heterogeneity_results["space"] == space) & (heterogeneity_results["is_bimodal"])
        ]
        if len(space_bimodal) > 0:
            axes[1].hist(space_bimodal["responder_fraction"], bins=20, alpha=0.5, label=space)

    axes[1].set_xlabel("Responder Fraction")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Responder Fraction Distribution (Bimodal Perturbations)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_dir / "heterogeneity_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare embedding spaces with distance, batch, and heterogeneity metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "embedding_dirs",
        type=Path,
        nargs="+",
        help="Directories containing embedding outputs (supports glob patterns)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/comparison"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--control-label",
        type=str,
        default="nontargeting",
        help="Label prefix for control samples",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=10,
        help="Minimum cells per perturbation",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Permutations for significance testing",
    )

    # Analysis selection
    parser.add_argument(
        "--distances",
        action="store_true",
        help="Run distance metrics analysis",
    )
    parser.add_argument(
        "--batch-effects",
        action="store_true",
        help="Run batch effects analysis",
    )
    parser.add_argument(
        "--heterogeneity",
        action="store_true",
        help="Run response heterogeneity analysis",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all analyses (default if none specified)",
    )

    args = parser.parse_args()

    # If no specific analyses requested, run all
    if not (args.distances or args.batch_effects or args.heterogeneity):
        args.all = True

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EMBEDDING SPACE COMPARISON")
    print("=" * 70)
    print(f"Output: {args.output_dir}")
    print()

    # Load embedding spaces
    print("Loading embedding spaces...")
    spaces = load_spaces(args.embedding_dirs)

    if len(spaces) == 0:
        raise ValueError("No embedding spaces loaded!")

    print(f"\nLoaded {len(spaces)} embedding spaces: {list(spaces.keys())}")

    # Get metadata from first space with metadata
    metadata = None
    for name, data in spaces.items():
        if len(data["metadata"]) > 0:
            metadata = data["metadata"]
            break

    if metadata is None or len(metadata) == 0:
        print("\nWarning: No metadata found in embedding outputs.")
        print("Some analyses may not work properly.")
        # Create dummy metadata
        first_space = list(spaces.values())[0]
        n_cells = len(first_space["embeddings"])
        metadata = pd.DataFrame({"index": range(n_cells)})

    # Extract labels from metadata
    # Perturbation labels
    if "gene_symbol_0" in metadata.columns:
        perturbation_labels = metadata["gene_symbol_0"].values.astype(str)
    elif "gene" in metadata.columns:
        perturbation_labels = metadata["gene"].values.astype(str)
    else:
        perturbation_labels = np.array(["unknown"] * len(metadata))

    # Aggregate nontargeting labels
    nontargeting_mask = np.array([str(p).startswith("nontargeting") for p in perturbation_labels])
    perturbation_labels[nontargeting_mask] = "nontargeting"

    # Batch labels
    if "plate" in metadata.columns and "well" in metadata.columns:
        batch_labels = (metadata["plate"].astype(str) + "_" + metadata["well"].astype(str)).values
    elif "plate_well" in metadata.columns:
        batch_labels = metadata["plate_well"].values.astype(str)
    elif "well" in metadata.columns:
        batch_labels = metadata["well"].values.astype(str)
    else:
        batch_labels = np.zeros(len(metadata), dtype=str)

    print("\nData summary:")
    print(f"  Cells: {len(metadata):,}")
    print(f"  Unique perturbations: {len(np.unique(perturbation_labels))}")
    print(f"  Unique batches: {len(np.unique(batch_labels))}")
    print(f"  Control cells: {nontargeting_mask.sum()}")

    # Run analyses
    if args.all or args.distances:
        run_distance_analysis(
            spaces=spaces,
            perturbation_labels=perturbation_labels,
            control_label=args.control_label,
            n_permutations=args.n_permutations,
            min_cells=args.min_cells,
            output_dir=args.output_dir,
        )

    if args.all or args.batch_effects:
        run_batch_analysis(
            spaces=spaces,
            batch_labels=batch_labels,
            perturbation_labels=perturbation_labels,
            output_dir=args.output_dir,
        )

    if args.all or args.heterogeneity:
        run_heterogeneity_analysis(
            spaces=spaces,
            perturbation_labels=perturbation_labels,
            control_label=args.control_label,
            min_cells=args.min_cells,
            output_dir=args.output_dir,
        )

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
