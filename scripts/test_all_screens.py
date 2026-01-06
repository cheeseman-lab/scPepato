#!/usr/bin/env python3
"""Comprehensive test of data loading and embedding across all primary screens.

Tests:
1. Loading filtered (raw features) data from each screen
2. Loading aligned (PC) data from each screen
3. Running UMAP, PHATE, t-SNE embeddings on aligned data
4. Generating combined visualization plots
"""

import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def log(msg: str):
    """Print with immediate flush for real-time output."""
    print(msg, flush=True)


from scpepato.data import (
    METADATA_COLS,
    detect_feature_columns_from_schema,
    load_filtered_data,
    load_single_cell_data,
    split_metadata_features,
    validate_features,
)
from scpepato.embedding import run_phate, run_tsne, run_umap

# Load screens from CSV
SCREENS_CSV = Path(__file__).parent.parent / "data" / "screens.csv"
N_CELLS_SAMPLE = 5000  # Small sample for quick testing


def load_primary_screens() -> list[dict]:
    """Load primary screen configurations from screens.csv."""
    df = pd.read_csv(SCREENS_CSV)
    # Filter to primary=True and has aligned_parquet
    primary = df[(df["primary"] == True) & (df["aligned_parquet"].notna())]

    screens = []
    for _, row in primary.iterrows():
        screens.append(
            {
                "screen": row["screen"],
                "cell_type": row["cell_type"],
                "channels": row["channels"],
                "aligned": row["aligned_parquet"],
                "filtered_dir": row["filtered_parquet_dir"],
                "has_mozzarellm": pd.notna(row["mozzarellm_path"]),
            }
        )
    return screens


def test_filtered_loading(screen_info: dict) -> dict:
    """Test loading filtered (raw features) data with efficient sampling."""
    log("\n  Loading filtered data (efficient)...")
    t0 = time.time()

    try:
        # Get first filtered file to detect feature columns
        from pathlib import Path

        fdir = Path(screen_info["filtered_dir"])
        pattern = f"P-*_W-*_CeCl-all_ChCo-{screen_info['channels']}__filtered.parquet"
        first_file = next(fdir.glob(pattern))

        # Detect feature columns from schema (no data loaded)
        feature_cols = detect_feature_columns_from_schema(first_file, data_type="filtered")

        # Only load metadata + features (skip location columns for memory)
        cols_to_load = [
            c for c in METADATA_COLS if c in ["well", "gene_symbol_0", "class"]
        ] + feature_cols

        # Load with efficient row sampling at file level
        df = load_filtered_data(
            screen_info["filtered_dir"],
            channels=screen_info["channels"],
            columns=cols_to_load,
            n_rows=N_CELLS_SAMPLE,  # Sample at load time, not after
        )

        # Split metadata/features
        metadata, features, feature_names = split_metadata_features(df, data_type="filtered")

        # Validate
        stats = validate_features(features)

        result = {
            "success": True,
            "n_cells": len(df),
            "n_features": len(feature_names),
            "has_nan": stats["has_nan"],
            "has_inf": stats["has_inf"],
            "time": time.time() - t0,
        }
        log(
            f"    OK: {result['n_cells']:,} cells, {result['n_features']} features ({result['time']:.1f}s)"
        )

    except Exception as e:
        import traceback

        result = {"success": False, "error": str(e), "time": time.time() - t0}
        log(f"    FAILED: {e}")
        traceback.print_exc()

    return result


def test_aligned_loading(screen_info: dict) -> dict:
    """Test loading aligned (PC) data with efficient sampling."""
    log("\n  Loading aligned data (efficient)...")
    t0 = time.time()

    try:
        # Detect feature columns from schema (no data loaded)
        feature_cols = detect_feature_columns_from_schema(
            screen_info["aligned"], data_type="aligned"
        )

        # Only load metadata + PC features (skip location columns)
        cols_to_load = [
            c for c in METADATA_COLS if c in ["well", "gene_symbol_0", "class"]
        ] + feature_cols

        # Load with efficient row sampling
        df = load_single_cell_data(
            screen_info["aligned"],
            columns=cols_to_load,
            n_rows=N_CELLS_SAMPLE,  # Sample at load time
        )

        # Split metadata/features
        metadata, features, feature_names = split_metadata_features(df, data_type="aligned")

        # Validate
        stats = validate_features(features)

        result = {
            "success": True,
            "n_cells": len(df),
            "n_features": len(feature_names),
            "has_nan": stats["has_nan"],
            "has_inf": stats["has_inf"],
            "time": time.time() - t0,
            "features": features,  # Keep for embedding tests
            "metadata": metadata,
        }
        log(
            f"    OK: {result['n_cells']:,} cells, {result['n_features']} PCs ({result['time']:.1f}s)"
        )

    except Exception as e:
        import traceback

        result = {"success": False, "error": str(e), "time": time.time() - t0}
        log(f"    FAILED: {e}")
        traceback.print_exc()

    return result


def test_embeddings(features: np.ndarray, n_pcs: int = 50) -> dict:
    """Test UMAP, PHATE, and t-SNE embeddings."""
    results = {}

    # Use first N PCs (limit to reduce memory)
    n_pcs = min(n_pcs, features.shape[1])
    X = features[:, :n_pcs].copy()  # Copy to allow original to be freed
    del features
    gc.collect()

    # UMAP
    log("\n  Running UMAP...")
    t0 = time.time()
    try:
        embedding = run_umap(X, n_neighbors=15, min_dist=0.1)
        results["umap"] = {
            "success": True,
            "shape": embedding.shape,
            "time": time.time() - t0,
            "embedding": embedding,
        }
        log(f"    OK: shape={embedding.shape} ({results['umap']['time']:.1f}s)")
    except Exception as e:
        results["umap"] = {"success": False, "error": str(e)}
        log(f"    FAILED: {e}")

    # PHATE
    log("\n  Running PHATE...")
    t0 = time.time()
    try:
        embedding = run_phate(X, knn=5, decay=40)
        results["phate"] = {
            "success": True,
            "shape": embedding.shape,
            "time": time.time() - t0,
            "embedding": embedding,
        }
        log(f"    OK: shape={embedding.shape} ({results['phate']['time']:.1f}s)")
    except Exception as e:
        results["phate"] = {"success": False, "error": str(e)}
        log(f"    FAILED: {e}")

    # t-SNE
    log("\n  Running t-SNE...")
    t0 = time.time()
    try:
        embedding = run_tsne(X, perplexity=30)
        results["tsne"] = {
            "success": True,
            "shape": embedding.shape,
            "time": time.time() - t0,
            "embedding": embedding,
        }
        log(f"    OK: shape={embedding.shape} ({results['tsne']['time']:.1f}s)")
    except Exception as e:
        results["tsne"] = {"success": False, "error": str(e)}
        log(f"    FAILED: {e}")

    return results


def plot_embeddings(embeddings: dict, screen: str, channels: str, cell_type: str, output_dir: Path):
    """Generate combined plot with UMAP, PHATE, and t-SNE side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = [("umap", "UMAP"), ("phate", "PHATE"), ("tsne", "t-SNE")]

    for ax, (method, label) in zip(axes, methods):
        emb_data = embeddings.get(method, {})
        if emb_data.get("success") and "embedding" in emb_data:
            emb = emb_data["embedding"]
            ax.scatter(emb[:, 0], emb[:, 1], s=1, alpha=0.5, c="steelblue", rasterized=True)
            ax.set_xlabel(f"{label}_1")
            ax.set_ylabel(f"{label}_2")
            ax.set_title(f"{label} ({emb_data['time']:.1f}s)")
        else:
            ax.text(0.5, 0.5, "FAILED", ha="center", va="center", fontsize=14)
            ax.set_xlabel(f"{label}_1")
            ax.set_ylabel(f"{label}_2")
            ax.set_title(label)

        ax.set_aspect("equal", adjustable="datalim")

    # Main title
    fig.suptitle(
        f"{screen} ({cell_type}): {channels}\n({N_CELLS_SAMPLE:,} cells)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save
    output_path = output_dir / f"{screen}_{channels}_embeddings.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"\n  Saved plot: {output_path}")

    return output_path


def main():
    log("=" * 80)
    log("COMPREHENSIVE TEST: ALL PRIMARY SCREENS")
    log("=" * 80)
    log(f"Sample size: {N_CELLS_SAMPLE:,} cells per screen")

    # Load screen configurations
    screens = load_primary_screens()
    log(f"Found {len(screens)} primary screens to test")

    # Create output directory
    output_dir = Path("outputs/test_all_screens")
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    all_results = {}

    for screen_info in screens:
        screen = screen_info["screen"]
        channels = screen_info["channels"]
        cell_type = screen_info["cell_type"]
        has_mozzarellm = screen_info["has_mozzarellm"]

        log(f"\n{'=' * 80}")
        log(f"SCREEN: {screen} ({cell_type}) / {channels}")
        log(f"Has mozzarellm: {has_mozzarellm}")
        log("=" * 80)

        all_results[f"{screen}_{channels}"] = {
            "screen": screen,
            "cell_type": cell_type,
            "channels": channels,
            "has_mozzarellm": has_mozzarellm,
        }

        # Test 1: Filtered data loading
        all_results[f"{screen}_{channels}"]["filtered"] = test_filtered_loading(screen_info)

        # Test 2: Aligned data loading
        aligned_result = test_aligned_loading(screen_info)
        all_results[f"{screen}_{channels}"]["aligned"] = {
            k: v for k, v in aligned_result.items() if k not in ("features", "metadata")
        }

        # Test 3: Embeddings (only if aligned loading succeeded)
        if aligned_result.get("success"):
            embeddings = test_embeddings(aligned_result["features"])
            all_results[f"{screen}_{channels}"]["embeddings"] = {
                k: {kk: vv for kk, vv in v.items() if kk != "embedding"}
                for k, v in embeddings.items()
            }

            # Generate combined plot
            plot_embeddings(embeddings, screen, channels, cell_type, output_dir)

            # Free memory
            del embeddings
        else:
            all_results[f"{screen}_{channels}"]["embeddings"] = {"skipped": True}

        # Free memory from aligned_result
        del aligned_result
        gc.collect()
        log("\n  Memory cleared for next screen")

    # Summary
    log(f"\n{'=' * 80}")
    log("SUMMARY")
    log("=" * 80)

    for key, results in all_results.items():
        screen = results["screen"]
        channels = results["channels"]
        cell_type = results["cell_type"]
        has_mozzarellm = results["has_mozzarellm"]
        moz_marker = " [mozzarellm]" if has_mozzarellm else ""

        log(f"\n{screen} ({cell_type}) - {channels}{moz_marker}:")

        # Filtered
        filt = results.get("filtered", {})
        if filt.get("success"):
            log(f"  Filtered: OK ({filt['n_cells']:,} cells, {filt['n_features']} features)")
        else:
            log(f"  Filtered: FAILED - {filt.get('error', 'unknown')}")

        # Aligned
        aligned = results.get("aligned", {})
        if aligned.get("success"):
            log(f"  Aligned:  OK ({aligned['n_cells']:,} cells, {aligned['n_features']} PCs)")
        else:
            log(f"  Aligned:  FAILED - {aligned.get('error', 'unknown')}")

        # Embeddings
        emb = results.get("embeddings", {})
        if emb.get("skipped"):
            log("  Embeddings: SKIPPED")
        else:
            for method in ["umap", "phate", "tsne"]:
                m = emb.get(method, {})
                if m.get("success"):
                    log(f"  {method.upper():6s}: OK ({m['time']:.1f}s)")
                else:
                    log(f"  {method.upper():6s}: FAILED - {m.get('error', 'unknown')}")

    # Overall pass/fail
    log(f"\n{'=' * 80}")
    all_passed = all(
        results.get("filtered", {}).get("success")
        and results.get("aligned", {}).get("success")
        and all(
            results.get("embeddings", {}).get(m, {}).get("success")
            for m in ["umap", "phate", "tsne"]
        )
        for results in all_results.values()
    )

    if all_passed:
        log("ALL TESTS PASSED")
    else:
        log("SOME TESTS FAILED")
    log("=" * 80)


if __name__ == "__main__":
    main()
