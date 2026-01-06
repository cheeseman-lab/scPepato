#!/usr/bin/env python3
"""Build data dictionary CSV cataloging all screen data paths."""

import re
from pathlib import Path

import pandas as pd

# Screens to include with metadata
SCREENS = {
    "aconcagua": {"cell_type": "HeLa", "n_genes": 5299},
    "baker": {"cell_type": "HeLa", "n_genes": 507},
    "cotopaxi": {"cell_type": "RPE1", "n_genes": 507},
    "denali": {"cell_type": "HeLa", "n_genes": 5299},
    "etna": {"cell_type": "HeLa", "n_genes": 1441},
    "jebel": {"cell_type": "RPE1", "n_genes": 5299},
    "kilimanjaro": {"cell_type": "HeLa", "n_genes": 5299},
    "mayon": {"cell_type": "HeLa", "n_genes": 5299},
}

# Images are in archive, not lab
IMAGES_BASE = Path("/archive/cheeseman/ops_analysis")

BASE_PATH = Path("/lab/ops_analysis/cheeseman")


def find_mozzarellm_resolution(cluster_channel_path: Path) -> str | None:
    """Find the resolution folder that contains mozzarellm_analysis."""
    all_path = cluster_channel_path / "all"
    if not all_path.exists():
        return None

    for res_dir in all_path.iterdir():
        if res_dir.is_dir() and (res_dir / "mozzarellm_analysis").exists():
            return res_dir.name
    return None


def get_screen_data(screen: str, metadata: dict) -> list[dict]:
    """Get all data paths for a screen."""
    rows = []

    analysis_dir = BASE_PATH / f"{screen}-analysis" / "analysis"
    brieflow_output = analysis_dir / "brieflow_output"

    # Library path
    library_path = analysis_dir / "config" / "barcode_library.tsv"
    library_path_str = str(library_path) if library_path.exists() else ""

    # Images path (in archive)
    images_path = (
        IMAGES_BASE / f"{screen}-analysis" / "analysis" / "brieflow_output" / "phenotype" / "images"
    )
    images_path_str = str(images_path) if images_path.exists() else ""

    if not brieflow_output.exists():
        # Placeholder row for missing screens
        rows.append(
            {
                "screen": screen,
                "cell_type": metadata["cell_type"],
                "n_genes": metadata["n_genes"],
                "channels": "",
                "aligned_parquet": "",
                "filtered_parquet_dir": "",
                "filtered_pattern": "",
                "n_filtered_files": 0,
                "aggregated_tsv": "",
                "cluster_tsv": "",
                "mozzarellm_path": "",
                "library_path": library_path_str,
                "images_path": images_path_str,
            }
        )
        return rows

    parquets_dir = brieflow_output / "aggregate" / "parquets"
    tsvs_dir = brieflow_output / "aggregate" / "tsvs"
    cluster_dir = brieflow_output / "cluster"

    if not parquets_dir.exists():
        return rows

    # Find all CeCl-all aligned parquets
    for parquet_file in parquets_dir.glob("CeCl-all_ChCo-*__aligned.parquet"):
        # Extract channels from filename
        match = re.match(r"CeCl-all_ChCo-(.+)__aligned\.parquet", parquet_file.name)
        if not match:
            continue

        channels = match.group(1)

        # Find corresponding aggregated tsv
        agg_tsv = tsvs_dir / f"CeCl-all_ChCo-{channels}__aggregated.tsv"

        # Find filtered parquets (per-well, raw morphological features)
        filtered_pattern = f"P-*_W-*_CeCl-all_ChCo-{channels}__filtered.parquet"
        filtered_files = list(parquets_dir.glob(filtered_pattern))
        n_filtered = len(filtered_files)

        # Find cluster output
        cluster_channel_dir = cluster_dir / channels
        mozzarellm_res = find_mozzarellm_resolution(cluster_channel_dir)

        if mozzarellm_res:
            cluster_tsv = (
                cluster_channel_dir / "all" / mozzarellm_res / "phate_leiden_clustering.tsv"
            )
            mozzarellm_path = cluster_channel_dir / "all" / mozzarellm_res / "mozzarellm_analysis"
        else:
            cluster_tsv = ""
            mozzarellm_path = ""

        rows.append(
            {
                "screen": screen,
                "cell_type": metadata["cell_type"],
                "n_genes": metadata["n_genes"],
                "channels": channels,
                "aligned_parquet": str(parquet_file) if parquet_file.exists() else "",
                "filtered_parquet_dir": str(parquets_dir) if n_filtered > 0 else "",
                "filtered_pattern": filtered_pattern if n_filtered > 0 else "",
                "n_filtered_files": n_filtered,
                "aggregated_tsv": str(agg_tsv) if agg_tsv.exists() else "",
                "cluster_tsv": str(cluster_tsv)
                if cluster_tsv and Path(cluster_tsv).exists()
                else "",
                "mozzarellm_path": str(mozzarellm_path)
                if mozzarellm_path and Path(mozzarellm_path).exists()
                else "",
                "library_path": library_path_str,
                "images_path": images_path_str,
            }
        )

    return rows


def main():
    all_rows = []

    for screen, metadata in SCREENS.items():
        print(f"Processing {screen}...")
        rows = get_screen_data(screen, metadata)
        all_rows.extend(rows)
        print(f"  Found {len(rows)} channel combos")

    df = pd.DataFrame(all_rows)

    # Create output directory
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "screens.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Total rows: {len(df)}")
    print(df.to_string())


if __name__ == "__main__":
    main()
