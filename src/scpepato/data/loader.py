"""Functions for loading brieflow output data.

This module provides utilities for loading single-cell and aggregated data
from brieflow parquet files, and splitting metadata from features.

Key features:
- Memory-efficient loading via PyArrow with column selection and row sampling
- Automatic metadata/feature column detection for aligned and filtered data
- Stratified subsampling for balanced representation across wells
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Metadata columns (identifiers, not features)
METADATA_COLS = [
    "plate",
    "well",
    "tile",
    "site",
    "cell_0",
    "cell_1",
    "sgRNA_0",
    "gene_symbol_0",
    "mapped_single_gene",
    "channels_min",
    "class",
    "confidence",
    "batch_values",
    # Screen-specific metadata
    "cell_barcode_0",
    "no_recomb_0",
    "gene_symbol_1",
    "no_recomb_1",
]

# Location columns (cell coordinates, for image linking)
LOCATION_COLS = [
    "i_0",
    "j_0",
    "i_1",
    "j_1",
    "distance",
    "fov_distance_0",
    "fov_distance_1",
    "nucleus_i",
    "nucleus_j",
    "nucleus_bounds_0",
    "nucleus_bounds_1",
    "nucleus_bounds_2",
    "nucleus_bounds_3",
    "cell_i",
    "cell_j",
    "cell_bounds_0",
    "cell_bounds_1",
    "cell_bounds_2",
    "cell_bounds_3",
    "cytoplasm_i",
    "cytoplasm_j",
    "cytoplasm_bounds_0",
    "cytoplasm_bounds_1",
    "cytoplasm_bounds_2",
    "cytoplasm_bounds_3",
]

# All non-feature columns
NON_FEATURE_COLS = METADATA_COLS + LOCATION_COLS


def get_parquet_columns(path: Union[str, Path]) -> list[str]:
    """Get column names from a parquet file without loading data.

    Args:
        path: Path to parquet file

    Returns:
        List of column names
    """
    return pq.read_schema(path).names


def detect_feature_columns_from_schema(
    path: Union[str, Path], data_type: str = "auto"
) -> list[str]:
    """Detect feature columns from parquet schema without loading data.

    Args:
        path: Path to parquet file
        data_type: Type of data ("auto", "aligned", "filtered")

    Returns:
        List of feature column names
    """
    cols = get_parquet_columns(path)

    # Check for PC columns (aligned data)
    pc_cols = [c for c in cols if c.startswith("PC_")]

    if data_type == "aligned" or (data_type == "auto" and len(pc_cols) > 0):
        return sorted(pc_cols, key=lambda x: int(x.split("_")[1]))

    # Raw morphological features (filtered data)
    compartments = ("nucleus_", "cell_", "cytoplasm_")
    location_suffixes = (
        "_i",
        "_j",
        "_bounds_",
        "_centroid_",
        "_r0",
        "_c0",
        "_r1",
        "_c1",
        "_location_r",
        "_location_c",
        "_center_mass_r",
        "_center_mass_c",
        "_feret_r",
        "_feret_c",
    )

    feature_cols = []
    for col in cols:
        if col in NON_FEATURE_COLS:
            continue
        if any(col.startswith(comp) for comp in compartments):
            if not any(suffix in col for suffix in location_suffixes):
                feature_cols.append(col)

    return feature_cols


def load_single_cell_data(
    path: Union[str, Path],
    columns: list[str] | None = None,
    n_rows: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load single-cell data from a brieflow parquet file.

    Memory-efficient loading with optional column selection and row sampling.

    Args:
        path: Path to parquet file (e.g., *__filtered.parquet or *__aligned.parquet)
        columns: Optional list of columns to load (for memory efficiency).
                 If None, loads all columns.
        n_rows: Optional number of rows to sample. If None, loads all rows.
                Sampling is done efficiently at the parquet level.
        random_state: Random seed for row sampling reproducibility.

    Returns:
        DataFrame with cell-level data
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Get total row count without loading data
    parquet_file = pq.ParquetFile(path)
    total_rows = parquet_file.metadata.num_rows

    if n_rows is not None and n_rows < total_rows:
        # Sample row indices
        rng = np.random.default_rng(random_state)
        row_indices = np.sort(rng.choice(total_rows, size=n_rows, replace=False))

        # Read in batches and filter - more memory efficient than loading all
        # For very large files, we read row groups and sample within
        table = pq.read_table(path, columns=columns)
        df = table.to_pandas()
        df = df.iloc[row_indices].reset_index(drop=True)
    else:
        df = pd.read_parquet(path, columns=columns)

    return df


def load_filtered_data(
    parquet_dir: Union[str, Path],
    channels: str,
    cell_class: str = "all",
    columns: list[str] | None = None,
    wells: list[str] | None = None,
    n_rows: int | None = None,
    n_rows_per_well: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load and concatenate filtered parquets (raw morphological features).

    Memory-efficient loading with optional column selection and row sampling.
    Filtered parquets are per-well files with raw CellProfiler-like features.

    Args:
        parquet_dir: Directory containing filtered parquet files
        channels: Channel combination (e.g., "DAPI_GLYCORNA")
        cell_class: Cell class filter ("all", "Mitotic", "Interphase")
        columns: Optional list of columns to load (for memory efficiency).
                 Use detect_feature_columns_from_schema() to get feature columns.
        wells: Optional list of wells to load (e.g., ["A1", "A2"]). If None, load all.
        n_rows: Total number of rows to sample across all wells.
                Rows are distributed proportionally to well sizes.
        n_rows_per_well: Number of rows to sample from each well.
                        Takes precedence over n_rows if both specified.
        random_state: Random seed for row sampling reproducibility.

    Returns:
        DataFrame with concatenated cell-level data from all wells
    """
    parquet_dir = Path(parquet_dir)
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Directory not found: {parquet_dir}")

    # Find matching filtered parquets
    pattern = f"P-*_W-*_CeCl-{cell_class}_ChCo-{channels}__filtered.parquet"
    files = list(parquet_dir.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No filtered parquets found matching: {pattern}")

    # Filter by wells if specified
    if wells is not None:
        wells_set = set(wells)
        files = [f for f in files if any(f"_W-{w}_" in f.name for w in wells_set)]

    files = sorted(files)
    rng = np.random.default_rng(random_state)

    # Get row counts for each file (without loading data)
    file_row_counts = {}
    for f in files:
        pf = pq.ParquetFile(f)
        file_row_counts[f] = pf.metadata.num_rows

    total_rows = sum(file_row_counts.values())

    # Determine rows to sample per file
    if n_rows_per_well is not None:
        # Fixed number per well
        rows_per_file = {f: min(n_rows_per_well, count) for f, count in file_row_counts.items()}
    elif n_rows is not None and n_rows < total_rows:
        # Proportional sampling
        rows_per_file = {}
        remaining = n_rows
        for f in files[:-1]:
            proportion = file_row_counts[f] / total_rows
            file_n = int(n_rows * proportion)
            rows_per_file[f] = min(file_n, file_row_counts[f])
            remaining -= rows_per_file[f]
        # Last file gets remainder
        rows_per_file[files[-1]] = min(remaining, file_row_counts[files[-1]])
    else:
        # Load all rows
        rows_per_file = file_row_counts

    # Load and concatenate with sampling
    dfs = []
    for f in files:
        n_to_sample = rows_per_file[f]
        file_total = file_row_counts[f]

        if n_to_sample < file_total:
            # Sample row indices
            row_indices = np.sort(rng.choice(file_total, size=n_to_sample, replace=False))
            table = pq.read_table(f, columns=columns)
            df = table.to_pandas()
            df = df.iloc[row_indices].reset_index(drop=True)
        else:
            df = pd.read_parquet(f, columns=columns)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_aggregated_data(
    path: Union[str, Path],
) -> pd.DataFrame:
    """Load aggregated (perturbation-level) data from brieflow TSV.

    Args:
        path: Path to aggregated TSV file (e.g., *__aggregated.tsv)

    Returns:
        DataFrame with perturbation-level data
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path, sep="\t")


def get_feature_columns(df: pd.DataFrame, data_type: str = "auto") -> list[str]:
    """Get feature column names from a DataFrame.

    Args:
        df: DataFrame with mixed columns
        data_type: Type of data
            - "auto": Detect based on column names (PC_* vs morphological)
            - "aligned": PC_* columns (principal components)
            - "filtered": Raw morphological features

    Returns:
        List of feature column names
    """
    # Check for PC columns (aligned data)
    pc_cols = [c for c in df.columns if c.startswith("PC_")]

    if data_type == "aligned" or (data_type == "auto" and len(pc_cols) > 0):
        return pc_cols

    # Raw morphological features (filtered data)
    # These are columns that start with compartment names and are numeric
    compartments = ("nucleus_", "cell_", "cytoplasm_")
    feature_cols = []

    for col in df.columns:
        # Skip known non-feature columns
        if col in NON_FEATURE_COLS:
            continue

        # Check if it's a compartment-prefixed column
        if any(col.startswith(comp) for comp in compartments):
            # Exclude location columns (bounds, centroid, etc.)
            location_suffixes = (
                "_i",
                "_j",
                "_bounds_",
                "_centroid_",
                "_r0",
                "_c0",
                "_r1",
                "_c1",
                "_location_r",
                "_location_c",
                "_center_mass_r",
                "_center_mass_c",
                "_feret_r",
                "_feret_c",
            )
            if not any(suffix in col for suffix in location_suffixes):
                # Check if numeric
                if df[col].dtype in (np.float64, np.float32, np.int64, np.int32, "Int64"):
                    feature_cols.append(col)

    return feature_cols


def get_raw_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get raw morphological feature column names from filtered data.

    Raw features follow the pattern: {compartment}_{channel}_{feature} or {compartment}_{feature}
    where compartment = nucleus, cell, cytoplasm

    Args:
        df: DataFrame with filtered data

    Returns:
        List of raw feature column names
    """
    return get_feature_columns(df, data_type="filtered")


def split_metadata_features(
    df: pd.DataFrame,
    non_feature_cols: list[str] | None = None,
    data_type: str = "auto",
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Split a cell data DataFrame into metadata and features.

    Args:
        df: DataFrame containing both metadata and feature columns
        non_feature_cols: List of non-feature column names. If None, uses NON_FEATURE_COLS.
        data_type: Type of data ("auto", "aligned", "filtered")

    Returns:
        Tuple of (metadata_df, features_array, feature_names)
        - metadata_df: DataFrame with metadata and location columns
        - features_array: NumPy array of shape (n_cells, n_features)
        - feature_names: List of feature column names
    """
    if non_feature_cols is None:
        non_feature_cols = NON_FEATURE_COLS

    # Get feature columns based on data type
    feature_cols = get_feature_columns(df, data_type=data_type)

    if len(feature_cols) == 0:
        raise ValueError(f"No feature columns found in DataFrame (data_type={data_type})")

    # Only keep metadata columns that exist in the dataframe
    existing_meta_cols = [c for c in non_feature_cols if c in df.columns]

    metadata = df[existing_meta_cols].copy()
    features = df[feature_cols].values.astype(np.float64)  # NumPy array for efficiency

    return metadata, features, feature_cols


def validate_features(features: np.ndarray) -> dict:
    """Validate feature array and return statistics.

    Args:
        features: NumPy array of features

    Returns:
        Dictionary with validation results
    """
    return {
        "shape": features.shape,
        "dtype": features.dtype,
        "has_nan": np.isnan(features).any(),
        "nan_count": np.isnan(features).sum(),
        "has_inf": np.isinf(features).any(),
        "inf_count": np.isinf(features).sum(),
        "min": float(np.nanmin(features)),
        "max": float(np.nanmax(features)),
        "mean": float(np.nanmean(features)),
        "std": float(np.nanstd(features)),
    }


def subsample_cells(
    df: pd.DataFrame,
    n: int,
    stratify_col: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Subsample cells from a DataFrame.

    Args:
        df: DataFrame to subsample
        n: Number of cells to sample
        stratify_col: Optional column to stratify sampling by (e.g., 'well', 'gene_symbol_0')
        random_state: Random seed for reproducibility

    Returns:
        Subsampled DataFrame
    """
    if n >= len(df):
        return df.copy()

    rng = np.random.default_rng(random_state)

    if stratify_col is None:
        idx = rng.choice(len(df), size=n, replace=False)
        return df.iloc[idx].reset_index(drop=True)

    # Stratified sampling
    groups = df.groupby(stratify_col)
    n_groups = len(groups)
    n_per_group = n // n_groups

    sampled_dfs = []
    for _, group_df in groups:
        if len(group_df) <= n_per_group:
            sampled_dfs.append(group_df)
        else:
            idx = rng.choice(len(group_df), size=n_per_group, replace=False)
            sampled_dfs.append(group_df.iloc[idx])

    result = pd.concat(sampled_dfs, ignore_index=True)

    # If we need more samples to reach n, sample from remainder
    if len(result) < n:
        remaining = df.drop(result.index, errors="ignore")
        if len(remaining) > 0:
            extra_n = min(n - len(result), len(remaining))
            extra_idx = rng.choice(len(remaining), size=extra_n, replace=False)
            result = pd.concat([result, remaining.iloc[extra_idx]], ignore_index=True)

    return result
