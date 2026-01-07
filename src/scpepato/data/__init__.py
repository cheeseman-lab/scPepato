"""Data loading and processing utilities for scPepato."""

from scpepato.data.features import (
    CellProfilerDataset,
    FeatureNormalizer,
    create_label_encoders,
    load_filtered_parquet,
    load_multiple_filtered_parquets,
    prepare_vae_data,
)
from scpepato.data.features import (
    get_feature_columns as get_cellprofiler_feature_columns,
)
from scpepato.data.loader import (
    LOCATION_COLS,
    METADATA_COLS,
    NON_FEATURE_COLS,
    detect_feature_columns_from_schema,
    get_feature_columns,
    get_parquet_columns,
    get_raw_feature_columns,
    load_aggregated_data,
    load_filtered_data,
    load_single_cell_data,
    split_metadata_features,
    subsample_cells,
    validate_features,
)

__all__ = [
    # Original loader functions
    "load_single_cell_data",
    "load_filtered_data",
    "load_aggregated_data",
    "split_metadata_features",
    "get_feature_columns",
    "get_raw_feature_columns",
    "get_parquet_columns",
    "detect_feature_columns_from_schema",
    "validate_features",
    "subsample_cells",
    "METADATA_COLS",
    "LOCATION_COLS",
    "NON_FEATURE_COLS",
    # VAE data utilities
    "CellProfilerDataset",
    "FeatureNormalizer",
    "create_label_encoders",
    "get_cellprofiler_feature_columns",
    "load_filtered_parquet",
    "load_multiple_filtered_parquets",
    "prepare_vae_data",
]
