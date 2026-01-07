"""CellProfiler feature loading and preprocessing for VAE training.

This module provides utilities for loading raw CellProfiler features from
filtered parquet files and preparing them for VAE training.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

# Metadata columns that should not be used as features
METADATA_COLS = [
    "plate",
    "well",
    "tile",
    "cell_0",
    "i_0",
    "j_0",
    "site",
    "cell_1",
    "i_1",
    "j_1",
    "distance",
    "fov_distance_0",
    "fov_distance_1",
    "sgRNA_0",
    "gene_symbol_0",
    "mapped_single_gene",
    "channels_min",
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
    "class",
    "confidence",
]


def get_feature_columns(parquet_path: str | Path) -> list[str]:
    """Get list of CellProfiler feature columns from a parquet file.

    Args:
        parquet_path: Path to a filtered parquet file

    Returns:
        List of feature column names (excluding metadata)
    """
    schema = pq.read_schema(parquet_path)
    all_cols = schema.names
    feature_cols = [c for c in all_cols if c not in METADATA_COLS]
    return feature_cols


def load_filtered_parquet(
    parquet_path: str | Path,
    feature_cols: list[str] | None = None,
    metadata_cols: list[str] | None = None,
    n_rows: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a filtered parquet file, returning features and metadata separately.

    Args:
        parquet_path: Path to filtered parquet file
        feature_cols: List of feature columns to load (None = all features)
        metadata_cols: List of metadata columns to load (None = default set)
        n_rows: Number of rows to sample (None = all rows)
        random_state: Random seed for sampling

    Returns:
        Tuple of (features_df, metadata_df)
    """
    parquet_path = Path(parquet_path)

    # Determine columns to load
    if feature_cols is None:
        feature_cols = get_feature_columns(parquet_path)

    if metadata_cols is None:
        # Default metadata for VAE training
        metadata_cols = ["plate", "well", "tile", "gene_symbol_0", "sgRNA_0", "class"]

    # Only load columns that exist
    schema = pq.read_schema(parquet_path)
    available_cols = set(schema.names)
    feature_cols = [c for c in feature_cols if c in available_cols]
    metadata_cols = [c for c in metadata_cols if c in available_cols]

    columns = feature_cols + metadata_cols

    # Load data
    df = pd.read_parquet(parquet_path, columns=columns)

    # Sample if requested
    if n_rows is not None and n_rows < len(df):
        df = df.sample(n=n_rows, random_state=random_state)

    # Split features and metadata
    features_df = df[feature_cols]
    metadata_df = df[metadata_cols]

    return features_df, metadata_df


def load_multiple_filtered_parquets(
    parquet_dir: str | Path,
    pattern: str = "*__filtered.parquet",
    feature_cols: list[str] | None = None,
    metadata_cols: list[str] | None = None,
    n_rows_per_file: int | None = None,
    n_total_rows: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load multiple filtered parquet files from a directory.

    Args:
        parquet_dir: Directory containing filtered parquet files
        pattern: Glob pattern for parquet files
        feature_cols: List of feature columns to load
        metadata_cols: List of metadata columns to load
        n_rows_per_file: Max rows to sample from each file
        n_total_rows: Max total rows (sampled after combining)
        random_state: Random seed for sampling

    Returns:
        Tuple of (features_df, metadata_df)
    """
    parquet_dir = Path(parquet_dir)
    parquet_files = sorted(parquet_dir.glob(pattern))

    if not parquet_files:
        raise FileNotFoundError(f"No files matching {pattern} in {parquet_dir}")

    # Get feature columns from first file if not specified
    if feature_cols is None:
        feature_cols = get_feature_columns(parquet_files[0])

    all_features = []
    all_metadata = []

    for pq_file in parquet_files:
        features, metadata = load_filtered_parquet(
            pq_file,
            feature_cols=feature_cols,
            metadata_cols=metadata_cols,
            n_rows=n_rows_per_file,
            random_state=random_state,
        )
        all_features.append(features)
        all_metadata.append(metadata)

    features_df = pd.concat(all_features, ignore_index=True)
    metadata_df = pd.concat(all_metadata, ignore_index=True)

    # Final sampling if requested
    if n_total_rows is not None and n_total_rows < len(features_df):
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(features_df), size=n_total_rows, replace=False)
        features_df = features_df.iloc[indices].reset_index(drop=True)
        metadata_df = metadata_df.iloc[indices].reset_index(drop=True)

    return features_df, metadata_df


class FeatureNormalizer:
    """Z-score normalizer for CellProfiler features.

    Computes mean and std from training data, applies to train/val/test.
    Handles constant features by setting std to 1.
    """

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.feature_names: list[str] | None = None

    def fit(self, features: pd.DataFrame | np.ndarray) -> "FeatureNormalizer":
        """Compute mean and std from features.

        Args:
            features: Feature array or DataFrame

        Returns:
            self
        """
        if isinstance(features, pd.DataFrame):
            self.feature_names = features.columns.tolist()
            features = features.values.astype(np.float32)
        else:
            features = features.astype(np.float32)

        self.mean = np.nanmean(features, axis=0)
        self.std = np.nanstd(features, axis=0)

        # Handle constant features (std=0)
        self.std[self.std < 1e-8] = 1.0

        return self

    def transform(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Apply z-score normalization.

        Args:
            features: Feature array or DataFrame

        Returns:
            Normalized features as numpy array
        """
        if self.mean is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        if isinstance(features, pd.DataFrame):
            features = features.values.astype(np.float32)
        else:
            features = features.astype(np.float32)

        normalized = (features - self.mean) / self.std
        return normalized.astype(np.float32)

    def fit_transform(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(features).transform(features)

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Convert normalized features back to original scale."""
        if self.mean is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        return normalized * self.std + self.mean

    def save(self, path: str | Path) -> None:
        """Save normalizer parameters to npz file."""
        np.savez(
            path,
            mean=self.mean,
            std=self.std,
            feature_names=np.array(self.feature_names) if self.feature_names else None,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        """Load normalizer parameters from npz file."""
        data = np.load(path, allow_pickle=True)
        normalizer = cls()
        normalizer.mean = data["mean"]
        normalizer.std = data["std"]
        if data["feature_names"] is not None:
            normalizer.feature_names = data["feature_names"].tolist()
        return normalizer


class CellProfilerDataset(Dataset):
    """PyTorch Dataset for CellProfiler features.

    Supports loading features with batch and perturbation labels for
    conditional VAE training.
    """

    def __init__(
        self,
        features: np.ndarray,
        batch_labels: np.ndarray | None = None,
        perturbation_labels: np.ndarray | None = None,
    ):
        """Initialize dataset.

        Args:
            features: Normalized feature array (n_cells, n_features)
            batch_labels: Integer batch labels (n_cells,) - e.g., well index
            perturbation_labels: Integer perturbation labels (n_cells,) - e.g., gene index
        """
        self.features = torch.from_numpy(features).float()
        self.batch_labels = (
            torch.from_numpy(batch_labels).long() if batch_labels is not None else None
        )
        self.perturbation_labels = (
            torch.from_numpy(perturbation_labels).long()
            if perturbation_labels is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Returns:
            Dictionary with 'features' and optionally 'batch' and 'perturbation'
        """
        sample = {"features": self.features[idx]}

        if self.batch_labels is not None:
            sample["batch"] = self.batch_labels[idx]

        if self.perturbation_labels is not None:
            sample["perturbation"] = self.perturbation_labels[idx]

        return sample


def create_label_encoders(
    metadata: pd.DataFrame,
    batch_col: str = "well",
    perturbation_col: str = "gene_symbol_0",
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """Create label encoders for batch and perturbation columns.

    Args:
        metadata: Metadata DataFrame
        batch_col: Column to use for batch labels. Use "plate_well" to auto-create
                   from plate and well columns.
        perturbation_col: Column to use for perturbation labels

    Returns:
        Tuple of (batch_encoder, pert_encoder, batch_labels, pert_labels)
        Encoders are dicts mapping string -> int
    """
    # Create plate_well column if requested
    if batch_col == "plate_well" and "plate_well" not in metadata.columns:
        metadata = metadata.copy()
        metadata["plate_well"] = metadata["plate"].astype(str) + "_" + metadata["well"].astype(str)

    # Batch encoder
    batch_values = metadata[batch_col].unique()
    batch_encoder = {v: i for i, v in enumerate(sorted(batch_values))}
    batch_labels = metadata[batch_col].map(batch_encoder).values

    # Perturbation encoder
    pert_values = metadata[perturbation_col].unique()
    pert_encoder = {v: i for i, v in enumerate(sorted(pert_values))}
    pert_labels = metadata[perturbation_col].map(pert_encoder).values

    return batch_encoder, pert_encoder, batch_labels, pert_labels


def prepare_vae_data(
    parquet_path: str | Path | None = None,
    parquet_dir: str | Path | None = None,
    pattern: str = "*__filtered.parquet",
    n_rows: int | None = None,
    val_fraction: float = 0.2,
    batch_col: str = "well",
    perturbation_col: str = "gene_symbol_0",
    random_state: int = 42,
) -> dict:
    """Prepare data for VAE training.

    Loads features, normalizes, creates train/val split, and returns
    everything needed for training.

    Args:
        parquet_path: Single parquet file to load
        parquet_dir: Directory of parquet files to load (alternative to parquet_path)
        pattern: Glob pattern for parquet files (if using parquet_dir)
        n_rows: Number of rows to sample
        val_fraction: Fraction of data for validation
        batch_col: Column for batch labels
        perturbation_col: Column for perturbation labels
        random_state: Random seed

    Returns:
        Dictionary with:
            - train_dataset: CellProfilerDataset for training
            - val_dataset: CellProfilerDataset for validation
            - normalizer: Fitted FeatureNormalizer
            - batch_encoder: Dict mapping batch values to indices
            - pert_encoder: Dict mapping perturbation values to indices
            - n_features: Number of input features
            - n_batches: Number of unique batches
            - n_perturbations: Number of unique perturbations
    """
    # Load data
    if parquet_path is not None:
        features_df, metadata_df = load_filtered_parquet(
            parquet_path, n_rows=n_rows, random_state=random_state
        )
    elif parquet_dir is not None:
        features_df, metadata_df = load_multiple_filtered_parquets(
            parquet_dir,
            pattern=pattern,
            n_total_rows=n_rows,
            random_state=random_state,
        )
    else:
        raise ValueError("Must provide either parquet_path or parquet_dir")

    # Create label encoders
    batch_encoder, pert_encoder, batch_labels, pert_labels = create_label_encoders(
        metadata_df, batch_col=batch_col, perturbation_col=perturbation_col
    )

    # Train/val split
    n_samples = len(features_df)
    n_val = int(n_samples * val_fraction)
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # Fit normalizer on training data only
    normalizer = FeatureNormalizer()
    normalizer.fit(features_df.iloc[train_indices])

    # Normalize all data
    train_features = normalizer.transform(features_df.iloc[train_indices])
    val_features = normalizer.transform(features_df.iloc[val_indices])

    # Create datasets
    train_dataset = CellProfilerDataset(
        features=train_features,
        batch_labels=batch_labels[train_indices],
        perturbation_labels=pert_labels[train_indices],
    )

    val_dataset = CellProfilerDataset(
        features=val_features,
        batch_labels=batch_labels[val_indices],
        perturbation_labels=pert_labels[val_indices],
    )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "normalizer": normalizer,
        "batch_encoder": batch_encoder,
        "pert_encoder": pert_encoder,
        "n_features": train_features.shape[1],
        "n_batches": len(batch_encoder),
        "n_perturbations": len(pert_encoder),
        "feature_names": normalizer.feature_names,
    }
