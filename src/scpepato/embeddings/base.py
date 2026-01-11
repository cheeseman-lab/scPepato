"""Base classes and utilities for embedding outputs.

This module provides standardized data structures and loading utilities
for embedding outputs from PCA, VAE, and other dimensionality reduction methods.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class EmbeddingOutput:
    """Container for embedding outputs with metadata.

    All embedding methods (PCA, VAE, etc.) produce outputs in this format
    for uniform downstream consumption.
    """

    embeddings: np.ndarray  # (n_cells, n_dims)
    metadata: pd.DataFrame  # Cell metadata (plate, well, gene, etc.)
    config: dict  # Method-specific configuration
    method: str  # 'pca', 'pca_batch_corrected', 'vanilla_vae', etc.

    @property
    def n_cells(self) -> int:
        return self.embeddings.shape[0]

    @property
    def n_dims(self) -> int:
        return self.embeddings.shape[1]


def load_embedding(output_dir: Union[str, Path]) -> EmbeddingOutput:
    """Load embedding output from a standardized directory.

    Parameters
    ----------
    output_dir : Path
        Directory containing embeddings.npy, metadata.parquet, manifest.json

    Returns
    -------
    EmbeddingOutput
        Container with embeddings, metadata, config, and method name
    """
    output_dir = Path(output_dir)

    # Load manifest
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            config = json.load(f)
        # Try different keys for method name
        method = config.get("method")
        if method is None:
            # Legacy VAE format: construct from model_type
            model_type = config.get("model_type")
            if model_type:
                method = f"{model_type.replace('-', '_')}_vae"
            else:
                method = "unknown"
    else:
        config = {}
        # Try to infer method from directory name for legacy outputs
        dir_name = output_dir.name
        if "vanilla" in dir_name:
            method = "vanilla_vae"
        elif "batch" in dir_name or "batch-aware" in dir_name:
            method = "batch_aware_vae"
        elif "conditional" in dir_name:
            method = "conditional_vae"
        elif "pca" in dir_name:
            method = "pca"
        else:
            method = "unknown"

    # Load embeddings (try different naming conventions)
    if (output_dir / "embeddings.npy").exists():
        embeddings = np.load(output_dir / "embeddings.npy")
    elif (output_dir / "train_embeddings.npy").exists():
        # Legacy VAE format: combine train and val embeddings
        train_emb = np.load(output_dir / "train_embeddings.npy")
        if (output_dir / "val_embeddings.npy").exists():
            val_emb = np.load(output_dir / "val_embeddings.npy")
            embeddings = np.vstack([train_emb, val_emb])
        else:
            embeddings = train_emb
    else:
        raise FileNotFoundError(f"No embeddings found in {output_dir}")

    # Load metadata
    metadata_path = output_dir / "metadata.parquet"
    if metadata_path.exists():
        metadata = pd.read_parquet(metadata_path)
    else:
        metadata = pd.DataFrame()

    return EmbeddingOutput(
        embeddings=embeddings,
        metadata=metadata,
        config=config,
        method=method,
    )


def save_embedding(
    output_dir: Union[str, Path],
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    method: str,
    config: Optional[dict] = None,
    model_path: Optional[Path] = None,
    normalizer_path: Optional[Path] = None,
) -> None:
    """Save embedding output in standardized format.

    Parameters
    ----------
    output_dir : Path
        Directory to save outputs
    embeddings : np.ndarray
        Embedding array (n_cells, n_dims)
    metadata : pd.DataFrame
        Cell metadata
    method : str
        Embedding method name
    config : dict, optional
        Method-specific configuration
    model_path : Path, optional
        Path to model file (will be copied to output_dir)
    normalizer_path : Path, optional
        Path to normalizer file (will be copied to output_dir)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    np.save(output_dir / "embeddings.npy", embeddings)

    # Save metadata
    metadata.to_parquet(output_dir / "metadata.parquet")

    # Build manifest
    manifest = {
        "method": method,
        "n_cells": embeddings.shape[0],
        "n_dims": embeddings.shape[1],
        "created": datetime.now().isoformat(),
        "files": {
            "embeddings": "embeddings.npy",
            "metadata": "metadata.parquet",
        },
    }

    if config:
        manifest["config"] = config

    # Save manifest
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load_multiple_embeddings(
    pattern: str, base_dir: Union[str, Path] = "outputs/embeddings"
) -> dict[str, EmbeddingOutput]:
    """Load multiple embedding outputs matching a pattern.

    Parameters
    ----------
    pattern : str
        Glob pattern for directories, e.g., "*/mayon_20000"
    base_dir : Path
        Base directory to search in

    Returns
    -------
    dict
        Mapping from method name to EmbeddingOutput
    """
    base_dir = Path(base_dir)
    results = {}

    for path in sorted(base_dir.glob(pattern)):
        if path.is_dir() and (path / "embeddings.npy").exists():
            emb = load_embedding(path)
            results[emb.method] = emb

    return results
