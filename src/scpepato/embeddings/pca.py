"""PCA embedding methods with optional batch correction.

This module provides two PCA methods:
1. `vanilla` - Standard PCA with feature normalization
2. `batch_corrected` - Batch center-scaling + PCA + TVN (Typical Variation Normalization)

The batch_corrected method applies three key transformations:
1. Pre-PCA batch centering: StandardScaler applied per batch
2. PCA transformation to reduce dimensionality
3. TVN post-processing: Uses control samples to normalize across batches
   with CORAL-style covariance alignment
"""

from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scpepato.embeddings.base import save_embedding


def centerscale_by_batch(
    features: np.ndarray,
    metadata: Optional[pd.DataFrame] = None,
    batch_col: Optional[str] = None,
) -> np.ndarray:
    """Center and scale features by batch using StandardScaler.

    Applies StandardScaler independently to each batch. This removes
    batch-specific location and scale effects before PCA.

    Parameters
    ----------
    features : np.ndarray
        Input features (n_samples, n_features)
    metadata : pd.DataFrame, optional
        Metadata with batch information
    batch_col : str, optional
        Column name for batch labels. If None, scales entire dataset.

    Returns
    -------
    np.ndarray
        Centered and scaled features
    """
    features = features.copy()

    if batch_col is None:
        features = StandardScaler(copy=False).fit_transform(features)
    else:
        if metadata is None:
            raise ValueError("metadata must be provided if batch_col is not None")
        batches = metadata[batch_col].unique()
        for batch in batches:
            ind = metadata[batch_col] == batch
            features[ind, :] = StandardScaler(copy=False).fit_transform(features[ind, :])

    return features


def centerscale_on_controls(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    control_key: str,
    batch_col: Optional[str] = None,
) -> np.ndarray:
    """Center and scale embeddings using control samples as reference.

    Uses the mean and std of control samples to normalize all samples.
    If batch_col is provided, normalizes within each batch independently.

    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings to normalize (n_samples, n_dims)
    metadata : pd.DataFrame
        Metadata with perturbation and batch information
    pert_col : str
        Column name for perturbation labels
    control_key : str
        Label prefix for control samples (e.g., "nontargeting")
    batch_col : str, optional
        Column name for batch labels. If None, normalizes globally.

    Returns
    -------
    np.ndarray
        Normalized embeddings
    """
    embeddings = embeddings.copy()

    # Boolean mask for control rows
    ctrl_mask_all = metadata[pert_col].astype(str).str.startswith(control_key)

    if batch_col is not None:
        for batch in metadata[batch_col].unique():
            batch_mask = metadata[batch_col] == batch
            batch_ctrl_mask = batch_mask & ctrl_mask_all

            # Only scale if we have controls in this batch
            if batch_ctrl_mask.sum() == 0:
                continue

            embeddings[batch_mask] = (
                StandardScaler(copy=False)
                .fit(embeddings[batch_ctrl_mask])
                .transform(embeddings[batch_mask])
            )

        return embeddings

    # No batching: use all controls
    return StandardScaler(copy=False).fit(embeddings[ctrl_mask_all]).transform(embeddings)


def tvn_on_controls(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    control_key: str,
    batch_col: Optional[str] = None,
) -> np.ndarray:
    """Apply Typical Variation Normalization (TVN) using control samples.

    TVN normalizes the covariance structure across batches using control samples.
    This is a CORAL-style alignment that:
    1. Centers and scales based on control samples
    2. Rotates to align with control principal axes (without dim reduction)
    3. Re-centers and scales per batch using controls
    4. Applies covariance whitening/coloring to align batch covariances

    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings to normalize (n_samples, n_dims)
    metadata : pd.DataFrame
        Metadata with perturbation and batch information
    pert_col : str
        Column name for perturbation labels
    control_key : str
        Label prefix for control samples (e.g., "nontargeting")
    batch_col : str, optional
        Column name for batch labels for per-batch CORAL alignment

    Returns
    -------
    np.ndarray
        TVN-normalized embeddings (same dimensionality as input)
    """
    embeddings = embeddings.copy()
    n_dims = embeddings.shape[1]

    # Step 1: Center and scale based on control samples
    embeddings = centerscale_on_controls(embeddings, metadata, pert_col, control_key)

    # Step 2: PCA rotation on controls to align principal axes
    # Use n_components to preserve dimensionality
    ctrl_mask = metadata[pert_col].astype(str).str.startswith(control_key)
    n_controls = ctrl_mask.sum()

    if n_controls > n_dims:
        # Enough controls: full rotation
        pca = PCA(n_components=n_dims)
        pca.fit(embeddings[ctrl_mask])
        embeddings = pca.transform(embeddings)
    else:
        # Fewer controls than dims: use all available components
        pca = PCA(n_components=min(n_controls - 1, n_dims))
        pca.fit(embeddings[ctrl_mask])
        # Only transform the fitted components, keep rest as-is
        transformed = pca.transform(embeddings)
        if transformed.shape[1] < n_dims:
            # Pad with original remaining dimensions
            embeddings = np.hstack([transformed, embeddings[:, transformed.shape[1] :]])
        else:
            embeddings = transformed

    # Step 3: Re-center and scale per batch using controls
    embeddings = centerscale_on_controls(embeddings, metadata, pert_col, control_key, batch_col)

    # Step 4: CORAL-style covariance alignment
    if batch_col is not None:
        n_current_dims = embeddings.shape[1]

        # Target covariance: pooled controls + regularization
        target_cov = np.cov(embeddings[ctrl_mask], rowvar=False, ddof=1) + 0.5 * np.eye(
            n_current_dims
        )

        batches = metadata[batch_col].unique()
        for batch in batches:
            batch_mask = metadata[batch_col] == batch
            batch_ctrl_mask = batch_mask & ctrl_mask

            if batch_ctrl_mask.sum() < 2:
                continue

            # Source covariance: this batch's controls + regularization
            source_cov = np.cov(embeddings[batch_ctrl_mask], rowvar=False, ddof=1) + 0.5 * np.eye(
                n_current_dims
            )

            # Whiten with source, color with target
            # Take real part to handle numerical precision issues
            whiten = linalg.fractional_matrix_power(source_cov, -0.5)
            color = linalg.fractional_matrix_power(target_cov, 0.5)

            embeddings[batch_mask] = np.matmul(embeddings[batch_mask], np.real(whiten))
            embeddings[batch_mask] = np.matmul(embeddings[batch_mask], np.real(color))

    return embeddings


def compute_pca(
    features: np.ndarray,
    metadata: pd.DataFrame,
    method: Literal["vanilla", "batch_corrected"] = "vanilla",
    n_components: int = 50,
    batch_col: Optional[str] = None,
    pert_col: Optional[str] = None,
    control_key: str = "nontargeting",
    random_state: int = 42,
) -> tuple[np.ndarray, PCA, dict]:
    """Compute PCA embeddings with optional batch correction.

    Parameters
    ----------
    features : np.ndarray
        Raw features (n_samples, n_features)
    metadata : pd.DataFrame
        Cell metadata
    method : {'vanilla', 'batch_corrected'}
        - 'vanilla': Standard PCA with global normalization
        - 'batch_corrected': Batch center-scaling + PCA + TVN
    n_components : int
        Number of PCA components
    batch_col : str, optional
        Column name for batch labels. Required for batch_corrected method.
    pert_col : str, optional
        Column name for perturbation labels. Required for batch_corrected method.
    control_key : str
        Label prefix for control samples
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    embeddings : np.ndarray
        PCA embeddings (n_samples, n_components)
    pca_model : PCA
        Fitted PCA model
    config : dict
        Configuration dictionary with method details
    """
    if method == "vanilla":
        # Standard approach: global normalization + PCA
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        pca = PCA(n_components=n_components, random_state=random_state)
        embeddings = pca.fit_transform(features_scaled)

        config = {
            "method": "pca",
            "variant": "vanilla",
            "n_components": n_components,
            "variance_explained": float(pca.explained_variance_ratio_.sum()),
            "random_state": random_state,
        }

    elif method == "batch_corrected":
        # Batch correction approach: per-batch scaling + PCA + TVN
        if batch_col is None:
            raise ValueError("batch_col required for batch_corrected method")
        if pert_col is None:
            raise ValueError("pert_col required for batch_corrected method")

        # Step 1: Batch center-scaling
        features_scaled = centerscale_by_batch(features, metadata, batch_col)

        # Step 2: PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        embeddings = pca.fit_transform(features_scaled)

        # Step 3: TVN normalization
        embeddings = tvn_on_controls(embeddings, metadata, pert_col, control_key, batch_col)

        config = {
            "method": "pca_batch_corrected",
            "variant": "batch_corrected",
            "n_components": n_components,
            "variance_explained": float(pca.explained_variance_ratio_.sum()),
            "batch_col": batch_col,
            "pert_col": pert_col,
            "control_key": control_key,
            "random_state": random_state,
        }

    else:
        raise ValueError(f"Unknown method: {method}. Use 'vanilla' or 'batch_corrected'")

    return embeddings, pca, config


def save_pca_embedding(
    output_dir: Union[str, Path],
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pca_model: PCA,
    scaler: StandardScaler,
    method: str,
    config: dict,
) -> None:
    """Save PCA embedding output in standardized format.

    Parameters
    ----------
    output_dir : Path
        Directory to save outputs
    embeddings : np.ndarray
        PCA embeddings
    metadata : pd.DataFrame
        Cell metadata
    pca_model : PCA
        Fitted PCA model
    scaler : StandardScaler
        Feature scaler (or None for batch_corrected)
    method : str
        Method name for manifest
    config : dict
        Configuration dictionary
    """
    import pickle

    output_dir = Path(output_dir)

    # Use base save function for standard files
    save_embedding(output_dir, embeddings, metadata, method, config)

    # Save PCA model
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(pca_model, f)

    # Save scaler if provided
    if scaler is not None:
        np.savez(
            output_dir / "normalizer.npz",
            mean=scaler.mean_,
            std=scaler.scale_,
        )
