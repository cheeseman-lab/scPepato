"""Embedding and dimensionality reduction methods for single-cell data.

This module provides wrappers around common dimensionality reduction methods
(PCA, UMAP, PHATE) with consistent interfaces for single-cell morphological data.
"""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler


def normalize_features(
    features: pd.DataFrame | np.ndarray,
    method: Literal["zscore", "robust", "none"] = "zscore",
) -> np.ndarray:
    """Normalize feature matrix.

    Args:
        features: Feature matrix (cells x features)
        method: Normalization method
            - "zscore": Standard z-score normalization (mean=0, std=1)
            - "robust": Robust scaling using median and IQR
            - "none": No normalization

    Returns:
        Normalized feature matrix as numpy array
    """
    if isinstance(features, pd.DataFrame):
        features = features.values

    if method == "zscore":
        scaler = StandardScaler()
        return scaler.fit_transform(features)
    elif method == "robust":
        scaler = RobustScaler()
        return scaler.fit_transform(features)
    elif method == "none":
        return features.astype(np.float64)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def run_pca(
    features: pd.DataFrame | np.ndarray,
    n_components: int = 50,
    normalize: Literal["zscore", "robust", "none"] = "zscore",
    random_state: int = 42,
) -> tuple[np.ndarray, PCA]:
    """Run PCA on feature matrix.

    Args:
        features: Feature matrix (cells x features)
        n_components: Number of principal components to compute
        normalize: Normalization method to apply before PCA
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (transformed data, fitted PCA object)
    """
    # Normalize
    X = normalize_features(features, method=normalize)

    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Fit PCA
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)

    return X_pca, pca


def run_umap(
    features: pd.DataFrame | np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    normalize: Literal["zscore", "robust", "none"] = "none",
    random_state: int = 42,
) -> np.ndarray:
    """Run UMAP on feature matrix.

    For best results, input should typically be PCA-reduced features.

    Args:
        features: Feature matrix (cells x features), often PCA-reduced
        n_components: Number of UMAP dimensions (usually 2)
        n_neighbors: Number of neighbors for UMAP graph
        min_dist: Minimum distance between points in embedding
        metric: Distance metric for UMAP
        normalize: Normalization method (usually "none" if input is PCA)
        random_state: Random seed for reproducibility

    Returns:
        UMAP embedding (cells x n_components)
    """
    import umap

    # Normalize if requested
    X = normalize_features(features, method=normalize)

    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Run UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    X_umap = reducer.fit_transform(X)

    return X_umap


def run_phate(
    features: pd.DataFrame | np.ndarray,
    n_components: int = 2,
    knn: int = 5,
    decay: int = 40,
    t: str | int = "auto",
    normalize: Literal["zscore", "robust", "none"] = "none",
    random_state: int = 42,
) -> np.ndarray:
    """Run PHATE on feature matrix.

    PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding)
    is particularly good at preserving both local and global structure.

    Args:
        features: Feature matrix (cells x features), often PCA-reduced
        n_components: Number of PHATE dimensions (usually 2)
        knn: Number of nearest neighbors for kernel
        decay: Decay rate for kernel
        t: Diffusion time ("auto" or integer)
        normalize: Normalization method (usually "none" if input is PCA)
        random_state: Random seed for reproducibility

    Returns:
        PHATE embedding (cells x n_components)
    """
    import phate

    # Normalize if requested
    X = normalize_features(features, method=normalize)

    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Run PHATE
    phate_op = phate.PHATE(
        n_components=n_components,
        knn=knn,
        decay=decay,
        t=t,
        random_state=random_state,
        verbose=0,
    )
    X_phate = phate_op.fit_transform(X)

    return X_phate


def run_tsne(
    features: pd.DataFrame | np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float | str = "auto",
    max_iter: int = 1000,
    metric: str = "euclidean",
    normalize: Literal["zscore", "robust", "none"] = "none",
    random_state: int = 42,
) -> np.ndarray:
    """Run t-SNE on feature matrix.

    For best results, input should typically be PCA-reduced features (50-100 dims).

    Args:
        features: Feature matrix (cells x features), often PCA-reduced
        n_components: Number of t-SNE dimensions (usually 2)
        perplexity: Perplexity parameter (related to number of neighbors)
        learning_rate: Learning rate ("auto" or float)
        max_iter: Maximum number of iterations
        metric: Distance metric
        normalize: Normalization method (usually "none" if input is PCA)
        random_state: Random seed for reproducibility

    Returns:
        t-SNE embedding (cells x n_components)
    """
    from sklearn.manifold import TSNE

    # Normalize if requested
    X = normalize_features(features, method=normalize)

    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Adjust perplexity if needed (must be < n_samples)
    n_samples = X.shape[0]
    if perplexity >= n_samples:
        perplexity = max(5.0, n_samples // 3)

    # Run t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        metric=metric,
        random_state=random_state,
        init="pca",
    )
    X_tsne = tsne.fit_transform(X)

    return X_tsne


def compute_embedding_pipeline(
    features: pd.DataFrame | np.ndarray,
    method: Literal["pca_umap", "pca_phate", "pca_tsne", "umap", "phate", "tsne"] = "pca_umap",
    pca_components: int = 50,
    final_components: int = 2,
    normalize: Literal["zscore", "robust", "none"] = "zscore",
    random_state: int = 42,
    **kwargs,
) -> dict:
    """Run a complete embedding pipeline.

    Args:
        features: Feature matrix (cells x features)
        method: Embedding pipeline to run
            - "pca_umap": PCA then UMAP (recommended)
            - "pca_phate": PCA then PHATE
            - "pca_tsne": PCA then t-SNE
            - "umap": Direct UMAP on normalized features
            - "phate": Direct PHATE on normalized features
            - "tsne": Direct t-SNE on normalized features
        pca_components: Number of PCA components (if using PCA)
        final_components: Number of final embedding dimensions
        normalize: Normalization method
        random_state: Random seed
        **kwargs: Additional arguments passed to UMAP/PHATE/t-SNE

    Returns:
        Dictionary with:
            - "embedding": Final 2D embedding
            - "pca": PCA result (if applicable)
            - "pca_model": Fitted PCA object (if applicable)
            - "method": Method used
    """
    result = {"method": method}

    if method == "pca_umap":
        X_pca, pca_model = run_pca(
            features,
            n_components=pca_components,
            normalize=normalize,
            random_state=random_state,
        )
        result["pca"] = X_pca
        result["pca_model"] = pca_model

        X_embed = run_umap(
            X_pca,
            n_components=final_components,
            normalize="none",
            random_state=random_state,
            **kwargs,
        )
        result["embedding"] = X_embed

    elif method == "pca_phate":
        X_pca, pca_model = run_pca(
            features,
            n_components=pca_components,
            normalize=normalize,
            random_state=random_state,
        )
        result["pca"] = X_pca
        result["pca_model"] = pca_model

        X_embed = run_phate(
            X_pca,
            n_components=final_components,
            normalize="none",
            random_state=random_state,
            **kwargs,
        )
        result["embedding"] = X_embed

    elif method == "umap":
        X_norm = normalize_features(features, method=normalize)
        X_embed = run_umap(
            X_norm,
            n_components=final_components,
            normalize="none",
            random_state=random_state,
            **kwargs,
        )
        result["embedding"] = X_embed

    elif method == "phate":
        X_norm = normalize_features(features, method=normalize)
        X_embed = run_phate(
            X_norm,
            n_components=final_components,
            normalize="none",
            random_state=random_state,
            **kwargs,
        )
        result["embedding"] = X_embed

    elif method == "pca_tsne":
        X_pca, pca_model = run_pca(
            features,
            n_components=pca_components,
            normalize=normalize,
            random_state=random_state,
        )
        result["pca"] = X_pca
        result["pca_model"] = pca_model

        X_embed = run_tsne(
            X_pca,
            n_components=final_components,
            normalize="none",
            random_state=random_state,
            **kwargs,
        )
        result["embedding"] = X_embed

    elif method == "tsne":
        X_norm = normalize_features(features, method=normalize)
        X_embed = run_tsne(
            X_norm,
            n_components=final_components,
            normalize="none",
            random_state=random_state,
            **kwargs,
        )
        result["embedding"] = X_embed

    else:
        raise ValueError(f"Unknown embedding method: {method}")

    return result
