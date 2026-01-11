"""
Batch effect assessment for embedding spaces.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def batch_mixing_score(
    embeddings: np.ndarray,
    batch_labels: np.ndarray,
    sample_size: Optional[int] = 10000,
    seed: int = 42,
) -> float:
    """
    Compute batch mixing score based on silhouette score.

    Lower silhouette score by batch = better mixing.
    We return NEGATIVE silhouette so higher = better mixing.

    Parameters
    ----------
    embeddings : np.ndarray
        Cell embeddings, shape (n_cells, n_features)
    batch_labels : np.ndarray
        Batch label for each cell
    sample_size : int, optional
        Subsample for efficiency. None = use all.
    seed : int
        Random seed for subsampling

    Returns
    -------
    float
        Negative silhouette score (higher = better batch mixing)
    """
    if sample_size is not None and len(embeddings) > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[idx]
        batch_labels = batch_labels[idx]

    # Need at least 2 batches
    unique_batches = np.unique(batch_labels)
    if len(unique_batches) < 2:
        return np.nan

    sil = silhouette_score(embeddings, batch_labels)

    # Return negative so higher = better mixing
    return -sil


def within_between_variance_ratio(
    embeddings: np.ndarray,
    batch_labels: np.ndarray,
) -> float:
    """
    Ratio of within-batch to between-batch variance.

    Lower ratio = less batch effect (batch explains less variance).

    Parameters
    ----------
    embeddings : np.ndarray
        Cell embeddings, shape (n_cells, n_features)
    batch_labels : np.ndarray
        Batch label for each cell

    Returns
    -------
    float
        Within/between variance ratio (lower = less batch effect)
    """
    unique_batches = np.unique(batch_labels)

    if len(unique_batches) < 2:
        return np.nan

    # Compute within-batch variance
    within_vars = []
    batch_means = []
    batch_sizes = []

    for batch in unique_batches:
        mask = batch_labels == batch
        batch_data = embeddings[mask]

        if len(batch_data) > 1:
            within_vars.append(np.var(batch_data, axis=0).mean())
            batch_means.append(batch_data.mean(axis=0))
            batch_sizes.append(len(batch_data))

    if len(within_vars) < 2:
        return np.nan

    # Weighted average of within-batch variance
    within_var = np.average(within_vars, weights=batch_sizes)

    # Between-batch variance
    batch_means = np.array(batch_means)
    global_mean = embeddings.mean(axis=0)

    between_var = (
        np.average([np.sum((bm - global_mean) ** 2) for bm in batch_means], weights=batch_sizes)
        / embeddings.shape[1]
    )

    return within_var / (between_var + 1e-10)


def perturbation_retrieval_across_batches(
    embeddings: np.ndarray,
    perturbation_labels: np.ndarray,
    batch_labels: np.ndarray,
    k: int = 10,
    sample_size: int = 1000,
    seed: int = 42,
) -> float:
    """
    Can we retrieve same perturbation across batches using k-NN?

    For each cell, check if its k nearest neighbors from OTHER batches
    have the same perturbation label.

    Parameters
    ----------
    embeddings : np.ndarray
        Cell embeddings
    perturbation_labels : np.ndarray
        Perturbation label for each cell
    batch_labels : np.ndarray
        Batch label for each cell
    k : int
        Number of neighbors to consider
    sample_size : int
        Number of cells to sample for efficiency
    seed : int
        Random seed

    Returns
    -------
    float
        Fraction of correct retrievals (higher = better)
    """
    from sklearn.neighbors import NearestNeighbors

    unique_batches = np.unique(batch_labels)

    if len(unique_batches) < 2:
        return np.nan

    rng = np.random.default_rng(seed)

    # Build index on all cells
    nn = NearestNeighbors(n_neighbors=k + 50, metric="euclidean")
    nn.fit(embeddings)

    correct_retrievals = []

    # Sample cells for efficiency
    sample_idx = rng.choice(len(embeddings), min(sample_size, len(embeddings)), replace=False)

    for idx in sample_idx:
        cell_batch = batch_labels[idx]
        cell_pert = perturbation_labels[idx]

        # Find neighbors
        distances, indices = nn.kneighbors(embeddings[idx : idx + 1])
        indices = indices[0]

        # Filter to other batches
        other_batch_mask = batch_labels[indices] != cell_batch
        other_batch_neighbors = indices[other_batch_mask][:k]

        if len(other_batch_neighbors) == 0:
            continue

        # Check how many have same perturbation
        same_pert = perturbation_labels[other_batch_neighbors] == cell_pert
        correct_retrievals.append(same_pert.mean())

    return np.mean(correct_retrievals) if correct_retrievals else np.nan


@dataclass
class BatchMetrics:
    """Collection of batch effect metrics."""

    space_name: str
    mixing_score: float
    variance_ratio: float
    retrieval_score: float

    def to_dict(self) -> dict:
        return {
            "space": self.space_name,
            "batch_mixing_score": self.mixing_score,
            "variance_ratio": self.variance_ratio,
            "cross_batch_retrieval": self.retrieval_score,
        }


def compute_batch_metrics(
    embeddings_dict: Dict[str, np.ndarray],
    batch_labels: np.ndarray,
    perturbation_labels: Optional[np.ndarray] = None,
    sample_size: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute batch effect metrics across multiple embedding spaces.

    Parameters
    ----------
    embeddings_dict : dict
        Dictionary mapping space name to embeddings array
    batch_labels : np.ndarray
        Batch label for each cell
    perturbation_labels : np.ndarray, optional
        Perturbation labels (for retrieval metric)
    sample_size : int
        Subsample size for efficiency
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Batch metrics for each space
    """
    results = []

    for space_name, embeddings in embeddings_dict.items():
        print(f"Computing batch metrics for {space_name}...")

        mixing = batch_mixing_score(embeddings, batch_labels, sample_size=sample_size, seed=seed)

        variance = within_between_variance_ratio(embeddings, batch_labels)

        if perturbation_labels is not None:
            retrieval = perturbation_retrieval_across_batches(
                embeddings, perturbation_labels, batch_labels, seed=seed
            )
        else:
            retrieval = np.nan

        metrics = BatchMetrics(
            space_name=space_name,
            mixing_score=mixing,
            variance_ratio=variance,
            retrieval_score=retrieval,
        )

        results.append(metrics.to_dict())

    return pd.DataFrame(results)
