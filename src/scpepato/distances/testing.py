"""
Permutation testing for distance significance.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .metrics import DISTANCE_FUNCTIONS


@dataclass
class DistanceResult:
    """Result of a distance computation with permutation test."""

    metric: str
    observed_distance: float
    p_value: float
    null_mean: float
    null_std: float
    n_permutations: int

    @property
    def significant(self) -> bool:
        """Is the distance significant at p < 0.05?"""
        return self.p_value < 0.05

    @property
    def highly_significant(self) -> bool:
        """Is the distance significant at p < 0.01?"""
        return self.p_value < 0.01


def permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "e_distance",
    n_permutations: int = 1000,
    seed: Optional[int] = 42,
) -> DistanceResult:
    """
    Permutation test for significance of distance between two samples.

    Null hypothesis: X and Y come from the same distribution.

    Parameters
    ----------
    X : np.ndarray
        First sample (e.g., perturbed cells), shape (n_x, n_features)
    Y : np.ndarray
        Second sample (e.g., control cells), shape (n_y, n_features)
    metric : str
        Distance metric to use
    n_permutations : int
        Number of permutations for null distribution
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    DistanceResult
        Contains observed distance, p-value, and null distribution stats
    """
    rng = np.random.default_rng(seed)

    distance_fn = DISTANCE_FUNCTIONS[metric]

    # Observed distance
    observed = distance_fn(X, Y)

    # Pool samples
    pooled = np.vstack([X, Y])
    n_x = len(X)
    n_total = len(pooled)

    # Generate null distribution
    null_distances = np.zeros(n_permutations)

    for i in range(n_permutations):
        perm = rng.permutation(n_total)
        X_perm = pooled[perm[:n_x]]
        Y_perm = pooled[perm[n_x:]]
        null_distances[i] = distance_fn(X_perm, Y_perm)

    # Compute p-value (proportion of null >= observed)
    # Add 1 to numerator and denominator for conservative estimate
    p_value = (np.sum(null_distances >= observed) + 1) / (n_permutations + 1)

    return DistanceResult(
        metric=metric,
        observed_distance=observed,
        p_value=p_value,
        null_mean=np.mean(null_distances),
        null_std=np.std(null_distances),
        n_permutations=n_permutations,
    )


def compute_perturbation_distances(
    embeddings: np.ndarray,
    perturbation_labels: np.ndarray,
    control_label: str = "nontargeting",
    metrics: List[str] = ["e_distance", "mse"],
    n_permutations: int = 1000,
    min_cells: int = 10,
    seed: int = 42,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Compute distances from each perturbation to control.

    Parameters
    ----------
    embeddings : np.ndarray
        Cell embeddings, shape (n_cells, n_features)
    perturbation_labels : np.ndarray
        Perturbation label for each cell
    control_label : str
        Label for control/NT cells
    metrics : list of str
        Distance metrics to compute
    n_permutations : int
        Permutations for significance testing
    min_cells : int
        Minimum cells required per perturbation
    seed : int
        Random seed
    show_progress : bool
        Show progress bar

    Returns
    -------
    pd.DataFrame
        Columns: perturbation, metric, distance, p_value, n_cells, significant
    """
    # Get control cells
    control_mask = perturbation_labels == control_label
    if control_mask.sum() < min_cells:
        raise ValueError(f"Not enough control cells (found {control_mask.sum()})")

    control_embeddings = embeddings[control_mask]

    # Get unique perturbations (excluding control)
    unique_perts = np.unique(perturbation_labels)
    unique_perts = unique_perts[unique_perts != control_label]

    results = []

    iterator = tqdm(unique_perts, desc="Computing distances") if show_progress else unique_perts

    for pert in iterator:
        pert_mask = perturbation_labels == pert
        n_cells = pert_mask.sum()

        if n_cells < min_cells:
            continue

        pert_embeddings = embeddings[pert_mask]

        for metric in metrics:
            result = permutation_test(
                pert_embeddings,
                control_embeddings,
                metric=metric,
                n_permutations=n_permutations,
                seed=seed,
            )

            results.append(
                {
                    "perturbation": pert,
                    "metric": metric,
                    "distance": result.observed_distance,
                    "p_value": result.p_value,
                    "null_mean": result.null_mean,
                    "null_std": result.null_std,
                    "n_cells": n_cells,
                    "significant_05": result.significant,
                    "significant_01": result.highly_significant,
                }
            )

    return pd.DataFrame(results)


def compute_all_distances(
    embeddings_dict: Dict[str, np.ndarray],
    perturbation_labels: np.ndarray,
    control_label: str = "nontargeting",
    metrics: List[str] = ["e_distance", "mse"],
    n_permutations: int = 1000,
    min_cells: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute distances across multiple embedding spaces.

    Parameters
    ----------
    embeddings_dict : dict
        Dictionary mapping space name to embeddings array
        e.g., {'PCA': pca_embeddings, 'VanillaVAE': vae_embeddings}
    perturbation_labels : np.ndarray
        Perturbation label for each cell
    control_label : str
        Label for control/NT cells
    metrics : list of str
        Distance metrics to compute
    n_permutations : int
        Permutations for significance testing
    min_cells : int
        Minimum cells required per perturbation
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Columns: space, perturbation, metric, distance, p_value, n_cells, significant
    """
    all_results = []

    for space_name, embeddings in embeddings_dict.items():
        print(f"\nComputing distances for {space_name}...")

        df = compute_perturbation_distances(
            embeddings=embeddings,
            perturbation_labels=perturbation_labels,
            control_label=control_label,
            metrics=metrics,
            n_permutations=n_permutations,
            min_cells=min_cells,
            seed=seed,
        )

        df["space"] = space_name
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)
