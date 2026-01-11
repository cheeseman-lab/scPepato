"""
Response heterogeneity detection using GMM.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


@dataclass
class ResponderResult:
    """Result of responder detection for one perturbation."""

    perturbation: str
    n_cells: int
    is_bimodal: bool
    responder_fraction: float
    bic_1: float  # BIC for k=1
    bic_2: float  # BIC for k=2
    component_means: Optional[np.ndarray]
    component_weights: Optional[np.ndarray]
    cell_assignments: Optional[np.ndarray]

    @property
    def bic_improvement(self) -> float:
        """How much better is k=2 vs k=1? (negative = k=2 better)"""
        return self.bic_2 - self.bic_1


def detect_bimodality(
    embeddings: np.ndarray,
    bic_threshold: float = -10,
    min_component_weight: float = 0.1,
) -> Tuple[bool, GaussianMixture, GaussianMixture]:
    """
    Detect if embeddings show bimodal distribution using GMM.

    Parameters
    ----------
    embeddings : np.ndarray
        Cell embeddings for one perturbation, shape (n_cells, n_features)
    bic_threshold : float
        BIC(k=2) - BIC(k=1) must be below this to call bimodal
    min_component_weight : float
        Minimum weight for smaller component to count as bimodal

    Returns
    -------
    is_bimodal : bool
        Whether distribution is bimodal
    gmm_1 : GaussianMixture
        Fitted GMM with k=1
    gmm_2 : GaussianMixture
        Fitted GMM with k=2
    """
    if len(embeddings) < 20:
        # Not enough cells
        gmm_1 = GaussianMixture(n_components=1, random_state=42).fit(embeddings)
        return False, gmm_1, gmm_1

    # Fit k=1 and k=2
    gmm_1 = GaussianMixture(n_components=1, random_state=42)
    gmm_1.fit(embeddings)

    gmm_2 = GaussianMixture(n_components=2, random_state=42)
    gmm_2.fit(embeddings)

    # Check BIC improvement
    bic_diff = gmm_2.bic(embeddings) - gmm_1.bic(embeddings)

    # Check component weights
    min_weight = gmm_2.weights_.min()

    is_bimodal = (bic_diff < bic_threshold) and (min_weight >= min_component_weight)

    return is_bimodal, gmm_1, gmm_2


def compute_responder_fraction(
    embeddings: np.ndarray,
    control_embeddings: np.ndarray,
    gmm_2: GaussianMixture,
) -> Tuple[float, np.ndarray]:
    """
    Compute fraction of cells that are "responders" (further from control).

    Parameters
    ----------
    embeddings : np.ndarray
        Perturbed cell embeddings
    control_embeddings : np.ndarray
        Control cell embeddings
    gmm_2 : GaussianMixture
        Fitted 2-component GMM

    Returns
    -------
    responder_fraction : float
        Fraction of cells in "responder" component
    assignments : np.ndarray
        Component assignment for each cell (0 or 1)
    """
    # Predict component assignments
    assignments = gmm_2.predict(embeddings)

    # Determine which component is "responder" (further from control mean)
    control_mean = control_embeddings.mean(axis=0)

    component_means = gmm_2.means_
    distances_to_control = [np.linalg.norm(cm - control_mean) for cm in component_means]

    responder_component = np.argmax(distances_to_control)

    responder_fraction = (assignments == responder_component).mean()

    # Relabel so responder = 1
    if responder_component == 0:
        assignments = 1 - assignments

    return responder_fraction, assignments


def analyze_perturbation_heterogeneity(
    embeddings: np.ndarray,
    perturbation_labels: np.ndarray,
    control_label: str = "nontargeting",
    bic_threshold: float = -10,
    min_component_weight: float = 0.1,
    min_cells: int = 20,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Analyze response heterogeneity for all perturbations.

    Parameters
    ----------
    embeddings : np.ndarray
        Cell embeddings, shape (n_cells, n_features)
    perturbation_labels : np.ndarray
        Perturbation label for each cell
    control_label : str
        Label for control cells
    bic_threshold : float
        Threshold for bimodality detection
    min_component_weight : float
        Minimum component weight for bimodality
    min_cells : int
        Minimum cells per perturbation
    show_progress : bool
        Show progress bar

    Returns
    -------
    pd.DataFrame
        Heterogeneity analysis results per perturbation
    """
    # Get control embeddings
    control_mask = perturbation_labels == control_label
    control_embeddings = embeddings[control_mask]

    # Get unique perturbations
    unique_perts = np.unique(perturbation_labels)
    unique_perts = unique_perts[unique_perts != control_label]

    results = []

    iterator = tqdm(unique_perts, desc="Analyzing heterogeneity") if show_progress else unique_perts

    for pert in iterator:
        pert_mask = perturbation_labels == pert
        pert_embeddings = embeddings[pert_mask]
        n_cells = len(pert_embeddings)

        if n_cells < min_cells:
            continue

        # Detect bimodality
        is_bimodal, gmm_1, gmm_2 = detect_bimodality(
            pert_embeddings,
            bic_threshold=bic_threshold,
            min_component_weight=min_component_weight,
        )

        # Compute responder fraction
        if is_bimodal:
            responder_frac, _ = compute_responder_fraction(
                pert_embeddings, control_embeddings, gmm_2
            )
        else:
            responder_frac = np.nan

        results.append(
            {
                "perturbation": pert,
                "n_cells": n_cells,
                "is_bimodal": is_bimodal,
                "responder_fraction": responder_frac,
                "bic_k1": gmm_1.bic(pert_embeddings),
                "bic_k2": gmm_2.bic(pert_embeddings),
                "bic_improvement": gmm_2.bic(pert_embeddings) - gmm_1.bic(pert_embeddings),
                "component_weights": gmm_2.weights_.tolist() if is_bimodal else None,
            }
        )

    return pd.DataFrame(results)


def compute_responder_fractions(
    embeddings_dict: Dict[str, np.ndarray],
    perturbation_labels: np.ndarray,
    control_label: str = "nontargeting",
    **kwargs,
) -> pd.DataFrame:
    """
    Compute responder fractions across multiple embedding spaces.

    Parameters
    ----------
    embeddings_dict : dict
        Dictionary mapping space name to embeddings
    perturbation_labels : np.ndarray
        Perturbation labels
    control_label : str
        Control label
    **kwargs
        Additional arguments to analyze_perturbation_heterogeneity

    Returns
    -------
    pd.DataFrame
        Results with 'space' column added
    """
    all_results = []

    for space_name, embeddings in embeddings_dict.items():
        print(f"\nAnalyzing heterogeneity for {space_name}...")

        df = analyze_perturbation_heterogeneity(
            embeddings=embeddings,
            perturbation_labels=perturbation_labels,
            control_label=control_label,
            **kwargs,
        )

        df["space"] = space_name
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)
