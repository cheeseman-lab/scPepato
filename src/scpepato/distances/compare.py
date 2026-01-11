"""
Compare distance metrics across embedding spaces.
"""

from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass
class DistanceComparison:
    """Results of comparing distances across spaces."""

    results_df: pd.DataFrame
    spaces: List[str]
    metrics: List[str]

    def summary_by_space(self) -> pd.DataFrame:
        """Summarize results by embedding space."""
        return (
            self.results_df.groupby(["space", "metric"])
            .agg(
                {
                    "distance": ["mean", "std", "median"],
                    "p_value": ["mean", "median"],
                    "significant_05": "mean",  # Fraction significant
                    "significant_01": "mean",
                }
            )
            .round(4)
        )

    def best_space_per_metric(self) -> pd.DataFrame:
        """Find which space has best separation per metric."""
        summary = (
            self.results_df.groupby(["space", "metric"])
            .agg(
                {
                    "significant_01": "mean",
                    "distance": "median",
                }
            )
            .reset_index()
        )

        best = summary.loc[summary.groupby("metric")["significant_01"].idxmax()]

        return best[["metric", "space", "significant_01", "distance"]]

    def plot_distance_distributions(
        self, metric: str = "e_distance", figsize: tuple = (12, 4)
    ) -> plt.Figure:
        """Plot distance distributions across spaces."""
        subset = self.results_df[self.results_df["metric"] == metric]

        fig, axes = plt.subplots(1, len(self.spaces), figsize=figsize, sharey=True)

        if len(self.spaces) == 1:
            axes = [axes]

        for ax, space in zip(axes, self.spaces):
            space_data = subset[subset["space"] == space]

            ax.hist(space_data["distance"], bins=30, alpha=0.7, edgecolor="black")
            ax.axvline(
                space_data["distance"].median(),
                color="red",
                linestyle="--",
                label=f"Median: {space_data['distance'].median():.3f}",
            )
            ax.set_title(f"{space}\n({space_data['significant_01'].mean() * 100:.1f}% sig)")
            ax.set_xlabel(f"{metric}")
            ax.legend()

        axes[0].set_ylabel("Count")
        fig.suptitle(f"Distance from NT by Embedding Space ({metric})")
        plt.tight_layout()

        return fig

    def plot_significance_comparison(self, figsize: tuple = (10, 6)) -> plt.Figure:
        """Plot fraction of significant perturbations per space."""
        summary = (
            self.results_df.groupby(["space", "metric"])
            .agg({"significant_01": "mean"})
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=figsize)

        sns.barplot(data=summary, x="space", y="significant_01", hue="metric", ax=ax)

        ax.set_ylabel("Fraction Significant (p < 0.01)")
        ax.set_xlabel("Embedding Space")
        ax.set_title("Perturbation Separation by Embedding Space")
        ax.legend(title="Metric")

        # Add percentage labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", label_type="edge")

        plt.tight_layout()
        return fig


def compare_spaces(
    embeddings_dict: Dict[str, np.ndarray],
    perturbation_labels: np.ndarray,
    control_label: str = "nontargeting",
    metrics: List[str] = ["e_distance", "mse", "mmd_linear"],
    n_permutations: int = 1000,
    min_cells: int = 10,
    seed: int = 42,
) -> DistanceComparison:
    """
    Compare perturbation distances across multiple embedding spaces.

    Parameters
    ----------
    embeddings_dict : dict
        Dictionary mapping space name to embeddings array
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
    DistanceComparison
        Object with results and analysis methods
    """
    from .testing import compute_all_distances

    results_df = compute_all_distances(
        embeddings_dict=embeddings_dict,
        perturbation_labels=perturbation_labels,
        control_label=control_label,
        metrics=metrics,
        n_permutations=n_permutations,
        min_cells=min_cells,
        seed=seed,
    )

    return DistanceComparison(
        results_df=results_df,
        spaces=list(embeddings_dict.keys()),
        metrics=metrics,
    )
