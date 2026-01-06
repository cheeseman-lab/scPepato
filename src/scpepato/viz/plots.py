"""Plotting functions for embedding visualization.

This module provides functions for visualizing single-cell embeddings,
with support for highlighting specific perturbations and comparing
different embedding methods.
"""

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_embedding(
    embedding: np.ndarray,
    ax: plt.Axes | None = None,
    color: np.ndarray | pd.Series | None = None,
    cmap: str = "viridis",
    alpha: float = 0.5,
    s: float = 1,
    title: str | None = None,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    colorbar: bool = True,
    colorbar_label: str | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot a 2D embedding.

    Args:
        embedding: 2D embedding array (n_cells x 2)
        ax: Matplotlib axes. If None, creates new figure.
        color: Color values for each point (numeric or categorical)
        cmap: Colormap for numeric color values
        alpha: Point transparency
        s: Point size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        colorbar: Whether to show colorbar for numeric colors
        colorbar_label: Label for colorbar
        **kwargs: Additional arguments passed to scatter

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=color,
        cmap=cmap,
        alpha=alpha,
        s=s,
        **kwargs,
    )

    if color is not None and colorbar and np.issubdtype(type(color[0]), np.number):
        cbar = plt.colorbar(scatter, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")

    return ax


def plot_embedding_by_perturbation(
    embedding: np.ndarray,
    perturbations: pd.Series | np.ndarray,
    highlight_genes: Sequence[str] | None = None,
    background_color: str = "lightgray",
    background_alpha: float = 0.2,
    highlight_alpha: float = 0.7,
    s: float = 1,
    highlight_s: float = 5,
    ax: plt.Axes | None = None,
    title: str | None = None,
    legend: bool = True,
    legend_loc: str = "best",
    cmap: str | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot embedding with specific perturbations highlighted.

    Args:
        embedding: 2D embedding array (n_cells x 2)
        perturbations: Perturbation labels for each cell (e.g., gene_symbol_0)
        highlight_genes: List of genes to highlight. If None, highlights all unique.
        background_color: Color for non-highlighted cells
        background_alpha: Alpha for non-highlighted cells
        highlight_alpha: Alpha for highlighted cells
        s: Point size for background
        highlight_s: Point size for highlighted points
        ax: Matplotlib axes
        title: Plot title
        legend: Whether to show legend
        legend_loc: Legend location
        cmap: Colormap name for highlighted genes (uses tab10 if None)
        **kwargs: Additional arguments passed to scatter

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    perturbations = np.asarray(perturbations)

    # Determine which genes to highlight
    if highlight_genes is None:
        unique_genes = np.unique(perturbations)
        highlight_genes = [g for g in unique_genes if g != "nontargeting"]

    # Create mask for highlighted cells
    highlight_mask = np.isin(perturbations, highlight_genes)

    # Plot background (non-highlighted cells)
    ax.scatter(
        embedding[~highlight_mask, 0],
        embedding[~highlight_mask, 1],
        c=background_color,
        alpha=background_alpha,
        s=s,
        label="other",
        **kwargs,
    )

    # Get colors for highlighted genes
    if cmap is None:
        colors = plt.cm.tab10.colors
    else:
        colors = plt.get_cmap(cmap).colors

    # Plot each highlighted gene
    for i, gene in enumerate(highlight_genes):
        gene_mask = perturbations == gene
        if gene_mask.sum() > 0:
            color = colors[i % len(colors)]
            ax.scatter(
                embedding[gene_mask, 0],
                embedding[gene_mask, 1],
                c=[color],
                alpha=highlight_alpha,
                s=highlight_s,
                label=f"{gene} (n={gene_mask.sum()})",
                **kwargs,
            )

    if title:
        ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_aspect("equal")

    if legend and len(highlight_genes) <= 20:
        ax.legend(loc=legend_loc, fontsize=8)

    return ax


def highlight_perturbations(
    embedding: np.ndarray,
    perturbations: pd.Series | np.ndarray,
    genes_of_interest: Sequence[str],
    nontargeting_label: str = "nontargeting",
    figsize: tuple[int, int] = (15, 5),
    background_alpha: float = 0.1,
    highlight_alpha: float = 0.6,
    s: float = 1,
    highlight_s: float = 3,
) -> plt.Figure:
    """Create a figure comparing genes of interest vs nontargeting.

    Creates a 3-panel figure:
    1. All cells colored by gene
    2. Genes of interest highlighted vs background
    3. Nontargeting cells highlighted vs background

    Args:
        embedding: 2D embedding array (n_cells x 2)
        perturbations: Perturbation labels for each cell
        genes_of_interest: List of genes to highlight
        nontargeting_label: Label for nontargeting controls
        figsize: Figure size
        background_alpha: Alpha for background cells
        highlight_alpha: Alpha for highlighted cells
        s: Point size for background
        highlight_s: Point size for highlighted

    Returns:
        Matplotlib figure
    """
    perturbations = np.asarray(perturbations)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: All highlighted genes
    plot_embedding_by_perturbation(
        embedding,
        perturbations,
        highlight_genes=genes_of_interest,
        ax=axes[0],
        title="Genes of Interest",
        background_alpha=background_alpha,
        highlight_alpha=highlight_alpha,
        s=s,
        highlight_s=highlight_s,
    )

    # Panel 2: Genes of interest vs all
    goi_mask = np.isin(perturbations, genes_of_interest)
    axes[1].scatter(
        embedding[~goi_mask, 0],
        embedding[~goi_mask, 1],
        c="lightgray",
        alpha=background_alpha,
        s=s,
        label="other",
    )
    axes[1].scatter(
        embedding[goi_mask, 0],
        embedding[goi_mask, 1],
        c="red",
        alpha=highlight_alpha,
        s=highlight_s,
        label=f"GOI (n={goi_mask.sum()})",
    )
    axes[1].set_title("Genes of Interest vs Other")
    axes[1].set_xlabel("Dim 1")
    axes[1].set_ylabel("Dim 2")
    axes[1].set_aspect("equal")
    axes[1].legend()

    # Panel 3: Nontargeting vs all
    nt_mask = perturbations == nontargeting_label
    axes[2].scatter(
        embedding[~nt_mask, 0],
        embedding[~nt_mask, 1],
        c="lightgray",
        alpha=background_alpha,
        s=s,
        label="other",
    )
    axes[2].scatter(
        embedding[nt_mask, 0],
        embedding[nt_mask, 1],
        c="blue",
        alpha=highlight_alpha,
        s=highlight_s,
        label=f"nontargeting (n={nt_mask.sum()})",
    )
    axes[2].set_title("Nontargeting vs Other")
    axes[2].set_xlabel("Dim 1")
    axes[2].set_ylabel("Dim 2")
    axes[2].set_aspect("equal")
    axes[2].legend()

    plt.tight_layout()
    return fig


def compare_embeddings(
    embeddings: dict[str, np.ndarray],
    perturbations: pd.Series | np.ndarray | None = None,
    highlight_genes: Sequence[str] | None = None,
    figsize_per_plot: tuple[int, int] = (5, 5),
    **kwargs,
) -> plt.Figure:
    """Compare multiple embedding methods side by side.

    Args:
        embeddings: Dictionary mapping method name to embedding array
        perturbations: Optional perturbation labels for coloring
        highlight_genes: Genes to highlight if perturbations provided
        figsize_per_plot: Size of each subplot
        **kwargs: Additional arguments passed to plotting functions

    Returns:
        Matplotlib figure
    """
    n_embeddings = len(embeddings)
    fig, axes = plt.subplots(
        1,
        n_embeddings,
        figsize=(figsize_per_plot[0] * n_embeddings, figsize_per_plot[1]),
    )

    if n_embeddings == 1:
        axes = [axes]

    for ax, (name, emb) in zip(axes, embeddings.items()):
        if perturbations is not None and highlight_genes is not None:
            plot_embedding_by_perturbation(
                emb,
                perturbations,
                highlight_genes=highlight_genes,
                ax=ax,
                title=name,
                legend=False,
                **kwargs,
            )
        else:
            plot_embedding(
                emb,
                ax=ax,
                title=name,
                **kwargs,
            )

    plt.tight_layout()
    return fig
