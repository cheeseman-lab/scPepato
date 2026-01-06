"""Embedding and dimensionality reduction methods for scPepato."""

from scpepato.embedding.methods import (
    compute_embedding_pipeline,
    normalize_features,
    run_pca,
    run_phate,
    run_tsne,
    run_umap,
)

__all__ = [
    "run_pca",
    "run_umap",
    "run_phate",
    "run_tsne",
    "normalize_features",
    "compute_embedding_pipeline",
]
