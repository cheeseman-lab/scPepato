"""Embedding methods for single-cell morphological features.

This module provides standardized embedding computation and loading:

Embedding Methods:
- PCA (vanilla): Standard PCA with global normalization
- PCA (batch_corrected): Batch center-scaling + PCA + TVN normalization
- VAE (vanilla, batch_aware, conditional): Variational autoencoder variants

Standardized Output Format:
    outputs/embeddings/{method}/{screen}_{n_cells}/
    ├── embeddings.npy        # (n_cells, n_dims) float32
    ├── metadata.parquet      # Cell metadata
    ├── manifest.json         # Config and file paths
    ├── model.{pkl,pt}        # Fitted model
    └── normalizer.npz        # Feature normalization stats

Example Usage:
    # Load any embedding
    from scpepato.embeddings import load_embedding
    emb = load_embedding("outputs/embeddings/pca/mayon_20000")

    # Compute PCA with batch correction
    from scpepato.embeddings import compute_pca
    embeddings, pca, config = compute_pca(features, metadata, method="batch_corrected")

    # Load and use VAE
    from scpepato.embeddings import load_vae_model, encode_features
    model, normalizer, config = load_vae_model("outputs/embeddings/vanilla_vae/mayon_20000")
    embeddings = encode_features(model, normalizer, new_features)
"""

from scpepato.embeddings.base import (
    EmbeddingOutput,
    load_embedding,
    load_multiple_embeddings,
    save_embedding,
)
from scpepato.embeddings.pca import (
    centerscale_by_batch,
    centerscale_on_controls,
    compute_pca,
    save_pca_embedding,
    tvn_on_controls,
)
from scpepato.embeddings.vae import (
    encode_features,
    load_vae_model,
    save_vae_embedding,
)

__all__ = [
    # Base
    "EmbeddingOutput",
    "load_embedding",
    "load_multiple_embeddings",
    "save_embedding",
    # PCA
    "compute_pca",
    "save_pca_embedding",
    "centerscale_by_batch",
    "centerscale_on_controls",
    "tvn_on_controls",
    # VAE
    "load_vae_model",
    "encode_features",
    "save_vae_embedding",
]
