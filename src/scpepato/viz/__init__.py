"""Visualization utilities for scPepato."""

from scpepato.viz.images import (
    clear_image_cache,
    encode_image_base64,
    extract_cell_image,
    get_cell_composite,
    get_image_path,
    load_tile_image,
    make_composite,
    normalize_channel,
)
from scpepato.viz.interactive import EmbeddingVisualizer, launch_visualizer
from scpepato.viz.plots import (
    compare_embeddings,
    highlight_perturbations,
    plot_embedding,
    plot_embedding_by_perturbation,
)

__all__ = [
    # Static plots
    "compare_embeddings",
    "highlight_perturbations",
    "plot_embedding",
    "plot_embedding_by_perturbation",
    # Image utilities
    "clear_image_cache",
    "encode_image_base64",
    "extract_cell_image",
    "get_cell_composite",
    "get_image_path",
    "load_tile_image",
    "make_composite",
    "normalize_channel",
    # Interactive
    "EmbeddingVisualizer",
    "launch_visualizer",
]
