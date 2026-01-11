"""VAE embedding utilities.

This module provides utilities for loading trained VAE models
and generating embeddings from new data.

VAE types supported:
- `vanilla_vae`: Standard VAE with no batch information
- `batch_aware_vae`: VAE with batch embeddings in decoder
- `conditional_vae`: VAE conditioned on perturbation labels
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from scpepato.data.features import FeatureNormalizer
from scpepato.embeddings.base import save_embedding


def load_vae_model(vae_dir: Union[str, Path], device: str = "cpu") -> tuple:
    """Load trained VAE model, normalizer, and config.

    Parameters
    ----------
    vae_dir : Path
        Directory containing VAE outputs (from train_vae.py)
    device : str
        Device to load model on

    Returns
    -------
    model : nn.Module
        Loaded VAE model in eval mode
    normalizer : FeatureNormalizer
        Feature normalizer
    config : dict
        Model configuration
    """
    vae_dir = Path(vae_dir)

    # Load checkpoint
    checkpoint_path = vae_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})

    # Get architecture from config
    latent_dim = config.get("latent_dim", 50)
    hidden_dims = config.get("hidden_dims", [512, 256, 128])

    # Infer input_dim from decoder output layer
    if "decoder.fc_out.weight" in state_dict:
        input_dim = state_dict["decoder.fc_out.weight"].shape[0]
    else:
        input_dim = state_dict["encoder.hidden.0.weight"].shape[1]

    # Detect model type from state dict keys
    if "decoder.batch_embedding.weight" in state_dict:
        model_type = "batch_aware"
        n_batches = state_dict["decoder.batch_embedding.weight"].shape[0]
    elif "encoder.pert_embedding.weight" in state_dict:
        model_type = "conditional"
        n_perturbations, pert_embed_dim = state_dict["encoder.pert_embedding.weight"].shape
    else:
        model_type = "vanilla"

    # Create model
    if model_type == "vanilla":
        from scpepato.models.vae import VanillaVAE

        model = VanillaVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
    elif model_type == "batch_aware":
        from scpepato.models.batch_vae import BatchAwareVAE

        model = BatchAwareVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            n_batches=n_batches,
        )
    elif model_type == "conditional":
        from scpepato.models.conditional_vae import ConditionalVAE

        model = ConditionalVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            n_perturbations=n_perturbations,
            pert_embed_dim=pert_embed_dim,
        )

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Load normalizer
    normalizer_path = vae_dir / "normalizer.npz"
    normalizer = FeatureNormalizer.load(normalizer_path)

    # Add model_type to config
    config["model_type"] = model_type
    config["input_dim"] = input_dim

    return model, normalizer, config


def encode_features(
    model: torch.nn.Module,
    normalizer: FeatureNormalizer,
    features: np.ndarray,
    device: str = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    """Encode raw features to latent embeddings using trained VAE.

    Parameters
    ----------
    model : nn.Module
        Trained VAE model
    normalizer : FeatureNormalizer
        Feature normalizer from training
    features : np.ndarray
        Raw features (n_samples, n_features)
    device : str
        Device for computation
    batch_size : int
        Batch size for encoding

    Returns
    -------
    np.ndarray
        Latent embeddings (n_samples, latent_dim)
    """
    # Normalize features
    features_normalized = normalizer.transform(features)

    # Encode in batches
    embeddings_list = []
    model.eval()

    with torch.no_grad():
        for i in range(0, len(features_normalized), batch_size):
            batch = features_normalized[i : i + batch_size]
            x = torch.tensor(batch, dtype=torch.float32, device=device)
            mu, _ = model.encode(x)
            embeddings_list.append(mu.cpu().numpy())

    return np.vstack(embeddings_list)


def save_vae_embedding(
    output_dir: Union[str, Path],
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    model_type: str,
    config: dict,
    model_path: Optional[Path] = None,
    normalizer_path: Optional[Path] = None,
) -> None:
    """Save VAE embedding output in standardized format.

    Parameters
    ----------
    output_dir : Path
        Directory to save outputs
    embeddings : np.ndarray
        Latent embeddings
    metadata : pd.DataFrame
        Cell metadata
    model_type : str
        VAE type (vanilla, batch_aware, conditional)
    config : dict
        Model configuration
    model_path : Path, optional
        Path to model checkpoint to copy
    normalizer_path : Path, optional
        Path to normalizer to copy
    """
    import shutil

    output_dir = Path(output_dir)

    # Build method name
    method = f"{model_type}_vae"

    # Use base save function for standard files
    save_embedding(output_dir, embeddings, metadata, method, config)

    # Copy model and normalizer if provided
    if model_path is not None and model_path.exists():
        shutil.copy(model_path, output_dir / "model.pt")

    if normalizer_path is not None and normalizer_path.exists():
        shutil.copy(normalizer_path, output_dir / "normalizer.npz")
