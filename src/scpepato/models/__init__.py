"""VAE models for single-cell morphological feature learning."""

from scpepato.models.batch_vae import BatchAwareDecoder, BatchAwareVAE
from scpepato.models.conditional_vae import (
    ConditionalDecoder,
    ConditionalEncoder,
    ConditionalVAE,
)
from scpepato.models.training import (
    EarlyStopping,
    TrainingConfig,
    TrainingHistory,
    WandbConfig,
    get_latent_embeddings,
    train_vae,
)
from scpepato.models.vae import Decoder, Encoder, VanillaVAE

__all__ = [
    # Vanilla VAE
    "VanillaVAE",
    "Encoder",
    "Decoder",
    # Batch-Aware VAE
    "BatchAwareVAE",
    "BatchAwareDecoder",
    # Conditional VAE
    "ConditionalVAE",
    "ConditionalEncoder",
    "ConditionalDecoder",
    # Training
    "TrainingConfig",
    "TrainingHistory",
    "WandbConfig",
    "EarlyStopping",
    "train_vae",
    "get_latent_embeddings",
]
