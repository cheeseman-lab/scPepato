"""VAE models for single-cell morphological feature learning."""

# Re-export from embeddings module for backwards compatibility
from scpepato.embeddings import (
    EmbeddingOutput,
)
from scpepato.embeddings import (
    encode_features as encode_new_data,
)
from scpepato.embeddings import (
    load_embedding as load_embedding_space,
)
from scpepato.embeddings import (
    load_vae_model as load_vae_and_normalizer,
)
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
    # Loading utilities (re-exported from embeddings)
    "EmbeddingOutput",
    "load_vae_and_normalizer",
    "encode_new_data",
    "load_embedding_space",
]
