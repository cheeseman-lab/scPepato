"""Batch-Aware Variational Autoencoder.

This module implements a VAE with batch embeddings in the decoder,
similar to scVI, to learn batch-invariant latent representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from scpepato.models.vae import Encoder


class BatchAwareDecoder(nn.Module):
    """Decoder with batch embedding concatenated to latent.

    The batch embedding allows the decoder to account for batch-specific
    effects, encouraging the latent space to be batch-invariant.
    """

    def __init__(
        self,
        latent_dim: int,
        n_batches: int,
        batch_embed_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        """Initialize decoder.

        Args:
            latent_dim: Dimension of latent space
            n_batches: Number of unique batches
            batch_embed_dim: Dimension of batch embedding
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output features
            dropout: Dropout probability
        """
        super().__init__()

        self.batch_embedding = nn.Embedding(n_batches, batch_embed_dim)

        # Build hidden layers (input is latent + batch embedding)
        layers = []
        prev_dim = latent_dim + batch_embed_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_dim, output_dim)

    def forward(self, z: Tensor, batch_idx: Tensor) -> Tensor:
        """Forward pass.

        Args:
            z: Latent representation (batch_size, latent_dim)
            batch_idx: Batch indices (batch_size,)

        Returns:
            Reconstructed features (batch_size, output_dim)
        """
        batch_emb = self.batch_embedding(batch_idx)
        z_batch = torch.cat([z, batch_emb], dim=1)
        h = self.hidden(z_batch)
        return self.fc_out(h)


class BatchAwareVAE(nn.Module):
    """Batch-Aware Variational Autoencoder.

    Similar to scVI, this model learns batch-invariant latent representations
    by conditioning the decoder on batch identity. The encoder sees only
    features, so the latent space captures biological variation independent
    of batch effects.

    Architecture:
        Encoder: features → latent (batch-agnostic)
        Decoder: latent + batch_embedding → features (batch-specific reconstruction)
    """

    def __init__(
        self,
        input_dim: int,
        n_batches: int,
        latent_dim: int = 50,
        hidden_dims: list[int] | None = None,
        batch_embed_dim: int = 16,
        dropout: float = 0.1,
        beta: float = 1.0,
    ):
        """Initialize Batch-Aware VAE.

        Args:
            input_dim: Number of input features
            n_batches: Number of unique batches (e.g., wells)
            latent_dim: Dimension of latent space
            hidden_dims: Hidden layer dimensions. Default: [512, 256, 128]
            batch_embed_dim: Dimension of batch embedding
            dropout: Dropout probability
            beta: Weight for KL divergence term
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.input_dim = input_dim
        self.n_batches = n_batches
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder (batch-agnostic)
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim, dropout)

        # Decoder (batch-aware)
        self.decoder = BatchAwareDecoder(
            latent_dim=latent_dim,
            n_batches=n_batches,
            batch_embed_dim=batch_embed_dim,
            hidden_dims=hidden_dims[::-1],
            output_dim=input_dim,
            dropout=dropout,
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input to latent distribution parameters."""
        return self.encoder(x)

    def decode(self, z: Tensor, batch_idx: Tensor) -> Tensor:
        """Decode latent representation with batch conditioning."""
        return self.decoder(z, batch_idx)

    def forward(self, x: Tensor, batch_idx: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Input features (batch_size, input_dim)
            batch_idx: Batch indices (batch_size,)

        Returns:
            Dictionary with 'recon', 'mu', 'logvar', 'z'
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, batch_idx)

        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def loss_function(
        self,
        x: Tensor,
        recon: Tensor,
        mu: Tensor,
        logvar: Tensor,
        reduction: str = "mean",
    ) -> dict[str, Tensor]:
        """Compute VAE loss."""
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction=reduction)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        if reduction == "mean":
            kl_loss = kl_loss.mean()
        elif reduction == "sum":
            kl_loss = kl_loss.sum()

        loss = recon_loss + self.beta * kl_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def get_latent(self, x: Tensor, use_mean: bool = True) -> Tensor:
        """Get latent representation (batch-agnostic)."""
        mu, logvar = self.encode(x)
        if use_mean:
            return mu
        return self.reparameterize(mu, logvar)

    @torch.no_grad()
    def reconstruct(self, x: Tensor, batch_idx: Tensor) -> Tensor:
        """Reconstruct input."""
        self.eval()
        output = self.forward(x, batch_idx)
        return output["recon"]
