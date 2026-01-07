"""Variational Autoencoder implementations for CellProfiler features.

This module provides VAE architectures for learning latent representations
of single-cell morphological features.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    """MLP encoder for VAE.

    Maps input features to latent distribution parameters (mu, logvar).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        dropout: float = 0.1,
    ):
        """Initialize encoder.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
            dropout: Dropout probability
        """
        super().__init__()

        # Build hidden layers
        layers = []
        prev_dim = input_dim
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

        # Output layers for mu and logvar
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Tuple of (mu, logvar), each (batch_size, latent_dim)
        """
        h = self.hidden(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """MLP decoder for VAE.

    Maps latent representation back to input space.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        """Initialize decoder.

        Args:
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions (reversed from encoder)
            output_dim: Number of output features
            dropout: Dropout probability
        """
        super().__init__()

        # Build hidden layers
        layers = []
        prev_dim = latent_dim
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

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass.

        Args:
            z: Latent representation (batch_size, latent_dim)

        Returns:
            Reconstructed features (batch_size, output_dim)
        """
        h = self.hidden(z)
        return self.fc_out(h)


class VanillaVAE(nn.Module):
    """Standard Variational Autoencoder.

    Learns a latent representation of CellProfiler features without
    explicit batch or perturbation handling.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 50,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        beta: float = 1.0,
    ):
        """Initialize VAE.

        Args:
            input_dim: Number of input features (e.g., 865 CellProfiler features)
            latent_dim: Dimension of latent space (default 50 to match PCA)
            hidden_dims: Hidden layer dimensions. Default: [512, 256, 128]
            dropout: Dropout probability
            beta: Weight for KL divergence term (beta-VAE)
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder and decoder
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, hidden_dims[::-1], input_dim, dropout)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick.

        Sample from N(mu, var) using N(0, 1) samples.

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Tuple of (mu, logvar)
        """
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation.

        Args:
            z: Latent representation (batch_size, latent_dim)

        Returns:
            Reconstructed features (batch_size, input_dim)
        """
        return self.decoder(z)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Dictionary with 'recon', 'mu', 'logvar', 'z'
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

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
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> dict[str, Tensor]:
        """Compute VAE loss.

        Loss = Reconstruction Loss + beta * KL Divergence

        Args:
            x: Original input
            recon: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance
            reduction: How to reduce loss ('mean', 'sum', 'none')

        Returns:
            Dictionary with 'loss', 'recon_loss', 'kl_loss'
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction=reduction)

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        if reduction == "mean":
            kl_loss = kl_loss.mean()
        elif reduction == "sum":
            kl_loss = kl_loss.sum()

        # Total loss
        loss = recon_loss + self.beta * kl_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def get_latent(self, x: Tensor, use_mean: bool = True) -> Tensor:
        """Get latent representation.

        Args:
            x: Input features
            use_mean: If True, return mu. If False, sample from distribution.

        Returns:
            Latent representation (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        if use_mean:
            return mu
        return self.reparameterize(mu, logvar)

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        """Reconstruct input.

        Args:
            x: Input features

        Returns:
            Reconstructed features
        """
        self.eval()
        output = self.forward(x)
        return output["recon"]
