"""Conditional Variational Autoencoder.

This module implements a VAE conditioned on perturbation labels,
allowing the model to learn perturbation-specific effects and
potentially predict effects of unseen perturbations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConditionalEncoder(nn.Module):
    """Encoder conditioned on perturbation label.

    The perturbation embedding is concatenated to the input features,
    allowing the encoder to learn perturbation-aware representations.
    """

    def __init__(
        self,
        input_dim: int,
        n_perturbations: int,
        pert_embed_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        dropout: float = 0.1,
    ):
        """Initialize conditional encoder.

        Args:
            input_dim: Number of input features
            n_perturbations: Number of unique perturbations
            pert_embed_dim: Dimension of perturbation embedding
            hidden_dims: Hidden layer dimensions
            latent_dim: Latent space dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.pert_embedding = nn.Embedding(n_perturbations, pert_embed_dim)

        # Build hidden layers (input is features + perturbation embedding)
        layers = []
        prev_dim = input_dim + pert_embed_dim
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
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: Tensor, pert_idx: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input features (batch_size, input_dim)
            pert_idx: Perturbation indices (batch_size,)

        Returns:
            Tuple of (mu, logvar)
        """
        pert_emb = self.pert_embedding(pert_idx)
        x_pert = torch.cat([x, pert_emb], dim=1)
        h = self.hidden(x_pert)
        return self.fc_mu(h), self.fc_logvar(h)


class ConditionalDecoder(nn.Module):
    """Decoder conditioned on perturbation label."""

    def __init__(
        self,
        latent_dim: int,
        n_perturbations: int,
        pert_embed_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        """Initialize conditional decoder.

        Args:
            latent_dim: Dimension of latent space
            n_perturbations: Number of unique perturbations
            pert_embed_dim: Dimension of perturbation embedding
            hidden_dims: Hidden layer dimensions
            output_dim: Number of output features
            dropout: Dropout probability
        """
        super().__init__()

        self.pert_embedding = nn.Embedding(n_perturbations, pert_embed_dim)

        # Build hidden layers (input is latent + perturbation embedding)
        layers = []
        prev_dim = latent_dim + pert_embed_dim
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

    def forward(self, z: Tensor, pert_idx: Tensor) -> Tensor:
        """Forward pass.

        Args:
            z: Latent representation (batch_size, latent_dim)
            pert_idx: Perturbation indices (batch_size,)

        Returns:
            Reconstructed features (batch_size, output_dim)
        """
        pert_emb = self.pert_embedding(pert_idx)
        z_pert = torch.cat([z, pert_emb], dim=1)
        h = self.hidden(z_pert)
        return self.fc_out(h)


class ConditionalVAE(nn.Module):
    """Conditional Variational Autoencoder.

    This VAE is conditioned on perturbation labels, learning a latent
    space where the perturbation effect is explicitly modeled. This
    enables:
    - Separation of cell state from perturbation effect
    - Prediction of perturbation effects on new cells
    - Generation of cells with specific perturbations

    Architecture:
        Encoder: features + pert_embedding → latent
        Decoder: latent + pert_embedding → features
    """

    def __init__(
        self,
        input_dim: int,
        n_perturbations: int,
        latent_dim: int = 50,
        hidden_dims: list[int] | None = None,
        pert_embed_dim: int = 32,
        dropout: float = 0.1,
        beta: float = 1.0,
    ):
        """Initialize Conditional VAE.

        Args:
            input_dim: Number of input features
            n_perturbations: Number of unique perturbations
            latent_dim: Dimension of latent space
            hidden_dims: Hidden layer dimensions. Default: [512, 256, 128]
            pert_embed_dim: Dimension of perturbation embedding
            dropout: Dropout probability
            beta: Weight for KL divergence term
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.input_dim = input_dim
        self.n_perturbations = n_perturbations
        self.latent_dim = latent_dim
        self.pert_embed_dim = pert_embed_dim
        self.beta = beta

        # Conditional encoder
        self.encoder = ConditionalEncoder(
            input_dim=input_dim,
            n_perturbations=n_perturbations,
            pert_embed_dim=pert_embed_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        # Conditional decoder
        self.decoder = ConditionalDecoder(
            latent_dim=latent_dim,
            n_perturbations=n_perturbations,
            pert_embed_dim=pert_embed_dim,
            hidden_dims=hidden_dims[::-1],
            output_dim=input_dim,
            dropout=dropout,
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: Tensor, pert_idx: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input with perturbation conditioning."""
        return self.encoder(x, pert_idx)

    def decode(self, z: Tensor, pert_idx: Tensor) -> Tensor:
        """Decode latent with perturbation conditioning."""
        return self.decoder(z, pert_idx)

    def forward(self, x: Tensor, pert_idx: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Input features (batch_size, input_dim)
            pert_idx: Perturbation indices (batch_size,)

        Returns:
            Dictionary with 'recon', 'mu', 'logvar', 'z'
        """
        mu, logvar = self.encode(x, pert_idx)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, pert_idx)

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
        recon_loss = F.mse_loss(recon, x, reduction=reduction)

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

    def get_latent(self, x: Tensor, pert_idx: Tensor, use_mean: bool = True) -> Tensor:
        """Get latent representation."""
        mu, logvar = self.encode(x, pert_idx)
        if use_mean:
            return mu
        return self.reparameterize(mu, logvar)

    @torch.no_grad()
    def reconstruct(self, x: Tensor, pert_idx: Tensor) -> Tensor:
        """Reconstruct input."""
        self.eval()
        output = self.forward(x, pert_idx)
        return output["recon"]

    @torch.no_grad()
    def transfer_perturbation(
        self,
        x: Tensor,
        source_pert_idx: Tensor,
        target_pert_idx: Tensor,
    ) -> Tensor:
        """Transfer perturbation effect from source to target.

        Encodes cells with source perturbation, decodes with target perturbation.
        This predicts what cells would look like with a different perturbation.

        Args:
            x: Input features
            source_pert_idx: Original perturbation indices
            target_pert_idx: Target perturbation indices

        Returns:
            Predicted features with target perturbation
        """
        self.eval()
        mu, _ = self.encode(x, source_pert_idx)
        return self.decode(mu, target_pert_idx)
