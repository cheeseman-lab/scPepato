"""Training utilities for VAE models.

This module provides training loops, early stopping, checkpointing,
and Weights & Biases integration for VAE training on CellProfiler features.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""

    enabled: bool = True
    project: str = "scpepato-vae"
    entity: str | None = None  # Your wandb team/username
    name: str | None = None  # Run name (auto-generated if None)
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    log_freq: int = 1  # Log every N epochs
    log_reconstructions: bool = True  # Log reconstruction samples
    log_latent_space: bool = True  # Log latent space visualizations
    n_reconstruction_samples: int = 16  # Number of samples to visualize
    log_gradients: bool = False  # Log gradient histograms (slower)
    save_model_artifact: bool = True  # Save model as wandb artifact


@dataclass
class TrainingConfig:
    """Configuration for VAE training."""

    # Training parameters
    batch_size: int = 512
    learning_rate: float = 1e-3
    epochs: int = 100
    early_stopping_patience: int = 10

    # Model parameters
    latent_dim: int = 50
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.1
    beta: float = 1.0

    # Scheduler
    use_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # Logging
    log_every: int = 10  # Log every N batches
    checkpoint_dir: str = "checkpoints"

    # Weights & Biases
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingHistory:
    """Training history tracker."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_recon_loss: list[float] = field(default_factory=list)
    val_recon_loss: list[float] = field(default_factory=list)
    train_kl_loss: list[float] = field(default_factory=list)
    val_kl_loss: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[float]]:
        """Convert to dictionary."""
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_recon_loss": self.train_recon_loss,
            "val_recon_loss": self.val_recon_loss,
            "train_kl_loss": self.train_kl_loss,
            "val_kl_loss": self.val_kl_loss,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
        }

    def save(self, path: str | Path) -> None:
        """Save history to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TrainingHistory":
        """Load history from JSON file."""
        with open(path) as f:
            data = json.load(f)
        history = cls()
        for key, value in data.items():
            setattr(history, key, value)
        return history


class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# =============================================================================
# Weights & Biases Utilities
# =============================================================================


def init_wandb(
    config: TrainingConfig,
    model: nn.Module,
    model_type: str = "VanillaVAE",
    n_features: int | None = None,
    n_train_samples: int | None = None,
    n_val_samples: int | None = None,
    extra_config: dict[str, Any] | None = None,
) -> bool:
    """Initialize Weights & Biases run.

    Args:
        config: Training configuration
        model: Model being trained
        model_type: Type of VAE model
        n_features: Number of input features
        n_train_samples: Number of training samples
        n_val_samples: Number of validation samples
        extra_config: Additional config to log

    Returns:
        True if wandb was initialized, False otherwise
    """
    if not WANDB_AVAILABLE or not config.wandb.enabled:
        return False

    # Build config dict for wandb
    wandb_config = {
        "model_type": model_type,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "latent_dim": config.latent_dim,
        "hidden_dims": config.hidden_dims,
        "dropout": config.dropout,
        "beta": config.beta,
        "use_scheduler": config.use_scheduler,
        "scheduler_patience": config.scheduler_patience,
        "scheduler_factor": config.scheduler_factor,
        "device": config.device,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "n_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    if n_features is not None:
        wandb_config["n_features"] = n_features
    if n_train_samples is not None:
        wandb_config["n_train_samples"] = n_train_samples
    if n_val_samples is not None:
        wandb_config["n_val_samples"] = n_val_samples
    if extra_config:
        wandb_config.update(extra_config)

    # Initialize wandb
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name,
        tags=config.wandb.tags,
        notes=config.wandb.notes,
        config=wandb_config,
    )

    # Watch model for gradients
    if config.wandb.log_gradients:
        wandb.watch(model, log="all", log_freq=100)

    return True


def log_wandb_metrics(
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    learning_rate: float,
    epoch_time: float,
) -> None:
    """Log metrics to wandb.

    Args:
        epoch: Current epoch number
        train_metrics: Training metrics dict
        val_metrics: Validation metrics dict
        learning_rate: Current learning rate
        epoch_time: Time taken for epoch
    """
    if not WANDB_AVAILABLE:
        return

    wandb.log(
        {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/recon_loss": train_metrics["recon_loss"],
            "train/kl_loss": train_metrics["kl_loss"],
            "val/loss": val_metrics["loss"],
            "val/recon_loss": val_metrics["recon_loss"],
            "val/kl_loss": val_metrics["kl_loss"],
            "learning_rate": learning_rate,
            "epoch_time": epoch_time,
        }
    )


@torch.no_grad()
def log_wandb_reconstructions(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    n_samples: int = 16,
    feature_names: list[str] | None = None,
) -> None:
    """Log reconstruction visualizations to wandb.

    Args:
        model: VAE model
        val_loader: Validation data loader
        device: Device to use
        n_samples: Number of samples to visualize
        feature_names: Names of features (optional)
    """
    if not WANDB_AVAILABLE:
        return

    model.eval()
    model_type = _get_model_type(model)

    # Get a batch of samples
    batch = next(iter(val_loader))
    # Subset batch for n_samples
    subset_batch = {k: v[:n_samples] for k, v in batch.items()}
    x = subset_batch["features"].to(device)
    output = _forward_model(model, subset_batch, device, model_type)
    x_recon = output["recon"]

    # Create comparison table
    x_np = x.cpu().numpy()
    x_recon_np = x_recon.cpu().numpy()

    # Log reconstruction error distribution
    recon_errors = np.mean((x_np - x_recon_np) ** 2, axis=1)
    wandb.log(
        {
            "reconstruction/mean_error": float(np.mean(recon_errors)),
            "reconstruction/std_error": float(np.std(recon_errors)),
            "reconstruction/max_error": float(np.max(recon_errors)),
        }
    )

    # Create a simple visualization of feature-wise reconstruction
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Original vs Reconstructed for first sample
    ax = axes[0, 0]
    n_features_to_show = min(50, x_np.shape[1])
    ax.scatter(x_np[0, :n_features_to_show], x_recon_np[0, :n_features_to_show], alpha=0.5, s=20)
    ax.plot(
        [x_np[0, :n_features_to_show].min(), x_np[0, :n_features_to_show].max()],
        [x_np[0, :n_features_to_show].min(), x_np[0, :n_features_to_show].max()],
        "r--",
        lw=2,
    )
    ax.set_xlabel("Original")
    ax.set_ylabel("Reconstructed")
    ax.set_title("Sample 0: Original vs Reconstructed (first 50 features)")

    # Plot 2: Per-feature reconstruction error
    ax = axes[0, 1]
    feature_errors = np.mean((x_np - x_recon_np) ** 2, axis=0)
    ax.hist(feature_errors, bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Mean Squared Error")
    ax.set_ylabel("Number of Features")
    ax.set_title("Per-Feature Reconstruction Error Distribution")

    # Plot 3: Per-sample reconstruction error
    ax = axes[1, 0]
    ax.bar(range(len(recon_errors)), recon_errors, color="steelblue", alpha=0.7)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Per-Sample Reconstruction Error")

    # Plot 4: Feature values comparison (boxplot-style)
    ax = axes[1, 1]
    sample_indices = [0, min(5, n_samples - 1), min(10, n_samples - 1), min(15, n_samples - 1)]
    sample_indices = [i for i in sample_indices if i < n_samples]
    for i, idx in enumerate(sample_indices):
        ax.scatter(
            [i - 0.15] * n_features_to_show,
            x_np[idx, :n_features_to_show],
            alpha=0.3,
            s=10,
            label=f"Original {idx}" if i == 0 else None,
            c="blue",
        )
        ax.scatter(
            [i + 0.15] * n_features_to_show,
            x_recon_np[idx, :n_features_to_show],
            alpha=0.3,
            s=10,
            label=f"Recon {idx}" if i == 0 else None,
            c="orange",
        )
    ax.set_xticks(range(len(sample_indices)))
    ax.set_xticklabels([f"Sample {i}" for i in sample_indices])
    ax.set_ylabel("Feature Value")
    ax.set_title("Feature Values: Original vs Reconstructed")
    ax.legend(loc="upper right")

    plt.tight_layout()
    wandb.log({"reconstruction/comparison": wandb.Image(fig)})
    plt.close(fig)


@torch.no_grad()
def log_wandb_latent_space(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    max_samples: int = 5000,
    perturbation_labels: np.ndarray | None = None,
    batch_labels: np.ndarray | None = None,
    label_names: dict[int, str] | None = None,
) -> None:
    """Log latent space visualization to wandb using UMAP.

    Args:
        model: VAE model
        val_loader: Validation data loader
        device: Device to use
        max_samples: Maximum samples to visualize
        perturbation_labels: Optional perturbation labels for coloring
        batch_labels: Optional batch labels for coloring
        label_names: Optional mapping from label indices to names
    """
    if not WANDB_AVAILABLE:
        return

    model.eval()
    model_type = _get_model_type(model)

    # Collect latent representations
    latents = []
    pert_labels_batch = []
    batch_labels_batch = []
    n_collected = 0

    for batch in val_loader:
        z = _get_model_latent(model, batch, device, model_type, use_mean=True)
        latents.append(z.cpu().numpy())

        if batch.get("perturbation") is not None:
            pert_labels_batch.append(batch["perturbation"].numpy())
        if batch.get("batch") is not None:
            batch_labels_batch.append(batch["batch"].numpy())

        n_collected += batch["features"].shape[0]
        if n_collected >= max_samples:
            break

    latents = np.concatenate(latents, axis=0)[:max_samples]

    if pert_labels_batch:
        pert_labels = np.concatenate(pert_labels_batch, axis=0)[:max_samples]
    else:
        pert_labels = None

    if batch_labels_batch:
        batch_lbls = np.concatenate(batch_labels_batch, axis=0)[:max_samples]
    else:
        batch_lbls = None

    # Compute UMAP embedding of latent space
    try:
        from umap import UMAP

        umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        latent_2d = umap.fit_transform(latents)
    except ImportError:
        # Fallback to PCA if UMAP not available
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latents)

    import matplotlib.pyplot as plt

    # Create visualization
    n_plots = 1 + (pert_labels is not None) + (batch_lbls is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Plot 1: Basic latent space
    ax = axes[0]
    ax.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.3, s=5)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Latent Space ({len(latents):,} cells)")

    plot_idx = 1

    # Plot 2: Colored by perturbation
    if pert_labels is not None:
        ax = axes[plot_idx]
        unique_labels = np.unique(pert_labels)

        if len(unique_labels) <= 20:
            cmap = plt.cm.get_cmap("tab20", len(unique_labels))
            for i, label in enumerate(unique_labels):
                mask = pert_labels == label
                name = label_names.get(label, f"Pert {label}") if label_names else f"Pert {label}"
                ax.scatter(
                    latent_2d[mask, 0], latent_2d[mask, 1], alpha=0.5, s=5, label=name, c=[cmap(i)]
                )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6, ncol=2)
        else:
            scatter = ax.scatter(
                latent_2d[:, 0], latent_2d[:, 1], c=pert_labels, alpha=0.5, s=5, cmap="viridis"
            )
            plt.colorbar(scatter, ax=ax, label="Perturbation ID")

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Latent Space by Perturbation")
        plot_idx += 1

    # Plot 3: Colored by batch
    if batch_lbls is not None:
        ax = axes[plot_idx]
        unique_batches = np.unique(batch_lbls)

        if len(unique_batches) <= 20:
            cmap = plt.cm.get_cmap("tab20", len(unique_batches))
            for i, batch in enumerate(unique_batches):
                mask = batch_lbls == batch
                ax.scatter(
                    latent_2d[mask, 0],
                    latent_2d[mask, 1],
                    alpha=0.5,
                    s=5,
                    label=f"Batch {batch}",
                    c=[cmap(i)],
                )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6)
        else:
            scatter = ax.scatter(
                latent_2d[:, 0], latent_2d[:, 1], c=batch_lbls, alpha=0.5, s=5, cmap="viridis"
            )
            plt.colorbar(scatter, ax=ax, label="Batch ID")

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Latent Space by Batch")

    plt.tight_layout()
    wandb.log({"latent_space/umap": wandb.Image(fig)})
    plt.close(fig)

    # Log latent space statistics
    wandb.log(
        {
            "latent_space/mean": float(np.mean(latents)),
            "latent_space/std": float(np.std(latents)),
            "latent_space/min": float(np.min(latents)),
            "latent_space/max": float(np.max(latents)),
        }
    )


def save_wandb_artifact(
    model: nn.Module,
    config: TrainingConfig,
    checkpoint_path: Path | str,
    model_type: str = "VanillaVAE",
) -> None:
    """Save model as wandb artifact.

    Args:
        model: Trained model
        config: Training configuration
        checkpoint_path: Path to saved checkpoint
        model_type: Type of model
    """
    if not WANDB_AVAILABLE or not config.wandb.save_model_artifact:
        return

    artifact = wandb.Artifact(
        name=f"{model_type.lower()}-checkpoint",
        type="model",
        description=f"Trained {model_type} checkpoint",
        metadata={
            "model_type": model_type,
            "latent_dim": config.latent_dim,
            "beta": config.beta,
        },
    )
    artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(artifact)


def finish_wandb() -> None:
    """Finish wandb run."""
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


# =============================================================================
# Training Functions
# =============================================================================


def _get_model_type(model: nn.Module) -> str:
    """Detect VAE model type from class name."""
    class_name = model.__class__.__name__
    if "BatchAware" in class_name:
        return "batch-aware"
    elif "Conditional" in class_name:
        return "conditional"
    return "vanilla"


def _forward_model(
    model: nn.Module,
    batch: dict,
    device: str,
    model_type: str | None = None,
) -> dict:
    """Forward pass handling different VAE model types.

    Args:
        model: VAE model
        batch: Batch dict with 'features', optionally 'batch' and 'perturbation'
        device: Device to use
        model_type: Model type (auto-detected if None)

    Returns:
        Model output dict with 'recon', 'mu', 'logvar', 'z'
    """
    if model_type is None:
        model_type = _get_model_type(model)

    x = batch["features"].to(device)

    if model_type == "batch-aware":
        batch_idx = batch["batch"].to(device)
        return model(x, batch_idx)
    elif model_type == "conditional":
        pert_idx = batch["perturbation"].to(device)
        return model(x, pert_idx)
    else:
        return model(x)


def _get_model_latent(
    model: nn.Module,
    batch: dict,
    device: str,
    model_type: str | None = None,
    use_mean: bool = True,
):
    """Get latent representation handling different VAE model types.

    Args:
        model: VAE model
        batch: Batch dict with 'features', optionally 'batch' and 'perturbation'
        device: Device to use
        model_type: Model type (auto-detected if None)
        use_mean: If True, return latent mean. If False, sample.

    Returns:
        Latent tensor
    """
    if model_type is None:
        model_type = _get_model_type(model)

    x = batch["features"].to(device)

    if model_type == "conditional":
        pert_idx = batch["perturbation"].to(device)
        return model.get_latent(x, pert_idx, use_mean=use_mean)
    else:
        # VanillaVAE and BatchAwareVAE have same get_latent signature
        return model.get_latent(x, use_mean=use_mean)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    log_every: int = 10,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: VAE model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        log_every: Log every N batches

    Returns:
        Dictionary with average losses
    """
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    model_type = _get_model_type(model)

    for batch_idx, batch in enumerate(train_loader):
        x = batch["features"].to(device)

        # Forward pass
        optimizer.zero_grad()
        output = _forward_model(model, batch, device, model_type)

        # Compute loss
        losses = model.loss_function(x, output["recon"], output["mu"], output["logvar"])
        loss = losses["loss"]

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += losses["loss"].item()
        total_recon += losses["recon_loss"].item()
        total_kl += losses["kl_loss"].item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Validate model.

    Args:
        model: VAE model
        val_loader: Validation data loader
        device: Device to use

    Returns:
        Dictionary with average losses
    """
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = 0

    model_type = _get_model_type(model)

    for batch in val_loader:
        x = batch["features"].to(device)

        # Forward pass
        output = _forward_model(model, batch, device, model_type)
        losses = model.loss_function(x, output["recon"], output["mu"], output["logvar"])

        # Accumulate losses
        total_loss += losses["loss"].item()
        total_recon += losses["recon_loss"].item()
        total_kl += losses["kl_loss"].item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon_loss": total_recon / n_batches,
        "kl_loss": total_kl / n_batches,
    }


def train_vae(
    model: nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    config: TrainingConfig,
    checkpoint_path: str | Path | None = None,
    model_type: str = "VanillaVAE",
    n_features: int | None = None,
    extra_wandb_config: dict[str, Any] | None = None,
) -> tuple[nn.Module, TrainingHistory]:
    """Train a VAE model with optional Weights & Biases logging.

    Args:
        model: VAE model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        checkpoint_path: Path to save best model (optional)
        model_type: Type of VAE model (for wandb logging)
        n_features: Number of input features (for wandb logging)
        extra_wandb_config: Additional config to log to wandb

    Returns:
        Tuple of (trained model, training history)
    """
    device = config.device
    model = model.to(device)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
    )

    # Initialize wandb
    use_wandb = init_wandb(
        config=config,
        model=model,
        model_type=model_type,
        n_features=n_features,
        n_train_samples=len(train_dataset),
        n_val_samples=len(val_dataset),
        extra_config=extra_wandb_config,
    )

    # Setup optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    scheduler = None
    if config.use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
        )

    # Setup early stopping
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)

    # Training history
    history = TrainingHistory()
    best_val_loss = float("inf")

    # Create checkpoint directory
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training on {device}")
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(val_dataset):,}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    if use_wandb:
        print(f"  Wandb: {config.wandb.project} (run: {wandb.run.name})")
    print()

    try:
        for epoch in range(config.epochs):
            epoch_start = time.time()

            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, device, config.log_every)

            # Validate
            val_metrics = validate(model, val_loader, device)

            # Update scheduler
            if scheduler:
                scheduler.step(val_metrics["loss"])

            # Record history
            lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start
            history.train_loss.append(train_metrics["loss"])
            history.val_loss.append(val_metrics["loss"])
            history.train_recon_loss.append(train_metrics["recon_loss"])
            history.val_recon_loss.append(val_metrics["recon_loss"])
            history.train_kl_loss.append(train_metrics["kl_loss"])
            history.val_kl_loss.append(val_metrics["kl_loss"])
            history.learning_rates.append(lr)
            history.epoch_times.append(epoch_time)

            # Log to wandb
            if use_wandb and (epoch + 1) % config.wandb.log_freq == 0:
                log_wandb_metrics(
                    epoch=epoch + 1,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    learning_rate=lr,
                    epoch_time=epoch_time,
                )

            # Save best model
            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
                if checkpoint_path:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": best_val_loss,
                            "config": config.__dict__,
                        },
                        checkpoint_path,
                    )

                # Log best metrics to wandb
                if use_wandb:
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_epoch"] = epoch + 1

            # Log visualizations periodically (every 10 epochs or at end)
            if use_wandb and ((epoch + 1) % 10 == 0 or epoch == 0):
                if config.wandb.log_reconstructions:
                    log_wandb_reconstructions(
                        model=model,
                        val_loader=val_loader,
                        device=device,
                        n_samples=config.wandb.n_reconstruction_samples,
                    )
                if config.wandb.log_latent_space:
                    log_wandb_latent_space(
                        model=model,
                        val_loader=val_loader,
                        device=device,
                        max_samples=5000,
                    )

            # Print progress
            best_marker = " *" if is_best else ""
            print(
                f"Epoch {epoch + 1:3d}/{config.epochs} | "
                f"Train: {train_metrics['loss']:.4f} (R:{train_metrics['recon_loss']:.4f} K:{train_metrics['kl_loss']:.4f}) | "
                f"Val: {val_metrics['loss']:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {epoch_time:.1f}s{best_marker}"
            )

            # Check early stopping
            if early_stopping(val_metrics["loss"]):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Load best model if checkpoint exists
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"\nLoaded best model from epoch {checkpoint['epoch'] + 1}")

            # Save model artifact to wandb
            if use_wandb and config.wandb.save_model_artifact:
                save_wandb_artifact(model, config, checkpoint_path, model_type)

        # Final latent space visualization
        if use_wandb and config.wandb.log_latent_space:
            log_wandb_latent_space(
                model=model,
                val_loader=val_loader,
                device=device,
                max_samples=10000,  # More samples for final visualization
            )

    finally:
        # Always finish wandb run
        if use_wandb:
            finish_wandb()

    return model, history


@torch.no_grad()
def get_latent_embeddings(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_mean: bool = True,
) -> np.ndarray:
    """Get latent embeddings for all samples in dataset.

    Args:
        model: Trained VAE model
        dataset: Dataset to embed
        batch_size: Batch size for inference
        device: Device to use
        use_mean: If True, use latent mean. If False, sample.

    Returns:
        Latent embeddings as numpy array (n_samples, latent_dim)
    """
    model = model.to(device)
    model.eval()

    model_type = _get_model_type(model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []

    for batch in loader:
        z = _get_model_latent(model, batch, device, model_type, use_mean=use_mean)
        embeddings.append(z.cpu().numpy())

    return np.concatenate(embeddings, axis=0)
