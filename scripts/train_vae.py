#!/usr/bin/env python3
"""Train VAE on CellProfiler features with Weights & Biases tracking.

Usage:
    python scripts/train_vae.py [options]

Examples:
    # Quick test (1K cells, 10 epochs, no wandb)
    python scripts/train_vae.py --n-cells 1000 --epochs 10 --no-wandb

    # Full training with wandb (50K cells)
    python scripts/train_vae.py --n-cells 50000 --epochs 100 --wandb-project scpepato-vae

    # Train batch-aware VAE
    python scripts/train_vae.py --model batch-aware --n-cells 50000

    # Train conditional VAE
    python scripts/train_vae.py --model conditional --n-cells 50000

    # Use SLURM for large runs
    srun -p 24 --gpus=1 --mem=64gb --time=04:00:00 \\
        conda run -n scpepato python scripts/train_vae.py --n-cells 100000
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from scpepato.data import prepare_vae_data
from scpepato.models import (
    BatchAwareVAE,
    ConditionalVAE,
    TrainingConfig,
    VanillaVAE,
    WandbConfig,
    get_latent_embeddings,
    train_vae,
)

# Default paths
DEFAULT_PARQUET_DIR = Path(
    "/lab/ops_analysis/cheeseman/mayon-analysis/analysis/brieflow_output/aggregate/parquets"
)
DEFAULT_PATTERN = "*CeCl-all*GLYCORNA__filtered.parquet"
OUTPUT_DIR = Path("outputs/vae")


def create_model(
    model_type: str,
    n_features: int,
    n_batches: int,
    n_perturbations: int,
    latent_dim: int,
    hidden_dims: list[int],
    dropout: float,
    beta: float,
):
    """Create VAE model of specified type."""
    if model_type == "vanilla":
        return VanillaVAE(
            input_dim=n_features,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            beta=beta,
        )
    elif model_type == "batch-aware":
        return BatchAwareVAE(
            input_dim=n_features,
            n_batches=n_batches,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            batch_embed_dim=16,
            dropout=dropout,
            beta=beta,
        )
    elif model_type == "conditional":
        return ConditionalVAE(
            input_dim=n_features,
            n_perturbations=n_perturbations,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            pert_embed_dim=32,
            dropout=dropout,
            beta=beta,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Train VAE on CellProfiler features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="vanilla",
        choices=["vanilla", "batch-aware", "conditional"],
        help="VAE model type",
    )

    # Data arguments
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help="Directory containing filtered parquet files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help="Glob pattern for parquet files in directory",
    )
    parser.add_argument(
        "--batch-col",
        type=str,
        default="plate_well",
        help="Column for batch labels (use 'plate_well' to combine plate+well)",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=50000,
        help="Number of cells to use",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction for validation",
    )

    # Model architecture arguments
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=50,
        help="Latent dimension (matches PCA for fair comparison)",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help="Hidden layer dimensions",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Beta for KL term (1.0 = standard VAE)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )

    # Wandb arguments
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="scpepato-vae",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="Weights & Biases entity (team/username). Uses WANDB_ENTITY env var if set.",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Wandb run name (auto-generated if not specified)",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        nargs="*",
        default=[],
        help="Wandb run tags",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: model type)",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Experiment name
    exp_name = args.name or f"{args.model}_vae"
    output_dir = args.output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model type mapping for nice names
    model_type_names = {
        "vanilla": "VanillaVAE",
        "batch-aware": "BatchAwareVAE",
        "conditional": "ConditionalVAE",
    }

    print("=" * 70)
    print(f"VAE TRAINING: {model_type_names[args.model]}")
    print("=" * 70)
    print(f"Data: {args.parquet_dir.name}/{args.pattern}")
    print(f"Cells: {args.n_cells:,}")
    print(f"Model: {args.model}")
    print(f"Batch col: {args.batch_col}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"Beta: {args.beta}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"Wandb: {'disabled' if args.no_wandb else args.wandb_project}")
    print()

    # Load data
    print("Loading data...")
    data = prepare_vae_data(
        parquet_dir=args.parquet_dir,
        pattern=args.pattern,
        n_rows=args.n_cells,
        val_fraction=args.val_fraction,
        batch_col=args.batch_col,
        random_state=args.seed,
    )

    print(f"  Features: {data['n_features']}")
    print(f"  Train samples: {len(data['train_dataset']):,}")
    print(f"  Val samples: {len(data['val_dataset']):,}")
    print(f"  Batches: {data['n_batches']}")
    print(f"  Perturbations: {data['n_perturbations']}")
    print()

    # Create model
    model = create_model(
        model_type=args.model,
        n_features=data["n_features"],
        n_batches=data["n_batches"],
        n_perturbations=data["n_perturbations"],
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        beta=args.beta,
    )

    print(f"Model: {model.__class__.__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Configure wandb
    wandb_config = WandbConfig(
        enabled=not args.no_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or f"{exp_name}_{args.n_cells // 1000}k",
        tags=[args.model, f"beta={args.beta}", f"latent={args.latent_dim}"] + args.wandb_tags,
        notes=f"Training {model_type_names[args.model]} on {args.n_cells:,} cells",
        log_reconstructions=True,
        log_latent_space=True,
        save_model_artifact=True,
    )

    # Training config
    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        beta=args.beta,
        device=args.device,
        wandb=wandb_config,
    )

    # Extra config for wandb
    extra_config = {
        "parquet_dir": str(args.parquet_dir),
        "pattern": args.pattern,
        "batch_col": args.batch_col,
        "n_cells": args.n_cells,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
    }

    # Train
    print("Training...")
    model, history = train_vae(
        model=model,
        train_dataset=data["train_dataset"],
        val_dataset=data["val_dataset"],
        config=config,
        checkpoint_path=output_dir / "best_model.pt",
        model_type=model_type_names[args.model],
        n_features=data["n_features"],
        extra_wandb_config=extra_config,
    )

    # Save history
    history.save(output_dir / "history.json")

    # Save normalizer
    data["normalizer"].save(output_dir / "normalizer.npz")

    # Get embeddings
    print("\nGetting latent embeddings...")
    train_embeddings = get_latent_embeddings(model, data["train_dataset"], device=args.device)
    val_embeddings = get_latent_embeddings(model, data["val_dataset"], device=args.device)

    # Save embeddings
    np.save(output_dir / "train_embeddings.npy", train_embeddings)
    np.save(output_dir / "val_embeddings.npy", val_embeddings)

    print("\nTraining complete!")
    print(f"  Best val loss: {min(history.val_loss):.4f}")
    print(f"  Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
