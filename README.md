# scPepato

Single-Cell Perturbation Analysis Toolkit for Optical Pooled Screening.

## Installation

```bash
conda create -n scpepato -c conda-forge python=3.11 uv pip -y
conda activate scpepato
uv pip install -r pyproject.toml
uv pip install -e .
```

## Usage

```bash
python scripts/run_embedding.py --n-cells 50000
```

## Documentation

- `CLAUDE.md` - Development guidance
- `docs/PLAN.md` - Current implementation plan
- `docs/BRIEFLOW.md` - Brieflow data format reference
