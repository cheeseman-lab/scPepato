"""Analysis module for batch effects and response heterogeneity."""

from .batch_effects import (
    BatchMetrics,
    batch_mixing_score,
    compute_batch_metrics,
    perturbation_retrieval_across_batches,
    within_between_variance_ratio,
)
from .responders import (
    ResponderResult,
    analyze_perturbation_heterogeneity,
    compute_responder_fraction,
    compute_responder_fractions,
    detect_bimodality,
)

__all__ = [
    "batch_mixing_score",
    "within_between_variance_ratio",
    "perturbation_retrieval_across_batches",
    "compute_batch_metrics",
    "BatchMetrics",
    "detect_bimodality",
    "compute_responder_fraction",
    "analyze_perturbation_heterogeneity",
    "compute_responder_fractions",
    "ResponderResult",
]
