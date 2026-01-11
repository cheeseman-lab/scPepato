"""Distance metrics for comparing cell populations."""

from .compare import DistanceComparison, compare_spaces
from .metrics import (
    DISTANCE_FUNCTIONS,
    compute_distance,
    e_distance,
    mmd_distance,
    mse_distance,
    wasserstein_distance,
)
from .testing import (
    DistanceResult,
    compute_all_distances,
    compute_perturbation_distances,
    permutation_test,
)

__all__ = [
    "e_distance",
    "mse_distance",
    "wasserstein_distance",
    "mmd_distance",
    "compute_distance",
    "DISTANCE_FUNCTIONS",
    "permutation_test",
    "compute_perturbation_distances",
    "compute_all_distances",
    "DistanceResult",
    "compare_spaces",
    "DistanceComparison",
]
