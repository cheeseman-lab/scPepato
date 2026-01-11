"""
Distance metrics for comparing cell populations.

Adapted from pertpy's distance module for morphological features.
Reference: pertpy (Heumos et al., Nature Methods 2025)
"""

import numpy as np
from scipy.spatial.distance import cdist


def e_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Energy distance (E-distance) between two samples.

    Best performing metric in pertpy benchmarks for perturbation separation.

    E(X,Y) = 2*E||X-Y|| - E||X-X'|| - E||Y-Y'||

    Parameters
    ----------
    X : np.ndarray
        First sample, shape (n_samples_x, n_features)
    Y : np.ndarray
        Second sample, shape (n_samples_y, n_features)

    Returns
    -------
    float
        Energy distance between X and Y (non-negative, 0 = identical distributions)
    """
    n, m = len(X), len(Y)

    if n == 0 or m == 0:
        return np.nan

    # Mean pairwise distances
    xy = cdist(X, Y, metric="euclidean").mean()
    xx = cdist(X, X, metric="euclidean").mean() if n > 1 else 0.0
    yy = cdist(Y, Y, metric="euclidean").mean() if m > 1 else 0.0

    return float(2 * xy - xx - yy)


def mse_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Mean squared error between mean vectors of two samples.

    Simple baseline metric - compares centroids only.

    Parameters
    ----------
    X : np.ndarray
        First sample, shape (n_samples_x, n_features)
    Y : np.ndarray
        Second sample, shape (n_samples_y, n_features)

    Returns
    -------
    float
        MSE between mean vectors
    """
    if len(X) == 0 or len(Y) == 0:
        return np.nan

    mean_x = X.mean(axis=0)
    mean_y = Y.mean(axis=0)

    return float(np.mean((mean_x - mean_y) ** 2))


def wasserstein_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Wasserstein distance (Earth Mover's Distance) between two samples.

    Uses optimal transport to compute minimum cost of transforming
    one distribution into another.

    Parameters
    ----------
    X : np.ndarray
        First sample, shape (n_samples_x, n_features)
    Y : np.ndarray
        Second sample, shape (n_samples_y, n_features)

    Returns
    -------
    float
        Wasserstein distance

    Notes
    -----
    Requires POT (Python Optimal Transport) library.
    For large samples, consider subsampling for efficiency.
    """
    try:
        import ot
    except ImportError:
        raise ImportError(
            "POT library required for Wasserstein distance. Install with: pip install POT"
        )

    n, m = len(X), len(Y)

    if n == 0 or m == 0:
        return np.nan

    # Uniform weights
    a = np.ones(n) / n
    b = np.ones(m) / m

    # Cost matrix (pairwise Euclidean distances)
    M = cdist(X, Y, metric="euclidean")

    # Compute optimal transport
    return float(ot.emd2(a, b, M))


def mmd_distance(X: np.ndarray, Y: np.ndarray, kernel: str = "linear") -> float:
    """
    Maximum Mean Discrepancy (MMD) between two samples.

    Kernel-based distance that compares distributions in a
    reproducing kernel Hilbert space.

    Parameters
    ----------
    X : np.ndarray
        First sample, shape (n_samples_x, n_features)
    Y : np.ndarray
        Second sample, shape (n_samples_y, n_features)
    kernel : str
        Kernel type: 'linear' or 'rbf'

    Returns
    -------
    float
        MMD^2 value (squared MMD)
    """
    n, m = len(X), len(Y)

    if n == 0 or m == 0:
        return np.nan

    if kernel == "linear":
        # Linear kernel: k(x,y) = x·y
        xx = np.dot(X, X.T).mean()
        yy = np.dot(Y, Y.T).mean()
        xy = np.dot(X, Y.T).mean()

    elif kernel == "rbf":
        # RBF kernel with median heuristic for bandwidth
        from scipy.spatial.distance import pdist

        # Compute bandwidth using median heuristic
        all_data = np.vstack([X, Y])
        pairwise_dists = pdist(all_data, metric="euclidean")
        sigma = np.median(pairwise_dists)
        gamma = 1.0 / (2 * sigma**2) if sigma > 0 else 1.0

        def rbf_kernel(A, B):
            dists = cdist(A, B, metric="sqeuclidean")
            return np.exp(-gamma * dists)

        xx = rbf_kernel(X, X).mean()
        yy = rbf_kernel(Y, Y).mean()
        xy = rbf_kernel(X, Y).mean()

    else:
        raise ValueError(f"Unknown kernel: {kernel}. Use 'linear' or 'rbf'.")

    mmd_squared = xx + yy - 2 * xy

    # Can be slightly negative due to numerical errors
    return float(max(0, mmd_squared))


# Registry of all distance functions
DISTANCE_FUNCTIONS = {
    "e_distance": e_distance,
    "mse": mse_distance,
    "wasserstein": wasserstein_distance,
    "mmd_linear": lambda X, Y: mmd_distance(X, Y, kernel="linear"),
    "mmd_rbf": lambda X, Y: mmd_distance(X, Y, kernel="rbf"),
}


def compute_distance(X: np.ndarray, Y: np.ndarray, metric: str = "e_distance") -> float:
    """
    Compute distance between two samples using specified metric.

    Parameters
    ----------
    X : np.ndarray
        First sample
    Y : np.ndarray
        Second sample
    metric : str
        One of: 'e_distance', 'mse', 'wasserstein', 'mmd_linear', 'mmd_rbf'

    Returns
    -------
    float
        Distance value
    """
    if metric not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Unknown metric: {metric}. Available: {list(DISTANCE_FUNCTIONS.keys())}")

    return DISTANCE_FUNCTIONS[metric](X, Y)
