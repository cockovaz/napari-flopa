import importlib.util
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

#: Above this the hierarchical linkage matrix (O(N²)) gets too large.
MAX_HIERARCHICAL_POINTS = 20_000

#: Parameter spec: name -> (kind, default, min, max) for numbers,
#: or ("choice", default, options) for a fixed set.
METHODS: dict[str, dict] = {
    "K-means": {
        "needs": "scipy",
        "params": {"n_clusters": ("int", 3, 2, 20)},
    },
    "Hierarchical": {
        "needs": "scipy",
        "params": {
            "n_clusters": ("int", 3, 2, 20),
            "linkage": (
                "choice",
                "ward",
                ["ward", "average", "complete", "single"],
            ),
        },
    },
    "Gaussian mixture": {
        "needs": "sklearn",
        "params": {
            "n_components": ("int", 3, 2, 20),
            "covariance_type": (
                "choice",
                "full",
                ["full", "tied", "diag", "spherical"],
            ),
        },
    },
    "DBSCAN": {
        "needs": "sklearn",
        "params": {
            "eps": ("float", 0.02, 0.001, 0.5),
            "min_samples": ("int", 10, 1, 10_000),
        },
    },
    "HDBSCAN": {
        "needs": "sklearn",
        "params": {"min_cluster_size": ("int", 50, 2, 100_000)},
    },
}

#: Short tag per method, used for the result column name.
METHOD_TAGS = {
    "K-means": "kmeans",
    "Hierarchical": "hier",
    "Gaussian mixture": "gmm",
    "DBSCAN": "dbscan",
    "HDBSCAN": "hdbscan",
}


def method_available(name: str) -> bool:
    return importlib.util.find_spec(METHODS[name]["needs"]) is not None


def _cluster_finite(xy: np.ndarray, method: str, params: dict) -> np.ndarray:
    if method == "K-means":
        from scipy.cluster.vq import kmeans2

        _, labels = kmeans2(xy, int(params["n_clusters"]), minit="++", seed=0)
        return labels
    if method == "Hierarchical":
        from scipy.cluster.hierarchy import fcluster, linkage

        if len(xy) > MAX_HIERARCHICAL_POINTS:
            raise ValueError(
                f"Hierarchical clustering is limited to "
                f"{MAX_HIERARCHICAL_POINTS:,} points (got {len(xy):,}); "
                "use K-means or a density method instead."
            )
        z = linkage(xy, method=params["linkage"])
        return fcluster(z, int(params["n_clusters"]), criterion="maxclust") - 1
    if method == "Gaussian mixture":
        from sklearn.mixture import GaussianMixture

        gm = GaussianMixture(
            n_components=int(params["n_components"]),
            covariance_type=params["covariance_type"],
            random_state=0,
        )
        return gm.fit_predict(xy)
    if method == "DBSCAN":
        from sklearn.cluster import DBSCAN

        return DBSCAN(
            eps=float(params["eps"]), min_samples=int(params["min_samples"])
        ).fit_predict(xy)
    if method == "HDBSCAN":
        from sklearn.cluster import HDBSCAN

        return HDBSCAN(
            min_cluster_size=int(params["min_cluster_size"])
        ).fit_predict(xy)
    raise ValueError(f"Unknown clustering method: {method}")


def cluster_points(
    g: ArrayLike, s: ArrayLike, method: str, **params: Any
) -> np.ndarray:
    """Cluster phasor points; returns an int label per point.

    ``-1`` marks noise (density methods) and points with non-finite G/S,
    which are left out of the fit. Use :func:`np.isfinite` on the inputs to
    tell the two apart.
    """
    if not method_available(method):
        raise ImportError(
            f"{method} needs {METHODS[method]['needs']}; "
            "install it with: pip install scikit-learn"
        )
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(g) & np.isfinite(s)
    labels = np.full(g.shape, -1, dtype=int)
    if ok.sum() < 2:
        return labels
    labels[ok] = _cluster_finite(
        np.column_stack([g[ok], s[ok]]), method, params
    )
    return labels
