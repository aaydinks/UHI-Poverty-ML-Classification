"""Nearest neighbours on tract centroids, Moran's I, and train-neighbour exposure."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088


def knn(lat, lon, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Indices and great-circle distances (km) of each point's k nearest neighbours."""
    coords = np.radians(np.column_stack([np.asarray(lat, float), np.asarray(lon, float)]))
    tree = BallTree(coords, metric="haversine")
    dist, idx = tree.query(coords, k=k + 1)
    # drop self (distance 0); if exact duplicates exist, drop the first match of self
    self_first = idx[:, 0] == np.arange(len(coords))
    nbrs = np.where(self_first[:, None], idx[:, 1:], idx[:, :-1])
    d = np.where(self_first[:, None], dist[:, 1:], dist[:, :-1])
    return nbrs, d * EARTH_RADIUS_KM


def morans_i(x, nbrs: np.ndarray, permutations: int = 999, seed: int = 37) -> dict:
    """Moran's I with row-standardised binary kNN weights and a permutation p-value."""
    x = np.asarray(x, dtype=float)
    z = x - x.mean()
    denom = np.sum(z * z)

    def stat(v):
        return float(np.sum(v * v[nbrs].mean(axis=1)) / denom)

    observed = stat(z)
    rng = np.random.default_rng(seed)
    perms = np.array([stat(rng.permutation(z)) for _ in range(permutations)])
    p = (1 + np.sum(perms >= observed)) / (permutations + 1)
    return {"I": observed, "expected_I": -1 / (len(x) - 1), "perm_mean": float(perms.mean()),
            "perm_sd": float(perms.std(ddof=1)), "p_value": float(p),
            "permutations": permutations, "k": nbrs.shape[1], "n": len(x)}


def neighbour_in_train_share(nbrs: np.ndarray, splits) -> dict:
    """Share of validation tracts with at least one of their k neighbours in training.

    `splits` is an iterable of (train_idx, test_idx) over the same rows as `nbrs`.
    """
    n = nbrs.shape[0]
    hit = np.zeros(n, dtype=bool)
    tested = np.zeros(n, dtype=bool)
    n_train_nbrs = np.zeros(n, dtype=float)
    for train_idx, test_idx in splits:
        in_train = np.zeros(n, dtype=bool)
        in_train[train_idx] = True
        counts = in_train[nbrs[test_idx]].sum(axis=1)
        hit[test_idx] = counts > 0
        n_train_nbrs[test_idx] = counts
        tested[test_idx] = True
    return {"n_validation": int(tested.sum()),
            "share_with_train_neighbour": float(hit[tested].mean()),
            "mean_train_neighbours": float(n_train_nbrs[tested].mean())}
