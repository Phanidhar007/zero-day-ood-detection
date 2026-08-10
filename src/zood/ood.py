"""Manual OOD scorers (energy-based + Mahalanobis). pytorch-ood NOT required.

Both scorers follow the "higher score = more OOD-like" convention. The energy
score is computed on the IDS model's class logits; the Mahalanobis distance is
computed on the penultimate-layer embeddings of a Gaussian fit over the known
classes.

The raw energy term ``-E(x) = T * logsumexp(logits / T)`` measures how
confidently the input falls inside the known-class simplex; its *sign* is
calibrated empirically in :mod:`zood.evaluate` so that high values correspond
to OOD (this matches how production detectors pick a threshold direction).
"""

from __future__ import annotations

import numpy as np


def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(
        np.exp(a - m).sum(axis=axis)
    )


def energy_ood_score(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Raw energy-based OOD signal: ``T * logsumexp(logits / T)``.

    = ``-E(x)`` where ``E(x) = -T log sum exp(logits/T)`` is the energy.
    High values mean the input maps confidently into the known-class simplex.
    The evaluation step orients the sign so that **higher => OOD**.
    """
    return T * _logsumexp(np.asarray(logits, dtype=np.float64) / T)


def mahalanobis_fit(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    shrink: float = 0.1,
):
    """Fit per-class Gaussians over known-class embeddings.

    Returns ``(means, cov_inv, cov)``:

    - ``means`` shape ``(n_classes, d)`` - per-class centroid,
    - ``cov`` - pooled covariance with shrinkage
      ``(1-shrink) * S + shrink * diag(diag(S))``, regularised with ``1e-6 I``,
    - ``cov_inv`` - its (pseudo)inverse.
    """
    d = embeddings.shape[1]
    means = np.zeros((n_classes, d))
    cov = np.zeros((d, d))
    total = 0
    for c in range(n_classes):
        Xc = embeddings[labels == c]
        if len(Xc) == 0:
            continue
        mu = Xc.mean(axis=0)
        means[c] = mu
        diff = Xc - mu
        cov += diff.T @ diff
        total += len(Xc)
    cov = cov / max(total - n_classes, 1)

    diag = np.diag(np.diag(cov))
    cov_reg = (1.0 - shrink) * cov + shrink * diag
    cov_reg = cov_reg + 1e-6 * np.eye(d)
    cov_inv = np.linalg.pinv(cov_reg)
    return means, cov_inv, cov_reg


def mahalanobis_ood_scores(
    embeddings: np.ndarray,
    means: np.ndarray,
    cov_inv: np.ndarray,
) -> np.ndarray:
    """Per-sample min Mahalanobis distance to the nearest class centroid.

    Higher distance = more OOD-like. Uses the Mahalanobis norm
    ``(x-mu) @ cov_inv @ (x-mu).T`` minimised over the known classes.
    """
    d = means.shape[1]
    x = embeddings.reshape(-1, 1, d)
    mu = means.reshape(1, -1, d)
    diff = x - mu                                   # (n, n_classes, d)
    dists = np.einsum("nkd,dl,nkl->nk", diff, cov_inv, diff)
    return np.sqrt(np.maximum(dists, 0.0)).min(axis=1)


def fit_mahalanobis_reference(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    n_classes: int,
    shrink: float = 0.1,
):
    """Convenience: fit the Mahalanobis reference on known-class embeddings."""
    return mahalanobis_fit(train_embeddings, train_labels, n_classes, shrink)
