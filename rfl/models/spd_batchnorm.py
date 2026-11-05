"""Combination of SPD batch-norm running means across federated clients.

The batch-norm layers themselves are those of yetanotherspdnet, used as they are. They
follow Gallet, Mian, Bouchard and Ginolhac, "On batch normalization for SPDnet", Signal
Processing 250 (2027) 110793, hal-05742151 ("the paper").

Each layer keeps a running mean R, used at evaluation in place of the batch mean. When
the server shares the normalisation layers (`fl_fedbn=False`), the running means of the
clients are combined in the geometry of the layer by `weighted_spd_mean`, with the
aggregation weights: uniform by default, proportional to the client sizes with
`--weight-by-samples`. With the size weights, for the arithmetic, harmonic and
log-Euclidean means, if each client held the exact mean of its records, this would give
the mean over the union of the records. For the affine-invariant mean the weighted Karcher mean of the
client means is the natural combination, not an identity. For the GAH mean the
combination is the geometric midpoint of the weighted arithmetic and harmonic means of
the client running means.
"""

from __future__ import annotations

from typing import Sequence

import torch

__all__ = ["weighted_spd_mean", "geometric_mean_2", "SHARED_MEAN_TYPES"]

# Mean types whose running statistics the server can share (fl_fedbn=False).
SHARED_MEAN_TYPES = ("affine_invariant", "log_euclidean", "arithmetic", "harmonic",
                     "geometric_arithmetic_harmonic")


# --------------------------------------------------------------------------- #
#  Matrix functions (float64, symmetric input)
# --------------------------------------------------------------------------- #
def _eig_fun(X: torch.Tensor, fun) -> torch.Tensor:
    X = 0.5 * (X + X.transpose(-1, -2))
    w, V = torch.linalg.eigh(X)
    return (V * fun(w).unsqueeze(-2)) @ V.transpose(-1, -2)


def _inv(X):
    return _eig_fun(X, lambda w: 1.0 / w)


def _sqrt(X):
    return _eig_fun(X, torch.sqrt)


def _invsqrt(X):
    return _eig_fun(X, lambda w: 1.0 / torch.sqrt(w))


def _log(X):
    return _eig_fun(X, torch.log)


def _exp(X):
    return _eig_fun(X, torch.exp)


def geometric_mean_2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A #_{1/2} B = A^{1/2} (A^{-1/2} B A^{-1/2})^{1/2} A^{1/2}."""
    As, Ai = _sqrt(A), _invsqrt(A)
    return As @ _sqrt(Ai @ B @ Ai) @ As


def _karcher_mean(S: torch.Tensor, w: torch.Tensor, n_iterations: int,
                  tol: float) -> torch.Tensor:
    """Weighted affine-invariant mean: Riemannian gradient descent with step control.

    G <- G^{1/2} expm(t T) G^{1/2}, T = sum_i w_i logm(G^{-1/2} S_i G^{-1/2}) (the
    fixed point of the paper, Eq. 3, at t = 1). The step t is halved whenever the
    gradient norm ||T|| does not decrease, so the iteration cannot diverge on widely
    spread matrices. Starts from the log-Euclidean mean.
    """
    def grad(G):
        Gi = _invsqrt(G)
        return (w * _log(Gi @ S @ Gi)).sum(0)

    G = _exp((w * _log(S)).sum(0))
    T = grad(G)
    norm, t = float(torch.linalg.norm(T)), 1.0
    for _ in range(n_iterations):
        if norm < tol:
            break
        Gs = _sqrt(G)
        G_new = Gs @ _exp(t * T) @ Gs
        G_new = 0.5 * (G_new + G_new.transpose(-1, -2))
        T_new = grad(G_new)
        n_new = float(torch.linalg.norm(T_new))
        if n_new < norm:
            G, T, norm, t = G_new, T_new, n_new, min(1.0, 2.0 * t)
        else:
            t *= 0.5
            if t < 1e-8:
                break
    return G


def weighted_spd_mean(mats: Sequence[torch.Tensor], weights: Sequence[float],
                      mean_type: str, n_iterations: int = 100,
                      tol: float = 1e-10) -> torch.Tensor:
    """Weighted mean of SPD matrices in the geometry `mean_type`.

    arithmetic        sum_i w_i S_i
    harmonic          (sum_i w_i S_i^{-1})^{-1}
    log_euclidean     expm(sum_i w_i logm S_i)
    affine_invariant  weighted Karcher mean (Eq. 3 of the paper, with step control)
    geometric_arithmetic_harmonic   arithmetic #_{1/2} harmonic

    The weights are normalised. Computed in float64, returned in the input dtype.

    Raises:
        ValueError: for another mean type.
    """
    dtype, device = mats[0].dtype, mats[0].device
    S = torch.stack([m.detach().to(device).double() for m in mats])
    w = torch.tensor([float(x) for x in weights], dtype=torch.float64, device=device)
    w = (w / w.sum()).view(-1, 1, 1)
    if mean_type == "arithmetic":
        out = (w * S).sum(0)
    elif mean_type == "harmonic":
        out = _inv((w * _inv(S)).sum(0))
    elif mean_type == "log_euclidean":
        out = _exp((w * _log(S)).sum(0))
    elif mean_type == "geometric_arithmetic_harmonic":
        out = geometric_mean_2((w * S).sum(0), _inv((w * _inv(S)).sum(0)))
    elif mean_type == "affine_invariant":
        out = _karcher_mean(S, w, n_iterations, tol)
    else:
        raise ValueError(f"no shared running mean for mean_type={mean_type!r}; "
                         f"supported: {SHARED_MEAN_TYPES}")
    return (0.5 * (out + out.transpose(-1, -2))).to(dtype)
