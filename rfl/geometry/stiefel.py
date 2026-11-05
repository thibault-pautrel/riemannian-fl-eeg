"""Stiefel-manifold primitives used by the federated aggregation rules and by DP.

A point of the compact Stiefel manifold St(n, p) is a tall matrix W of shape (n, p),
with n >= p and W^T W = I_p. This is the shape of an SPDNet BiMap weight, whose
forward map is Sigma -> W^T Sigma W. The routines reject wide matrices, because the
tangent algebra below assumes the column-orthonormal convention.

All functions act on a single 2-D matrix. Callers loop over named parameters.

Where each function is used:

    polar_projection          uf(.), the SPDNet post-step re-projection and the last
                              step of ProjAvg, RLAvg and RetAvg.
    tangent_projection        P_W, the projection of per-sample gradients (before
                              clipping) and of the ambient noise in the DP mechanism.
    lifting, retraction       RLAvg (RFedRL).
    inverse_polar_retraction  RetAvg (the PriRFed baseline).

The lifting of RLAvg is a first-order approximation of the inverse retraction. Its
error is O(eps^2) in the distance between the base point and the client point. The
inverse polar retraction of RetAvg is exact wherever the polar retraction is
invertible.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "polar_projection",
    "tangent_projection",
    "lifting",
    "retraction",
    "inverse_polar_retraction",
]



class RetractionNotInvertible(ValueError):
    """A client point lies where the polar retraction cannot be inverted (RetAvg).

    Training has moved too far in one round, typically with a learning rate too large:
    the run is recorded as diverged (rfl/training/divergence.py).
    """

def polar_projection(X: torch.Tensor) -> torch.Tensor:
    """Orthogonal polar factor uf(X), the nearest column-orthonormal matrix to X.

    With the thin SVD X = U S Vh, uf(X) = U Vh. It solves the orthogonal Procrustes
    problem argmin_{Q^T Q = I} ||Q - X||_F and is idempotent on St(n, p).

    Args:
        X: Matrix of shape (n, p) with n >= p.

    Returns:
        A column-orthonormal matrix of shape (n, p).

    Raises:
        ValueError: If X is wide (n < p).
    """
    if X.shape[0] < X.shape[1]:
        raise ValueError(
            f"polar_projection expects a tall (n >= p) matrix, got {tuple(X.shape)}. "
            "BiMap Stiefel weights are (n_in, n_out) with n_out <= n_in.")
    U, _S, Vh = torch.linalg.svd(X.clone(), full_matrices=False)
    return U @ Vh


def tangent_projection(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Orthogonal projection P_W(X) of X onto the tangent space of St(n, p) at W.

    The tangent space at W is {V : W^T V + V^T W = 0}, and
    P_W(X) = X - W sym(W^T X) with sym(A) = (A + A^T) / 2.

    In the DP mechanism P_W is evaluated at the current iterate, so it is the same
    map for neighbouring datasets. Being an orthogonal projector, it does not
    increase norms, so clipping after projection keeps the sensitivity Delta = 2C/B.
    Projected isotropic Gaussian noise is isotropic in the tangent space, so mu0 is
    unchanged.

    Args:
        W: A point of St(n, p).
        X: A matrix of the same shape (an ambient gradient or noise sample).

    Returns:
        The tangent vector P_W(X).
    """
    W = W.clone()
    X = X.clone()
    # W^T X + X^T W = 2 sym(W^T X), hence the factor 0.5.
    return X - 0.5 * W @ (W.T @ X + X.T @ W)


class StiefelChart:
    """Orthonormal coordinates of the tangent space of St(n, p) at uf(W), in float64.

    The base point is X = uf(W), so the chart is exact even when W is orthonormal only
    up to float32 round-off. With X_perp an
    orthonormal basis of the complement of span(X), a tangent vector V = X A + X_perp B
    (A skew, p x p, B of shape (n - p, p)) has the coordinates
    (sqrt(2) A_ij for i < j, B), of dimension p (p - 1) / 2 + (n - p) p. The
    coordinate norm equals the Frobenius norm of V.

    The DP mechanism clips and noises in these coordinates (rfl/federated/privacy/dp_grad.py):
    the step is then a Gaussian mechanism in R^dim with sensitivity 2C/B, whatever the
    precision of W.
    """

    def __init__(self, W: torch.Tensor):
        X = polar_projection(W.detach().to(torch.float64))
        n, p = X.shape
        Q, _ = torch.linalg.qr(X, mode="complete")
        self.X, self.X_perp = X, Q[:, p:]
        self.n, self.p = n, p
        self._iu = torch.triu_indices(p, p, 1, device=X.device)
        self.dim = p * (p - 1) // 2 + (n - p) * p
        self.dtype = W.dtype

    def coords(self, G: torch.Tensor) -> torch.Tensor:
        """Coordinates of P_X(G), the tangent part of the ambient matrix G."""
        G = G.detach().to(self.X.device, torch.float64)
        M = self.X.T @ G
        A = 0.5 * (M - M.T)
        a = math.sqrt(2.0) * A[self._iu[0], self._iu[1]]
        b = (self.X_perp.T @ G).reshape(-1)
        return torch.cat([a, b])

    def vector(self, c: torch.Tensor) -> torch.Tensor:
        """Tangent vector at X with coordinates c, as an (n, p) matrix (float64)."""
        k = self._iu.shape[1]
        A = torch.zeros(self.p, self.p, dtype=torch.float64, device=self.X.device)
        A[self._iu[0], self._iu[1]] = c[:k].to(torch.float64) / math.sqrt(2.0)
        A = A - A.T
        B = c[k:].to(torch.float64).reshape(self.n - self.p, self.p)
        return self.X @ A + self.X_perp @ B

    def project(self, G: torch.Tensor) -> torch.Tensor:
        """P_X(G) in the dtype of the base weight."""
        return self.vector(self.coords(G)).to(self.dtype)


def lifting(W: torch.Tensor, W_target: torch.Tensor) -> torch.Tensor:
    """First-order inverse retraction L_W(W_target) = P_W(W_target - W).

    Used by RLAvg to average client displacements in the tangent space at the
    current global point.
    """
    return tangent_projection(W, W_target.clone() - W.clone())


def retraction(W: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Polar retraction R_W(V) = uf(W + V) of the tangent vector V at W."""
    return polar_projection(W.clone() + V.clone())


def inverse_polar_retraction(W: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Inverse polar retraction: the tangent vector V at W with uf(W + V) = Y.

    This is the R_W^{-1} required by the RetAvg aggregation of PriRFed
    (Huang et al. 2026, Eq. 8).

    Derivation. Write W + V = Y S with S symmetric positive definite, so that
    uf(W + V) = Y. Tangency at W, W^T V + V^T W = 0 with V = Y S - W, gives the
    continuous Lyapunov equation

        M S + S M^T = 2 I_p,    M = W^T Y.

    It has a unique SPD solution when M + M^T is positive definite, which holds when
    Y is close enough to W. Then V = Y S - W is tangent at W and uf(W + V) = Y.
    In particular inverse_polar_retraction(W, W) = 0.

    Args:
        W: Base point on St(n, p).
        Y: Target point on St(n, p), same shape as W.

    Returns:
        The tangent vector V at W, of shape (n, p). The inputs are detached.

    Raises:
        ValueError: If the shapes differ, if the matrices are wide, or if an
            eigenvalue of M = W^T Y has a non-positive real part (Y is outside the
            region where the polar retraction at W is invertible).
    """
    if W.shape != Y.shape:
        raise ValueError(f"shape mismatch: W {tuple(W.shape)} vs Y {tuple(Y.shape)}")
    if W.shape[0] < W.shape[1]:
        raise ValueError(
            f"inverse_polar_retraction expects tall (n >= p) matrices, got "
            f"{tuple(W.shape)}.")

    W = W.detach().clone()
    Y = Y.detach().clone()
    M = W.T @ Y

    # The inverse exists when M S + S M^T = 2 I has a symmetric positive definite
    # solution S, which holds when every eigenvalue of M has a positive real part
    # (then S = int_0^inf exp(-M t) 2 exp(-M^T t) dt). M + M^T > 0 is sufficient, not
    # necessary.
    re_min = float(torch.linalg.eigvals(M).real.min())
    if re_min <= 1e-9:
        raise RetractionNotInvertible(
            "inverse_polar_retraction: W^T Y has an eigenvalue with non-positive real "
            f"part ({re_min:.3e}); Y is outside the region where the polar retraction "
            "at W is invertible.")

    S = _solve_lyapunov_spd(
        M, 2.0 * torch.eye(M.shape[0], dtype=M.dtype, device=M.device))
    S = 0.5 * (S + S.T)  # remove round-off asymmetry
    return Y @ S - W


def _solve_lyapunov_spd(M: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Solve M S + S M^T = Q in Kronecker form, (I (x) M + M (x) I) vec(S) = vec(Q).

    The p^2 x p^2 system is non-singular when no two eigenvalues of M sum to zero,
    which eigenvalues with positive real parts guarantee. BiMap blocks are small (p <= ~20), so a direct
    solve is cheap and accurate. M is not symmetric in general, so diagonalising
    its symmetric part would not be valid.
    """
    p = M.shape[0]
    eye = torch.eye(p, dtype=M.dtype, device=M.device)
    K = torch.kron(eye, M) + torch.kron(M, eye)
    s = torch.linalg.solve(K, Q.reshape(-1))
    return s.reshape(p, p)
