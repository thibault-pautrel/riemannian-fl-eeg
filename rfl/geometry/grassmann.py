"""Grassmann manifold Gr(d, p) represented by orthogonal projectors.

A p-dimensional subspace of R^d is represented by its orthogonal projector

    Gr(d, p) = {P in S_d : P^2 = P, tr P = p},

a compact submanifold of the symmetric matrices S_d with the Frobenius inner product
(see the geometry appendix of the DP-RFedProj paper). Intrinsic dimension p (d - p), ambient dimension d (d + 1) / 2.
"""
from __future__ import annotations

import numpy as np


def intrinsic_dim(d: int, p: int) -> int:
    """Dimension d_M = p (d - p) of Gr(d, p)."""
    return p * (d - p)


def ambient_dim(d: int) -> int:
    """Dimension d (d + 1) / 2 of the symmetric matrices S_d."""
    return d * (d + 1) // 2


def sym(A: np.ndarray) -> np.ndarray:
    """Symmetric part (A + A^T) / 2."""
    return 0.5 * (A + A.T)


def basis(P: np.ndarray, p: int) -> np.ndarray:
    """Orthonormal basis U (d x p) of the range of P, so that P = U U^T."""
    return np.linalg.eigh(sym(P))[1][:, -p:]


def tangent_proj(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Orthogonal projection of X in S_d onto the tangent space at P.

    P_P(X) = P X (I - P) + (I - P) X P. It keeps the two off-diagonal blocks of X in
    the splitting range(P) + range(P)^perp.
    """
    PX = P @ X
    PXP = PX @ P
    return (PX - PXP) + (PX - PXP).T


def proj(A: np.ndarray, p: int) -> np.ndarray:
    """Nearest-point projection Pi of A in S_d onto Gr(d, p): projector on the top-p eigenvectors."""
    U = basis(A, p)
    return U @ U.T


# --------------------------------------------------------------------------- #
#  Retraction and inverse retraction used by RetAvg (PriRFed's aggregation)
# --------------------------------------------------------------------------- #
# As on Stiefel, RetAvg uses the projection-based retraction
# R_P(X) = Pi(P + X). Its inverse has no explicit formula, but it is one linear solve.
# Write P = U U^T and complete U with U_c, so that X = U_c K U^T + U K^T U_c^T for a
# (d - p) x p matrix K. Q = Pi(P + X) holds when range(Q) is invariant under P + X,
# that is V_c^T (P + X) V = 0 for bases V of range(Q) and V_c of its complement. This is
# linear in K: p (d - p) equations for p (d - p) unknowns.

def retraction(P: np.ndarray, X: np.ndarray, p: int) -> np.ndarray:
    """Projection-based retraction R_P(X) = Pi(P + X), for X in the tangent space at P."""
    return proj(P + X, p)


def _bases(P: np.ndarray, p: int) -> tuple:
    V = np.linalg.eigh(sym(P))[1]
    return V[:, -p:], V[:, :-p]


def inverse_retraction(P: np.ndarray, Q: np.ndarray, p: int) -> np.ndarray:
    """R_P^{-1}(Q): the tangent vector X at P with Pi(P + X) = Q, by one linear solve."""
    U, Uc = _bases(P, p)
    V, Vc = _bases(Q, p)
    A1, B1, A2, B2 = Vc.T @ Uc, U.T @ V, Vc.T @ U, Uc.T @ V
    r = A1.shape[1]
    n = r * p
    # V_c^T X V = A1 K B1 + A2 K^T B2, and vec(K^T) = T vec(K) (column-major vec).
    i, j = np.meshgrid(np.arange(r), np.arange(p), indexing="ij")
    T = np.zeros((n, n))
    T[(j + i * p).ravel(), (i + j * r).ravel()] = 1.0
    L = np.kron(B1.T, A1) + np.kron(B2.T, A2) @ T
    K = np.linalg.solve(L, -(A2 @ B1).reshape(-1, order="F")).reshape(r, p, order="F")
    return Uc @ K @ U.T + U @ K.T @ Uc.T


def random_point(d: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random point of Gr(d, p)."""
    U, _ = np.linalg.qr(rng.standard_normal((d, p)))
    return U @ U.T


def sym_gaussian(d: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Isotropic Gaussian N(0, sigma^2 I) on S_d with the Frobenius inner product.

    sym(N) with N_ij ~ N(0, sigma^2) has variance sigma^2 along every unit vector of an
    orthonormal basis of S_d (E_ii, and (E_ij + E_ji) / sqrt 2 for i < j).
    """
    return sym(sigma * rng.standard_normal((d, d)))
