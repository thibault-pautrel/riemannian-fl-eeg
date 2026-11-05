"""Synthetic federation for PCA: a spiked covariance model with heterogeneous clients.

Client i holds m records drawn from the spiked covariance model (Johnstone, 2001),

    z = U_i diag(sqrt(lam)) s + (nu / sqrt(d)) w,      s ~ N(0, I_p),  w ~ N(0, I_d),

then projected onto the unit ball, z <- z / max(1, ||z||), the standard normalisation
of differentially private PCA (Chaudhuri et al., 2013; Dwork et al., 2014). U_i is an
orthonormal basis of a client-specific subspace: a common basis U plus a perturbation of
size `het`, so the clients disagree on the principal subspace. The perturbation is scaled
by 1 / sqrt(d - p), so the angle between the client subspaces and U is the same for every
(d, p). The signal variances
lam_j average 0.06 whatever p, and the noise energy nu^2 does not depend on d.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Federation:
    """Client datasets and the pooled quantities used for evaluation."""
    clients: List[np.ndarray]       # n arrays of shape (m, d)
    d: int
    p: int
    sigma_bar: np.ndarray           # (1/n) sum_i (1/m) Z_i^T Z_i
    f_star: float                   # min of F over Gr(d, p)
    P_star: np.ndarray              # minimiser of F: projector on the top-p eigenvectors

    @property
    def n(self) -> int:
        return len(self.clients)

    @property
    def m_star(self) -> int:
        return min(len(Z) for Z in self.clients)


def make_federation(*, n: int, m: int, d: int, p: int, signal: float = 0.06,
                    decay: float = 0.5, noise: float = 0.3, het: float = 0.3,
                    seed: int = 0) -> Federation:
    """Draw a federation of n clients with m records each.

    Args:
        n: number of clients.
        m: records per client.
        d: ambient dimension of the records.
        p: dimension of the principal subspace.
        signal: mean signal variance per principal direction, mean(lam).
        decay: lam_j proportional to 1 + decay (p - j) / p, j = 1..p.
        noise: noise energy nu^2.
        het: size of the client-specific perturbation of the subspace. U_i is the Q factor
            of U + het G_i / sqrt(d - p), so each column of U is tilted by an angle of
            about arctan(het), 17 degrees, whatever d and p.
        seed: random seed of the data.
    """
    rng = np.random.default_rng((1, seed))         # stream distinct from the protocol's
    w = 1.0 + decay * (p - np.arange(1, p + 1)) / p
    lam = signal * w / w.mean()
    U, _ = np.linalg.qr(rng.standard_normal((d, p)))
    clients = []
    for _ in range(n):
        Ui, _ = np.linalg.qr(U + het * rng.standard_normal((d, p)) / np.sqrt(d - p))
        s = rng.standard_normal((m, p))
        Z = (s * np.sqrt(lam)) @ Ui.T + np.sqrt(noise / d) * rng.standard_normal((m, d))
        Z /= np.maximum(1.0, np.linalg.norm(Z, axis=1, keepdims=True))
        clients.append(Z)
    sigma_bar = sum(Z.T @ Z / len(Z) for Z in clients) / n
    ev, V = np.linalg.eigh(sigma_bar)
    f_star = -0.5 * float(ev[-p:].sum())
    return Federation(clients, d, p, sigma_bar, f_star, V[:, -p:] @ V[:, -p:].T)
