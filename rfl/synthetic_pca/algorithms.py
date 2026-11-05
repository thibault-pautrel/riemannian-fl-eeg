"""DP-RFedProj for federated PCA on Gr(d, p), with ProjAvg, RLAvg or RetAvg aggregation.

Federated PCA on the Grassmann manifold (as in Nguyen et al., 2024):

    min_{P in Gr(d, p)} F(P) = (1/n) sum_i (1/m) sum_l f(P; z_il),   f(P; z) = -1/2 z^T P z.

f equals the reconstruction error 1/2 ||z - P z||^2 up to the constant -1/2 ||z||^2.
Its ambient gradient is -1/2 z z^T and its Riemannian gradient at P is

    grad f(P; z) = P_P(-1/2 z z^T) = -1/2 (a b^T + b a^T),   a = P z,  b = z - P z,

of norm ||a|| ||b|| / sqrt 2 <= ||z||^2 / (2 sqrt 2).

Local steps (DP-RFedProj with RSGD): per-sample gradients clipped to C, averaged over a
minibatch of size B drawn without replacement, plus the tangent projection of
N(0, sigma^2 I) on S_d, then P <- Pi(P - eta g~). All aggregators share these local
steps, so they share the privacy guarantee of our accountant for a given sigma. The server
aggregates the final local iterates P_i of the k sampled clients:

    projavg  P <- Pi(mean_i P_i)
    rlavg    P <- Pi(P + mean_i P_P(P_i - P))
    retavg   P <- R_P(mean_i R_P^{-1}(P_i)), PriRFed's rule, with the projection retraction
             R_P(X) = Pi(P + X), whose inverse is one linear solve of size p (d - p)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from rfl.geometry import grassmann as gr
from rfl.synthetic_pca.data import Federation

AGGREGATORS = ("projavg", "rlavg", "retavg")


@dataclass
class Config:
    """One federated run."""
    aggregator: str = "projavg"     # 'projavg', 'rlavg' or 'retavg'
    T: int = 200                    # rounds
    k: int = 10                     # clients per round
    tau: int = 10                   # local steps per round
    B: int = 50                     # minibatch size
    alpha: float = 2.0              # per-round stepsize, constant; local stepsize eta = alpha / tau
    C: float = 0.5                  # clip constant
    sigma: float = 0.0              # noise std sigma_dp (0 = non-private)
    seed: int = 0                   # seed of the protocol (init, schedule, batches, noise)


@dataclass
class History:
    """Metrics of the global iterate after each round (index 0 = initial point)."""
    gap: List[float] = field(default_factory=list)      # F(P_t) - F*
    grad2: List[float] = field(default_factory=list)    # ||grad F(P_t)||^2
    sin2: List[float] = field(default_factory=list)     # ||sin Theta(P_t, P*)||_F^2 = p - <P_t, P*>
    clipped: int = 0                                    # per-sample gradients clipped
    samples: int = 0                                    # per-sample gradients computed

    @property
    def clip_fraction(self) -> float:
        """Fraction of the per-sample gradients that clipping shrank."""
        return self.clipped / max(self.samples, 1)


def metrics(fed: Federation, P: np.ndarray) -> tuple:
    """Optimality gap, squared Riemannian gradient norm and principal-angle distance."""
    gap = -0.5 * float(np.sum(P * fed.sigma_bar)) - fed.f_star
    g = gr.tangent_proj(P, -0.5 * fed.sigma_bar)
    sin2 = fed.p - float(np.sum(P * fed.P_star))
    return gap, float(np.sum(g * g)), sin2


def _clipped_grad(Z: np.ndarray, P: np.ndarray, C: float, hist: "History | None" = None
                  ) -> np.ndarray:
    """Average over the rows z of Z of clip_C(grad f(P; z)). Counts the clipped ones in `hist`."""
    A = Z @ P                       # rows a = P z (P is symmetric)
    Bm = Z - A                      # rows b = z - P z
    norms = np.linalg.norm(A, axis=1) * np.linalg.norm(Bm, axis=1) / np.sqrt(2.0)
    c = np.minimum(1.0, C / np.maximum(norms, 1e-300))
    if hist is not None:
        hist.clipped += int(np.sum(norms > C * (1 + 1e-12)))   # ignore rounding at the bound
        hist.samples += len(Z)
    M = A.T @ (c[:, None] * Bm)
    return -(M + M.T) / (2.0 * len(Z))


def aggregate(P: np.ndarray, local: List[np.ndarray], p: int, aggregator: str) -> np.ndarray:
    """Server update from the current iterate P and the final local iterates."""
    if aggregator == "projavg":
        return gr.proj(sum(local) / len(local), p)
    if aggregator == "rlavg":
        return gr.proj(P + gr.tangent_proj(P, sum(local) / len(local) - P), p)
    if aggregator == "retavg":
        X = sum(gr.inverse_retraction(P, Q, p) for Q in local) / len(local)
        return gr.retraction(P, X, p)
    raise ValueError(aggregator)


def run(fed: Federation, cfg: Config, P0: np.ndarray | None = None) -> History:
    """Run DP-RFedProj and record the metrics of the global iterate after every round.

    Args:
        fed: the federation.
        cfg: the run configuration.
        P0: initial point of Gr(d, p). Drawn from `cfg.seed` if None.
    """
    if cfg.aggregator not in AGGREGATORS:
        raise ValueError(cfg.aggregator)
    # Two streams, both distinct from the data's: the protocol (initial point, sampled
    # clients, minibatches) and the DP noise. With sigma = 0 the noise stream is unused,
    # so the run without noise shares the initial point, clients and minibatches.
    rng = np.random.default_rng((2, cfg.seed))
    noise_rng = np.random.default_rng((4, cfg.seed))
    d, p = fed.d, fed.p
    eta = cfg.alpha / cfg.tau
    X = gr.random_point(d, p, rng) if P0 is None else P0.copy()
    hist = History()

    def record() -> None:
        g, n2, s2 = metrics(fed, X)
        hist.gap.append(g)
        hist.grad2.append(n2)
        hist.sin2.append(s2)

    record()
    for _ in range(cfg.T):
        S = rng.choice(fed.n, size=cfg.k, replace=False)
        local = []
        for i in S:
            Zi = fed.clients[i]
            Y = X.copy()
            for _ in range(cfg.tau):
                batch = Zi[rng.choice(len(Zi), size=cfg.B, replace=False)]
                g = _clipped_grad(batch, Y, cfg.C, hist)
                if cfg.sigma > 0:
                    g = g + gr.tangent_proj(Y, gr.sym_gaussian(d, cfg.sigma, noise_rng))
                Y = gr.proj(Y - eta * g, p)
            local.append(Y)
        X = aggregate(X, local, p, cfg.aggregator)
        record()
    return hist


def dp_term(alpha: float, sigma: float, d_M: int, k: int, tau: int) -> float:
    """DP-noise term of the convergence bound, eta sigma^2 d_M / k = alpha sigma^2 d_M / (k tau).

    Up to a constant: eta = alpha / tau is the local stepsize.

    To first order a round moves the iterate by -alpha grad F plus the average of the
    k tau noise draws of the round, of variance alpha^2 sigma^2 / (k tau) along each of
    the d_M tangent directions. The cost of privacy should scale like this term.
    """
    return alpha * sigma ** 2 * d_M / (k * tau)
