"""PriRFed privacy accounting (Huang et al., 2026, Theorem 7), as a comparison baseline.

Both DP-RFedProj and PriRFed are run with the same mechanism (C, B, q, tau, T). Only the
outer accounting differs.

  * DP-RFedProj composes all T * tau local steps as one tensor power
    C_q(G_mu0)^{otimes (T tau)} and converts it to (eps, delta) once, at the end
    (`federated.privacy.accountant`).
  * PriRFed treats each client's local run in a round as one (eps_r, delta_r)-DP
    output and composes the T rounds with advanced composition.

The per-round inner object is our exact C_q(G_mu0)^{otimes tau}. This is a best case for
PriRFed: its published analysis uses a looser closed-form Gaussian bound inside each
round. The measured gap is therefore due to the outer wrapper alone. Report it as
"PriRFed accounting (Thm 7), best-cased with our per-round accountant".

At client sampling rate rho = s/N, Theorem 7 gives eps_t = log(1 + rho (e^{eps_r} - 1)),
delta_t = rho * delta_r, and for any split delta' = dhat + T * delta_t,

    eps' = min(T * eps_t,  sqrt(2 T log(1/dhat)) * eps_t + T * eps_t (e^{eps_t} - 1)).

The split is optimised to give PriRFed its tightest epsilon. Amplification by client
sampling (rho < 1) is only sound when participation is hidden from the adversary. Under
an honest-but-curious server that observes participation, rho = 1 is the comparison to
report. rho < 1 is provided for the paper-native accounting only.

Consistency check: at T = 1 the basic branch gives eps' = epsilon(target_delta, tau),
which is the DP-RFedProj budget. So sigma_PriRFed = sigma_ours at T = 1, and the ratio
grows with T.

Reference: Huang et al. (PriRFed), Machine Learning 115:84, 2026, Theorem 7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from rfl.federated.privacy.accountant import (
    SubsampledGaussianAccountant, mu0_from_clip, subsampling_rate,
)

__all__ = [
    "prirfed_theorem7_epsilon",
    "calibrate_sigma_prirfed",
    "compare_accountants",
    "PriRFedBudget",
]


# --------------------------------------------------------------------------- #
#  Theorem-7 outer wrapper
# --------------------------------------------------------------------------- #
def _amplify(eps_r: float, rho: float) -> float:
    """Client-sampling amplification eps_t = log(1 + rho (e^{eps_r} - 1)). Identity at rho = 1.

    Only sound when participation is hidden from the adversary (see the module docstring).
    """
    if rho >= 1.0:
        return eps_r
    if not math.isfinite(eps_r):
        return math.inf
    return math.log1p(rho * math.expm1(eps_r))


def prirfed_theorem7_epsilon(
    acc: SubsampledGaussianAccountant,
    tau: int,
    T: int,
    target_delta: float,
    *,
    rho: float = 1.0,
    n_dhat: int = 96,
) -> Tuple[float, Dict[str, float]]:
    """PriRFed Theorem-7 epsilon of a mechanism over T rounds of tau local steps.

    The per-round budget (eps_r, delta_r) is read off `acc` at n_steps = tau, then
    amplified by client sampling and composed over T rounds (module docstring). The
    basic branch (dhat = 0) and the advanced branch on a log grid of dhat are both
    evaluated, and the smallest eps' is kept.

    Args:
        acc: accountant carrying the mechanism (mu0, q).
        tau: local steps per round (inner composition count).
        T: communication rounds (outer composition count).
        target_delta: overall delta' budget.
        rho: client sampling rate s/N in (0, 1]. 1.0 is full participation.
        n_dhat: number of dhat values tried in the advanced branch.

    Returns:
        (eps_prime, info). `info` holds the tight branch ('basic' or 'advanced'),
        dhat, rho, delta_r, delta_tilde, eps_r, eps_tilde and eps_prime.

    Raises:
        ValueError: if T or tau < 1, or target_delta or rho is out of range.
    """
    if T < 1 or tau < 1:
        raise ValueError("T and tau must be >= 1")
    if not (0.0 < target_delta < 1.0):
        raise ValueError("target_delta must be in (0, 1)")
    if not (0.0 < rho <= 1.0):
        raise ValueError(f"rho = s/N must be in (0, 1]; got {rho}")

    # delta' = dhat + T * rho * delta_r, so delta_r = (target - dhat) / (T * rho).

    # Basic branch: dhat = 0.
    dr_basic = target_delta / (T * rho)
    er_basic = et_basic = eps_basic = math.inf
    if 0.0 < dr_basic < 1.0:
        try:
            er_basic = acc.epsilon(dr_basic, tau)
            et_basic = _amplify(er_basic, rho)
            eps_basic = T * et_basic if math.isfinite(et_basic) else math.inf
        except ValueError:
            pass        # delta_r below the accountant's noise floor: unusable split

    best_eps = eps_basic
    best = {"branch": "basic", "dhat": 0.0, "rho": float(rho),
            "delta_r": float(dr_basic), "delta_tilde": float(rho * dr_basic),
            "eps_r": float(er_basic), "eps_tilde": float(et_basic)}

    # Advanced-composition branch: optimise dhat in (0, target_delta).
    for dhat in np.logspace(np.log10(target_delta) - 6.0,
                            np.log10(target_delta) - 1e-6, n_dhat):
        delta_r = (target_delta - dhat) / (T * rho)
        if not (0.0 < delta_r < 1.0):
            continue
        try:
            er = acc.epsilon(delta_r, tau)
        except ValueError:
            continue    # delta_r below the noise floor as dhat -> target: skip it
        if not math.isfinite(er) or er <= 0.0:
            continue
        et = _amplify(er, rho)
        if not math.isfinite(et) or et <= 0.0:
            continue
        eps_adv = (math.sqrt(2.0 * T * math.log(1.0 / dhat)) * et
                   + T * et * (math.exp(et) - 1.0))
        if eps_adv < best_eps:
            best_eps = eps_adv
            best = {"branch": "advanced", "dhat": float(dhat), "rho": float(rho),
                    "delta_r": float(delta_r), "delta_tilde": float(rho * delta_r),
                    "eps_r": float(er), "eps_tilde": float(et)}

    best["eps_prime"] = float(best_eps)
    return float(best_eps), best


# --------------------------------------------------------------------------- #
#  sigma calibration under Theorem 7
# --------------------------------------------------------------------------- #
def calibrate_sigma_prirfed(
    *,
    C: float, B: int, m_star: int, tau: int, T: int,
    target_eps: float, target_delta: float,
    rho: float = 1.0,
    sigma_lo: float = 1e-3, sigma_hi: float = 1e4,
    sigma_hi_cap: float = 1e9,
    tol: float = 1e-3, n_dhat: int = 96,
    **acc_kw,
) -> Tuple[float, float, Dict[str, float]]:
    """Smallest sigma meeting (target_eps, target_delta) under PriRFed Theorem 7.

    Same mechanism as `accountant.calibrate_sigma`, different outer accounting. At large
    T the required noise can be orders of magnitude larger, so `sigma_hi` is multiplied
    by 4 until the target is bracketed, up to `sigma_hi_cap`. sigma is then nudged up
    until the target is honoured.

    Args:
        C: clip constant.
        B: minibatch size.
        m_star: smallest client training-set size.
        tau: local steps per round.
        T: communication rounds.
        target_eps: target epsilon.
        target_delta: target delta.
        rho: client sampling rate s/N.
        sigma_lo: lower end of the search range.
        sigma_hi: initial upper end of the search range.
        sigma_hi_cap: largest allowed upper end.
        tol: absolute tolerance of the root finder on sigma.
        n_dhat: dhat grid size, see `prirfed_theorem7_epsilon`.
        **acc_kw: forwarded to `SubsampledGaussianAccountant` (in practice `h`).

    Returns:
        (sigma, realized_eps, info), with `info` from `prirfed_theorem7_epsilon`.

    Raises:
        ValueError: if the target is infeasible at `sigma_hi_cap`.
    """
    q = subsampling_rate(B, m_star)

    def eps_at(sigma: float) -> float:
        acc = SubsampledGaussianAccountant(mu0_from_clip(C, B, sigma), q, **acc_kw)
        e, _ = prirfed_theorem7_epsilon(acc, tau, T, target_delta,
                                        rho=rho, n_dhat=n_dhat)
        return e

    # Raise the ceiling until the target is bracketed.
    e_hi = eps_at(sigma_hi)
    while e_hi > target_eps and sigma_hi < sigma_hi_cap:
        sigma_hi = min(sigma_hi * 4.0, sigma_hi_cap)
        e_hi = eps_at(sigma_hi)
    if e_hi > target_eps:
        raise ValueError(
            f"target_eps={target_eps} infeasible even at sigma_hi={sigma_hi:.3g} "
            f"(eps={e_hi:.3f}); PriRFed Theorem-7 round-composition is astronomically "
            f"loose at T={T}, rho={rho} (raise sigma_hi_cap if you really want this point).")
    e_lo = eps_at(sigma_lo)
    if e_lo <= target_eps:
        info = prirfed_theorem7_epsilon(
            SubsampledGaussianAccountant(mu0_from_clip(C, B, sigma_lo), q, **acc_kw),
            tau, T, target_delta, rho=rho, n_dhat=n_dhat)[1]
        return sigma_lo, e_lo, info

    sigma = float(brentq(lambda s: eps_at(s) - target_eps,
                         sigma_lo, sigma_hi, xtol=tol))
    # Round sigma up until the target is honoured (same guard as calibrate_sigma).
    realized = eps_at(sigma)
    guard = 0
    while realized > target_eps and guard < 64:
        sigma *= 1.0 + 4.0 * tol
        realized = eps_at(sigma)
        guard += 1
    if realized > target_eps:
        raise RuntimeError(f"PriRFed calibration did not reach target_eps={target_eps} "
                           f"(eps={realized:.6g} at sigma={sigma:.6g})")
    info = prirfed_theorem7_epsilon(
        SubsampledGaussianAccountant(mu0_from_clip(C, B, sigma), q, **acc_kw),
        tau, T, target_delta, rho=rho, n_dhat=n_dhat)[1]
    return sigma, realized, info


# --------------------------------------------------------------------------- #
#  Side-by-side comparison at the same target
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PriRFedBudget:
    """Noise calibrated by both accountants for one mechanism and one target budget.

    sigma_B (`sigma_ours`) is the DP-RFedProj noise and sigma_A (`sigma_prirfed`) the
    PriRFed Theorem-7 noise at rho = 1. Their ratio is the accounting overhead. The
    `*_rho` fields repeat the PriRFed calibration at rho = s/N with client-sampling
    amplification. They are None when rho = 1.
    """
    # Mechanism and target.
    C: float
    B: int
    m_star: int
    tau: int
    T: int
    n_steps: int            # T * tau
    q: float
    target_eps: float
    target_delta: float
    # DP-RFedProj: one T*tau-fold composition.
    sigma_ours: float       # sigma_B
    mu0_ours: float
    realized_eps_ours: float
    # PriRFed Theorem 7 at rho = 1 (sound under observable participation).
    sigma_prirfed: float    # sigma_A
    mu0_prirfed: float
    realized_eps_prirfed: float
    prirfed_branch: str     # 'basic' or 'advanced': which Thm-7 term was tight
    sigma_ratio: float      # sigma_A / sigma_B, >= 1 and = 1 at T = 1
    # Client participation.
    n_clients: int          # N
    s_clients: int          # s = ceil(N * client_fraction)
    rho: float              # s / N
    # PriRFed Theorem 7 at rho = s/N, amplified (sound only if participation is hidden).
    sigma_prirfed_rho: Optional[float]
    realized_eps_prirfed_rho: Optional[float]
    prirfed_branch_rho: Optional[str]
    sigma_ratio_rho: Optional[float]          # sigma_A(rho) / sigma_B


def compare_accountants(
    *,
    C: float, B: int, m_star: int, m_max: int,
    T: int, local_epochs: int,
    target_eps: float, target_delta: float,
    n_clients: int = 1, client_fraction: float = 1.0,
    h: float = 0.01, n_dhat: int = 96,
) -> PriRFedBudget:
    """Calibrate sigma under both accountants for the same mechanism and target.

    tau = local_epochs * ceil(m_max / B) and q = B / m_star, as in
    `orchestration.setup_dp`. Both sigmas are linear in C, so the ratio
    sigma_A / sigma_B does not depend on C.

    Args:
        C: clip constant.
        B: minibatch size.
        m_star: smallest client training-set size.
        m_max: largest client training-set size.
        T: communication rounds.
        local_epochs: local epochs per round.
        target_eps: target epsilon.
        target_delta: target delta.
        n_clients: number of clients N.
        client_fraction: fraction of clients sampled per round. s = ceil(N * fraction).
        h: accountant lattice spacing.
        n_dhat: dhat grid size for Theorem 7.

    Returns:
        A `PriRFedBudget`. If the amplified rho < 1 calibration is infeasible, its
        fields are set to inf and the branch to 'infeasible'.
    """
    from rfl.federated.privacy.accountant import (
        calibrate_sigma, n_steps_conservative,
    )

    tau = local_epochs * int(math.ceil(m_max / float(B)))
    n_steps = n_steps_conservative(T, local_epochs, m_max, B)   # == T * tau
    q = subsampling_rate(B, m_star)

    # DP-RFedProj (sigma_B).
    sigma_b, eps_b = calibrate_sigma(
        C=C, B=B, m_star=m_star, n_steps=n_steps,
        target_eps=target_eps, target_delta=target_delta, h=h)
    mu0_b = mu0_from_clip(C, B, sigma_b)

    # PriRFed at rho = 1 (sigma_A).
    sigma_a, eps_a, info = calibrate_sigma_prirfed(
        C=C, B=B, m_star=m_star, tau=tau, T=T,
        target_eps=target_eps, target_delta=target_delta,
        rho=1.0, h=h, n_dhat=n_dhat)
    mu0_a = mu0_from_clip(C, B, sigma_a)

    # rho = s/N, with s = ceil(N * fraction) as in the server's client sampling.
    N = max(1, int(n_clients))
    s_sel = max(1, int(math.ceil(N * float(client_fraction))))
    rho = s_sel / float(N)

    # PriRFed at rho < 1, with amplification. Skipped at rho = 1 (same as above).
    sigma_a_rho = eps_a_rho = branch_rho = ratio_rho = None
    if rho < 1.0:
        try:
            sigma_a_rho, eps_a_rho, info_rho = calibrate_sigma_prirfed(
                C=C, B=B, m_star=m_star, tau=tau, T=T,
                target_eps=target_eps, target_delta=target_delta,
                rho=rho, h=h, n_dhat=n_dhat)
            branch_rho = info_rho["branch"]
            ratio_rho = (sigma_a_rho / sigma_b) if sigma_b > 0 else float("inf")
        except ValueError:
            # Infeasible even at sigma_hi_cap.
            sigma_a_rho = float("inf"); eps_a_rho = float("inf")
            branch_rho = "infeasible"; ratio_rho = float("inf")

    return PriRFedBudget(
        C=C, B=B, m_star=m_star, tau=tau, T=T, n_steps=n_steps, q=q,
        target_eps=target_eps, target_delta=target_delta,
        sigma_ours=sigma_b, mu0_ours=mu0_b, realized_eps_ours=eps_b,
        sigma_prirfed=sigma_a, mu0_prirfed=mu0_a, realized_eps_prirfed=eps_a,
        prirfed_branch=info["branch"],
        sigma_ratio=(sigma_a / sigma_b if sigma_b > 0 else float("inf")),
        n_clients=N, s_clients=s_sel, rho=rho,
        sigma_prirfed_rho=sigma_a_rho, realized_eps_prirfed_rho=eps_a_rho,
        prirfed_branch_rho=branch_rho, sigma_ratio_rho=ratio_rho,
    )