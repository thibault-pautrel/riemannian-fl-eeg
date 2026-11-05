"""Tests of the PriRFed Theorem-7 accountant (rfl/federated/privacy/prirfed_accountant.py).

Reference: Huang et al. (PriRFed), Machine Learning 115:84, 2026.

Theorem 7 (disjoint client datasets, our setting):
    eps' = min(T et, sqrt(2T log(1/dhat)) et + T et (e^et - 1)),   delta' = dhat + T dt,
    et = log(1 + rho (e^eps - 1)),   dt = rho delta,   rho = s/N.
Theorem 5 (general datasets) uses et = log(1 + rho (e^{s eps} - 1)) and dt = rho s delta,
so Theorem 7 is never looser.
"""
import math

import numpy as np
import pytest

from rfl.federated.privacy.accountant import (
    SubsampledGaussianAccountant, mu0_from_clip, subsampling_rate,
)
from rfl.federated.privacy.prirfed_accountant import (
    _amplify, calibrate_sigma_prirfed, compare_accountants, prirfed_theorem7_epsilon,
)

DELTA = 1e-5

# (name, C, B, m_star, m_max, T, local_epochs, N, client_fraction)
CELLS = [
    ("bnci_spdnet", 1.0, 24, 300, 340, 154, 1, 9, 1.0),
    ("cho_eegnet", 2.5, 32, 400, 460, 100, 1, 10, 0.5),
    ("weibo_spdnet", 0.6, 16, 200, 240, 200, 1, 4, 1.0),
]


def paper_theorem(acc, tau, T, target_delta, rho, s=None, theorem=7, n_dhat=4001):
    """eps' of Theorem 7 (or 5), transcribed from the statement, minimised over dhat."""
    def amplified(eps):
        k = 1 if theorem == 7 else s
        return math.log1p(rho * math.expm1(k * eps))

    def per_round_delta(dt):
        return dt / rho if theorem == 7 else dt / (rho * s)

    best, info = math.inf, None
    delta = per_round_delta(target_delta / T)          # basic branch, dhat = 0
    if 0 < delta < 1:
        try:
            et = amplified(acc.epsilon(delta, tau))
            if math.isfinite(et):
                best, info = T * et, "basic"
        except ValueError:
            pass
    for dhat in np.logspace(math.log10(target_delta) - 12,
                            math.log10(target_delta) - 1e-9, n_dhat):
        delta = per_round_delta((target_delta - dhat) / T)
        if not 0 < delta < 1:
            continue
        try:
            eps = acc.epsilon(delta, tau)
        except ValueError:
            continue
        if not math.isfinite(eps) or eps <= 0:
            continue
        et = amplified(eps)
        if not math.isfinite(et) or et <= 0:
            continue
        e = math.sqrt(2 * T * math.log(1 / dhat)) * et + T * et * math.expm1(et)
        if e < best:
            best, info = e, "advanced"
    return best, info


def _acc(C, B, m_star, sigma):
    return SubsampledGaussianAccountant(mu0_from_clip(C, B, sigma), subsampling_rate(B, m_star))


@pytest.mark.parametrize("cell", CELLS, ids=[c[0] for c in CELLS])
@pytest.mark.parametrize("sigma", [1.0, 4.0])
def test_theorem7_matches_transcription(cell, sigma):
    _, C, B, ms, mx, T, E, _, _ = cell
    tau = E * math.ceil(mx / B)
    acc = _acc(C, B, ms, sigma)
    got, info = prirfed_theorem7_epsilon(acc, tau, T, DELTA, rho=1.0)
    ref, ref_branch = paper_theorem(acc, tau, T, DELTA, rho=1.0)
    assert got == pytest.approx(ref, rel=5e-3)
    assert info["branch"] == ref_branch


@pytest.mark.parametrize("eps", [0.1, 1.0, 5.0])
@pytest.mark.parametrize("rho", [0.25, 0.5, 1.0])
def test_amplification_formula(eps, rho):
    assert _amplify(eps, rho) == pytest.approx(math.log(1 + rho * (math.exp(eps) - 1)),
                                               abs=1e-12)


def test_partial_participation_matches_transcription_and_thm7_beats_thm5():
    _, C, B, ms, mx, T, E, N, frac = CELLS[1]
    tau = E * math.ceil(mx / B)
    acc = _acc(C, B, ms, 2.0)
    s = math.ceil(N * frac)
    rho = s / N
    got, _ = prirfed_theorem7_epsilon(acc, tau, T, DELTA, rho=rho)
    ref7, _ = paper_theorem(acc, tau, T, DELTA, rho=rho, s=s, theorem=7)
    ref5, _ = paper_theorem(acc, tau, T, DELTA, rho=rho, s=s, theorem=5)
    assert got == pytest.approx(ref7, rel=5e-3)
    assert ref7 <= ref5 + 1e-9


@pytest.mark.parametrize("cell", CELLS, ids=[c[0] for c in CELLS])
def test_dhat_optimum_is_interior(cell):
    _, C, B, ms, mx, T, E, _, _ = cell
    tau = E * math.ceil(mx / B)
    _, info = prirfed_theorem7_epsilon(_acc(C, B, ms, 2.0), tau, T, DELTA, rho=1.0)
    if info["branch"] == "advanced":
        lo, hi = DELTA * 1e-6, DELTA
        pos = math.log10(info["dhat"] / lo) / math.log10(hi / lo)
        assert 0.02 < pos < 0.98, info["dhat"]


def test_input_validation():
    acc = _acc(1.0, 24, 300, 1.0)
    for kw in (dict(tau=0, T=1), dict(tau=1, T=0)):
        with pytest.raises(ValueError):
            prirfed_theorem7_epsilon(acc, target_delta=DELTA, **kw)
    with pytest.raises(ValueError):
        prirfed_theorem7_epsilon(acc, 1, 1, DELTA, rho=1.5)


def test_calibration_honours_target():
    sigma, realized, info = calibrate_sigma_prirfed(
        C=1.0, B=24, m_star=300, tau=15, T=10, target_eps=3.0, target_delta=DELTA)
    assert realized <= 3.0 + 1e-9
    assert info["eps_prime"] == pytest.approx(realized)


def test_no_overhead_at_one_round():
    """At T = 1 both accountants reduce to one C_q(G_mu0)^{otimes tau}."""
    cmp = compare_accountants(C=1.0, B=24, m_star=300, m_max=340, T=1, local_epochs=1,
                              target_eps=3.0, target_delta=DELTA, n_clients=9)
    assert cmp.sigma_ratio == pytest.approx(1.0, abs=0.02)
    assert cmp.rho == 1.0 and cmp.sigma_prirfed_rho is None


@pytest.mark.slow
def test_overhead_grows_with_rounds():
    prev = 0.0
    for T in (1, 10, 50, 154):
        cmp = compare_accountants(C=1.0, B=24, m_star=300, m_max=340, T=T,
                                  local_epochs=1, target_eps=3.0, target_delta=DELTA,
                                  n_clients=9)
        assert cmp.sigma_ratio >= 1.0 - 1e-6
        assert cmp.sigma_ratio >= prev - 1e-6
        prev = cmp.sigma_ratio


def test_overhead_is_independent_of_C():
    kw = dict(B=24, m_star=300, m_max=340, T=10, local_epochs=1, target_eps=3.0,
              target_delta=DELTA)
    a = compare_accountants(C=1.0, **kw)
    b = compare_accountants(C=3.0, **kw)
    assert b.sigma_ratio == pytest.approx(a.sigma_ratio, rel=1e-3)
