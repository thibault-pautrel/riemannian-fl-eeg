"""Tests of the certified f-DP accountant (rfl/federated/privacy/accountant.py).

The accountant must return an upper bound on the true privacy profile. Ground truth is
available in two places: the closed-form step profile, checked against a brute-force
conjugate, and the q = 1 composition, which is exactly G_{mu0 sqrt(n)}.
"""
import math

import numpy as np
import pytest
from scipy.stats import norm

from rfl.federated.privacy.accountant import (
    SubsampledGaussianAccountant,
    calibrate_sigma,
    gaussian_profile,
    mu0_from_clip,
    n_steps_conservative,
    step_profile,
    subsampling_rate,
)

DELTA = 1e-5


def brute_step_profile(eps, mu0, q, n_alpha=2_000_001):
    """delta(eps) = 1 - min_alpha (e^eps alpha + min{f_q, f_q^{-1}}(alpha)) on a dense grid."""
    a = np.linspace(1e-12, 1 - 1e-12, n_alpha)
    fq = q * norm.cdf(norm.ppf(1 - a) - mu0) + (1 - q) * (1 - a)
    finv = np.interp(a, fq[::-1], a[::-1])        # f_q is decreasing: invert by reflection
    h = np.minimum(fq, finv)
    return np.clip(np.array([1.0 - np.min(np.exp(e) * a + h) for e in eps]), 0, 1)


# --------------------------------------------------------------------------- #
#  Parameter helpers
# --------------------------------------------------------------------------- #
def test_parameter_helpers():
    assert mu0_from_clip(1.0, 32, 2.0) == pytest.approx(2.0 / 64.0)
    assert mu0_from_clip(1.0, 32, 0.0) == math.inf
    assert subsampling_rate(32, 128) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        subsampling_rate(129, 128)
    # tau = local_epochs * ceil(m_max / B)
    assert n_steps_conservative(T=10, local_epochs=2, m_max=100, B=32) == 10 * 2 * 4


# --------------------------------------------------------------------------- #
#  Per-step profile
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mu0,q", [(1.1141, 0.08), (0.3, 0.5), (2.5, 0.01),
                                   (0.05, 0.25), (4.0, 0.9)])
def test_step_profile_matches_brute_force(mu0, q):
    eps = np.array([0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0])
    err = np.max(np.abs(step_profile(eps, mu0, q) - brute_step_profile(eps, mu0, q)))
    assert err < 2e-6, err


@pytest.mark.parametrize("mu0", [0.1, 0.5, 1.1141, 3.0])
def test_step_profile_q1_is_gaussian(mu0):
    eps = np.array([0.0, 0.1, 0.5, 1.0, 3.0, 6.0])
    ref = np.array([gaussian_profile(e, mu0) for e in eps])
    assert np.max(np.abs(step_profile(eps, mu0, 1.0) - ref)) < 1e-12


def test_step_profile_degenerate_mu0():
    assert np.all(step_profile([0.0, 1.0], 0.0, 0.5) == 0.0)
    assert np.all(step_profile([0.0, 1.0], math.inf, 0.5) == 1.0)


@pytest.mark.parametrize("mu0,q,h", [(1.1141, 0.08, 0.01), (0.3, 0.5, 0.01),
                                     (2.5, 0.01, 0.005), (0.05, 0.25, 0.01)])
def test_step_pld_dominates_exact_profile(mu0, q, h):
    diag = SubsampledGaussianAccountant(mu0, q, h=h).certify()
    assert diag["min_gap"] >= -1e-11
    assert diag["total_mass"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
#  Composition
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mu0,n", [(0.15, 300), (0.30, 100), (0.80, 14)])
def test_q1_composition_upper_bounds_analytic_gaussian(mu0, n):
    """At q = 1 the composed truth is G_{mu0 sqrt(n)}. The accountant bounds it from
    above, and tightly."""
    acc = SubsampledGaussianAccountant(mu0, q=1.0, h=0.02)
    mu_eff = mu0 * math.sqrt(n)
    for e in np.linspace(0.3, 12.0, 60):
        gap = acc.delta(e, n) - gaussian_profile(e, mu_eff)
        assert -1e-12 <= gap < 1e-3


def test_degenerate_accountants():
    acc0 = SubsampledGaussianAccountant(0.0, 0.5)
    assert acc0.delta(1.0, 10) == 0.0 and acc0.epsilon(DELTA, 10) == 0.0
    acc_inf = SubsampledGaussianAccountant(math.inf, 0.5)
    assert acc_inf.delta(1.0, 10) == 1.0 and acc_inf.epsilon(DELTA, 10) == math.inf
    with pytest.raises(ValueError):
        SubsampledGaussianAccountant(-1.0, 0.5)


@pytest.mark.slow
@pytest.mark.parametrize("mu0,q,n", [(1.1141, 0.08, 200), (0.4, 0.2, 500)])
def test_fft_composition_matches_monte_carlo(mu0, q, n):
    """Sum n draws from the step PLD and compare delta(eps) with the FFT composition."""
    acc = SubsampledGaussianAccountant(mu0, q, h=0.01)
    step = acc._step()
    p = step.mass / step.mass.sum()
    ell = step.eps0 + np.arange(step.mass.size) * step.h
    rng = np.random.default_rng(0)
    N = 100_000
    L = np.concatenate([ell[rng.choice(ell.size, size=(10_000, n), p=p)].sum(axis=1)
                        for _ in range(N // 10_000)])
    pinf_n = 1.0 - (1.0 - step.p_inf) ** n
    for e in (1.0, 5.0):
        if acc.delta(e, n) <= 50 * acc.noise_floor(n):
            continue
        w = np.clip(1.0 - np.exp(e - L), 0.0, None)
        mc = (1 - pinf_n) * w.mean() + pinf_n
        se = (1 - pinf_n) * w.std(ddof=1) / math.sqrt(N)
        fft = acc.delta(e, n)
        assert abs(fft - mc) < 6 * se or abs(fft - mc) < 1e-9, (e, fft, mc, se)


@pytest.mark.slow
@pytest.mark.parametrize("mu0,q,n", [(1.1141, 0.08, 200), (0.15, 0.05, 2000)])
def test_epsilon_stable_under_h_refinement(mu0, q, n):
    es = [SubsampledGaussianAccountant(mu0, q, h=h).epsilon(DELTA, n)
          for h in (0.02, 0.01, 0.005, 0.0025)]
    assert max(es) - min(es) < 0.02 * max(es), es


def test_epsilon_monotone_in_sigma_and_n():
    C, B, m_star = 1.0, 24, 300
    q = subsampling_rate(B, m_star)
    eps_sigma = [SubsampledGaussianAccountant(mu0_from_clip(C, B, s), q).epsilon(DELTA, 1000)
                 for s in (0.5, 1.0, 2.0, 4.0)]
    assert all(a > b for a, b in zip(eps_sigma, eps_sigma[1:])), eps_sigma
    acc = SubsampledGaussianAccountant(mu0_from_clip(C, B, 1.0), q)
    eps_n = [acc.epsilon(DELTA, n) for n in (250, 500, 1000, 2000)]
    assert all(a < b for a, b in zip(eps_n, eps_n[1:])), eps_n


# --------------------------------------------------------------------------- #
#  Calibration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", [1.0, 3.0, 8.0])
def test_calibrate_sigma_honours_target(target):
    n = n_steps_conservative(T=50, local_epochs=1, m_max=340, B=24)
    sigma, realized = calibrate_sigma(C=1.0, B=24, m_star=300, n_steps=n,
                                      target_eps=target, target_delta=DELTA)
    assert realized <= target + 1e-9
    assert realized > 0.95 * target                   # not wastefully conservative
    acc = SubsampledGaussianAccountant.from_params(1.0, 24, 300, sigma)
    assert acc.epsilon(DELTA, n) == pytest.approx(realized)


def test_sigma_is_linear_in_C():
    kw = dict(B=32, m_star=800, n_steps=500, target_eps=3.0, target_delta=DELTA)
    s1, _ = calibrate_sigma(C=1.0, **kw)
    s2, _ = calibrate_sigma(C=2.5, **kw)
    assert s2 == pytest.approx(2.5 * s1, rel=1e-6)


def test_delta_below_noise_floor_is_refused():
    acc = SubsampledGaussianAccountant(mu0_from_clip(1.0, 24, 1.0),
                                       subsampling_rate(24, 1200))
    floor = acc.noise_floor(2000)
    if floor <= 0.0:
        pytest.skip("noise floor is exactly 0")
    try:
        eps = acc.epsilon(floor / 10.0, 2000)
    except ValueError:
        return
    assert math.isinf(eps)


@pytest.mark.parametrize("mu0,q,n", [(1.0, 0.1, 50), (2.0, 0.02, 200), (0.5, 0.05, 400)])
def test_reported_eps_is_never_optimistic(mu0, q, n):
    acc = SubsampledGaussianAccountant(mu0, q)
    for delta in (1e-5, 1e-4):
        eps = acc.epsilon(delta, n)
        assert acc.delta(eps, n) <= delta

