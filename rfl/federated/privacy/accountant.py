"""Certified f-DP / GDP accountant for DP-RFedProj.

Record-level guarantee under substitution adjacency, with minibatch subsampling without
replacement (WOR). The transcript of T rounds of tau local steps is

    C_q(G_mu0)^{otimes n}-DP,   n = T * tau,

    mu0 = 2C / (B * sigma)     per-step GDP parameter (noise on the averaged gradient)
    q   = B / m_star           record-level subsampling rate
    C_q(f) = min{f_q, f_q^{-1}}**,   f_q = q * f + (1 - q) * G_0   (Dong et al., 2022)

The trade-off function is converted to (eps, delta) with the f-DP duality
delta(eps) = 1 + f*(-e^eps). The budget depends only on (C, B, m_star, sigma, T, tau).
The geometry, the aggregation rule and the optimiser are post-processing and never enter.
The module uses numpy and scipy only.

Why the reported eps is a certified upper bound
-----------------------------------------------
Per-step profile in closed form. The convex hull ** can be skipped because h and h**
have the same convex conjugate. With h = min{f_q, f_q^{-1}}, x = e^eps and
m(c) = inf_{alpha in [0, 1]} (c * alpha + f_q(alpha)),

    delta(eps) = 1 - min{m(x), x * m(1/x)}.                                    (*)

The second branch follows from the substitution beta = f_q^{-1}(alpha). The derivative

    f_q'(alpha) = -q * exp(mu0 * Phi^{-1}(1 - alpha) - mu0^2 / 2) - (1 - q)

is strictly increasing on (0, 1) with range (-inf, -(1 - q)). So m(c) = c when
c <= 1 - q. Otherwise the unique stationary point is u* = (log((c - 1 + q) / q)
+ mu0^2 / 2) / mu0, alpha* = Phi(-u*). Formula (*) then only needs Phi, Phi^{-1}, log and
exp, and is exact to machine precision.

Certified composition. Composition uses a discrete privacy-loss distribution (PLD) that
dominates the true per-step mechanism. By the dominating-pair composition theorem its
n-fold self-convolution dominates the n-fold composition. Three facts give domination.

  (a) delta is convex in x = e^eps. m is concave (an infimum of affine maps) and
      x * m(1/x) is its perspective, so delta = 1 - min of concave maps is convex.
  (b) The PLD matches the exact profile at the lattice points and is affine in x between
      them. An affine interpolant of a convex function lies above it.
  (c) Both tails are rounded up. Mass above the lattice goes to p_inf (infinite loss) and
      mass below it is placed at the lowest lattice point.

`SubsampledGaussianAccountant.certify` checks (b) numerically on a dense probe grid.
`SubsampledGaussianAccountant.verify` checks the q = 1 case against the analytic identity
G_mu0^{otimes n} = G_{mu0 sqrt(n)}.

Reference: J. Dong, A. Roth, W. J. Su, "Gaussian differential privacy", JRSS-B, 2022.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

__all__ = [
    "mu0_from_clip",
    "subsampling_rate",
    "n_steps_conservative",
    "gaussian_profile",
    "step_profile",
    "SubsampledGaussianAccountant",
    "calibrate_sigma",
]

_PHI = norm.cdf
_PHI_INV = norm.ppf
_ISF = norm.isf      # Phi^{-1}(1 - alpha), accurate for tiny alpha

# Widest per-step privacy-loss lattice. The adaptive lattice is a sub-range of it.
# Tails are rounded up, so a narrower lattice is looser but still sound.
_EPS_LO_MIN = -30.0
_EPS_HI_MAX = 60.0


# --------------------------------------------------------------------------- #
#  Mechanism parameters
# --------------------------------------------------------------------------- #
def mu0_from_clip(C: float, B: int, sigma: float) -> float:
    """Per-step GDP parameter mu0 = Delta / sigma = 2C / (B * sigma).

    Delta = 2C/B is the substitution sensitivity of the averaged clipped gradient, and
    sigma is the std of the Gaussian noise added to that average. Both C and sigma are
    the values applied by the mechanism.

    Args:
        C: clip constant.
        B: minibatch size.
        sigma: noise std. A non-positive value means no noise.

    Returns:
        mu0, or +inf when sigma <= 0.
    """
    if sigma <= 0:
        return float("inf")
    return 2.0 * C / (B * sigma)


def subsampling_rate(B: int, m_star: int) -> float:
    """Record-level WOR subsampling rate q = B / m_star.

    Args:
        B: minibatch size.
        m_star: training-set size of the smallest client.

    Raises:
        ValueError: if q is not in (0, 1], i.e. B > m_star.
    """
    q = B / float(m_star)
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q = B/m_star = {q} must be in (0, 1]; need B <= m_star.")
    return q


def n_steps_conservative(T: int, local_epochs: int, m_max: int, B: int) -> int:
    """Composition count n = T * tau, with tau = local_epochs * ceil(m_max / B).

    tau is taken at the largest client (most local steps), while q = B / m_star is
    taken at the smallest one (largest rate). Composition is monotone in n and C_q is
    monotone in q, so the pair (max q, max n) bounds every client of a heterogeneous
    cohort.
    """
    steps_per_epoch = int(math.ceil(m_max / float(B)))
    return int(T * local_epochs * steps_per_epoch)


# --------------------------------------------------------------------------- #
#  Reference: analytic Gaussian profile (Dong et al. 2022, Cor. 1)
# --------------------------------------------------------------------------- #
def gaussian_profile(eps: float, mu: float) -> float:
    """delta(eps) of a mu-GDP mechanism: Phi(mu/2 - eps/mu) - e^eps Phi(-mu/2 - eps/mu)."""
    if mu <= 0:
        return 0.0
    return float(_PHI(mu / 2 - eps / mu) - np.exp(eps) * _PHI(-mu / 2 - eps / mu))


# --------------------------------------------------------------------------- #
#  Closed-form per-step privacy profile of C_q(G_mu0)
# --------------------------------------------------------------------------- #


def _u_star(c: np.ndarray, mu0: float, q: float) -> np.ndarray:
    """Stationary point u* = (log((c - 1 + q) / q) + mu0^2 / 2) / mu0 of c*alpha + f_q(alpha).

    Expressed in the chart u = Phi^{-1}(1 - alpha). Only valid for c > 1 - q, which the
    caller ensures. For q = 1 and c below about 1e-16, c - 1 + q rounds to 0, the log
    is -inf and so is u*. The profile formulas then return their correct limits, so
    the divide-by-zero warning is silenced.
    """
    with np.errstate(divide="ignore"):
        return (np.log((c - 1.0 + q) / q) + 0.5 * mu0 * mu0) / mu0


def step_profile(eps: "np.ndarray | float", mu0: float, q: float) -> np.ndarray:
    """Exact privacy profile delta(eps) of one subsampled-Gaussian step C_q(G_mu0).

    With x = e^eps, delta(eps) = max{delta_a(x), delta_b(x)} where

        delta_a(x) = q * Phi(mu0 - u) - (x - 1 + q) * Phi(-u),       u = u*(x),
        delta_b(x) = (1 - x(1 - q)) * Phi(v) - x * q * Phi(v - mu0),  v = u*(1/x),

    delta_a(x) = 1 - x for x <= 1 - q, and delta_b(x) = 0 for x >= 1 / (1 - q). Both
    branches are written in terms of u* rather than alpha*, so the leading 1 cancels
    analytically. Computing 1 - m(x) numerically would lose every digit once delta is
    below about 1e-12, which is where the composition tail lives. At q = 1 the formula
    reduces to the Gaussian profile of `gaussian_profile`.

    Args:
        eps: scalar or array of privacy-loss levels.
        mu0: per-step GDP parameter.
        q: subsampling rate in (0, 1].

    Returns:
        Array of delta values, same shape as `np.atleast_1d(eps)`.
    """
    eps = np.atleast_1d(np.asarray(eps, dtype=float))
    if mu0 <= 0.0:
        return np.zeros_like(eps)
    if not math.isfinite(mu0):
        return np.ones_like(eps)
    x = np.exp(eps)

    # ---- branch a:  1 - m(x) ------------------------------------------------
    d_a = np.where(x <= 1.0 - q, 1.0 - x, 0.0)
    hi = x > 1.0 - q
    if np.any(hi):
        xh = x[hi]
        u = _u_star(xh, mu0, q)
        d_a[hi] = q * _PHI(mu0 - u) - (xh - 1.0 + q) * _PHI(-u)

    # ---- branch b:  1 - x*m(1/x) -------------------------------------------
    d_b = np.zeros_like(x)
    inv = 1.0 / x
    lo = inv > 1.0 - q                      # else x*m(1/x) = 1 exactly -> d_b = 0
    if np.any(lo):
        xl = x[lo]
        v = _u_star(inv[lo], mu0, q)
        d_b[lo] = (1.0 - xl * (1.0 - q)) * _PHI(v) - xl * q * _PHI(v - mu0)

    return np.clip(np.maximum(d_a, d_b), 0.0, 1.0)


# --------------------------------------------------------------------------- #
#  Adaptive per-step grid bounds
# --------------------------------------------------------------------------- #
def _adaptive_grid_bounds(mu0: float, q: float, h: float, *, tail: float = 1e-16,
                          rel_margin: float = 1.25, pad_steps: int = 20
                          ) -> Tuple[float, float]:
    """Lattice range [eps_lo, eps_hi] covering the essential support of the step loss.

    The Gaussian privacy loss is mu0 * z + mu0^2 / 2 with z ~ N(0, 1). Cutting at
    sqrt(2 log(1/tail)) standard deviations bounds its support, and subsampling only
    shrinks the loss toward 0. The margin is relative to the half-width (plus a few
    lattice steps) so the lattice length stays O(1) in mu0. Both tails are rounded up
    anyway, so a narrow grid costs tightness, never soundness.
    """
    if not math.isfinite(mu0) or mu0 <= 0.0:
        return _EPS_LO_MIN, _EPS_HI_MAX
    Z = math.sqrt(2.0 * math.log(1.0 / tail))
    half = (0.5 * mu0 * mu0 + mu0 * Z) * rel_margin + pad_steps * h
    lo = max(-half, _EPS_LO_MIN)
    hi = min(half, _EPS_HI_MAX)
    return float(lo), float(hi)


def _effective_h(mu0: float, h: float) -> float:
    """Lattice spacing actually used: min(h, mu0 / 40), floored at 1e-4.

    Any h gives a valid bound. The slack is O(h) per step and accumulates over the
    composition, so h must be small relative to mu0, the scale of the loss. At
    mu0 = 0.15 a fixed h = 0.01 reports eps = 1.97 against a true 1.38.
    """
    if not math.isfinite(mu0) or mu0 <= 0.0:
        return h
    return float(min(h, max(mu0 / 40.0, 1e-4)))


# --------------------------------------------------------------------------- #
#  Discrete privacy-loss distribution + exact FFT composition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _PLD:
    """Discrete privacy-loss distribution (P-side) on a uniform eps-lattice."""
    eps0: float          # smallest privacy-loss grid value (a multiple of h)
    h: float             # grid spacing
    mass: np.ndarray     # finite-part masses at eps0, eps0+h, ...
    p_inf: float         # mass at +infinity (P-support where Q has none)
    noise: float = 0.0   # est. spurious mass from FFT round-off (0 for the step PLD)

    def delta(self, eps: float) -> float:
        """delta(eps) = E_P[(1 - e^{eps - ell})_+] + p_inf."""
        start = int(math.ceil((eps - self.eps0) / self.h))
        start = max(start, 0)
        if start >= self.mass.size:
            return float(self.p_inf)
        ell = self.eps0 + np.arange(start, self.mass.size) * self.h
        return float(np.dot(self.mass[start:], 1.0 - np.exp(eps - ell)) + self.p_inf)

    @property
    def total(self) -> float:
        return float(self.mass.sum() + self.p_inf)


def _build_step_pld(mu0: float, q: float, eps_lo: float, eps_hi: float,
                    h: float) -> _PLD:
    """Build a step PLD whose privacy profile dominates C_q(G_mu0) everywhere.

    The lattice is {k * h}, anchored at the origin, so composition never accumulates a
    sub-h offset. With x = e^eps, the profile of a discrete PLD is piecewise linear in x
    with knots x_j. Matching the exact profile delta_i at every knot gives the masses as
    a local second difference,

        m_i = [(delta_{i+1} - delta_i) - e^h (delta_i - delta_{i-1})] / (e^h - 1),

    for i = 1..K-1, with delta_K := delta_{K-1}. The round-off is about eps_machine / h
    per entry and does not accumulate. m_i >= 0 is the convexity of delta in x, so a
    negative entry signals a numerical fault. Tails are rounded up: p_inf = delta(eps_hi)
    and the residual mass is placed at eps_lo. A final guard raises p_inf by any
    floating-point shortfall at the knots.

    Raises:
        ValueError: if the lattice has fewer than 3 points.
        AssertionError: if domination cannot be certified numerically.
    """
    n_lo = int(math.floor(eps_lo / h + 1e-9))
    n_hi = int(math.ceil(eps_hi / h - 1e-9))
    grid = np.arange(n_lo, n_hi + 1) * h
    K = grid.size
    if K < 3:
        raise ValueError("privacy-loss lattice needs at least 3 points; lower h.")
    d = step_profile(grid, mu0, q)                       # exact, closed form
    d = np.minimum.accumulate(d)                         # delta is non-increasing
    p_inf = float(max(d[-1], 0.0))

    eh = math.exp(h)
    dd = np.diff(np.concatenate([d, d[-1:]]))            # dd[i] = d[i+1] - d[i]
    m = np.zeros(K)
    # m[i] = ( dd[i] - e^h * dd[i-1] ) / (e^h - 1),  i = 1..K-1
    m[1:] = (dd[1:K] - eh * dd[0:K - 1]) / (eh - 1.0)

    neg = float(-m[m < 0].sum()) if np.any(m < 0) else 0.0
    if neg > 1e-8:
        raise AssertionError(
            f"negative PLD mass {neg:.3e} (mu0={mu0}, q={q}, h={h}): delta is not "
            "convex in e^eps on this lattice -- refuse to certify.")
    m = np.clip(m, 0.0, None)

    # The residual mass (the losses below eps_lo, plus round-off) goes to +inf, which
    # only raises delta: the PLD stays dominating, and E_P[e^{-L}] <= 1 still holds.
    m[0] = 0.0
    resid = 1.0 - p_inf - float(m.sum())
    if resid < -1e-8:
        raise AssertionError(
            f"PLD mass overflow ({resid:.3e}); the lattice [{eps_lo}, {eps_hi}] is "
            "too narrow at the top.")
    p_inf = p_inf + max(resid, 0.0)

    pld = _PLD(eps0=float(grid[0]), h=h, mass=m, p_inf=p_inf)

    # Floating-point domination guard at the knots.
    got = np.array([pld.delta(float(e)) for e in grid])
    shortfall = float(np.max(d - got))
    if shortfall > 0.0:
        if shortfall > 1e-6:
            raise AssertionError(
                f"PLD undershoots the exact profile by {shortfall:.3e} -- refuse to "
                "certify (mu0={mu0}, q={q}, h={h}).")
        pld = _PLD(eps0=pld.eps0, h=h, mass=m, p_inf=p_inf + shortfall)
    return pld


def _truncate(pld: _PLD, tol: float = 1e-18) -> _PLD:
    """Drop tails of mass <= tol, rounding every dropped loss up.

    The upper tail is folded into p_inf (infinite loss) and the lower tail into the
    lowest retained point. Both moves can only increase delta, so domination survives.
    """
    m = pld.mass
    cum = np.cumsum(m)
    total = float(cum[-1]) if cum.size else 0.0
    if total <= 0.0:
        return pld
    # cum is non-decreasing, so searchsorted is valid here.
    lo = int(np.searchsorted(cum, tol, side="right"))
    # tail is non-increasing, so scan instead of searchsorted.
    tail = total - cum
    below = np.flatnonzero(tail <= tol)
    hi = int(below[0]) if below.size else m.size - 1             # keep [lo, hi]
    lo = min(lo, m.size - 1)
    hi = max(hi, lo)
    kept = np.array(m[lo:hi + 1], dtype=float, copy=True)
    kept[0] += float(m[:lo].sum())                    # sub-grid losses rounded UP
    dropped_hi = float(m[hi + 1:].sum())              # supra-grid losses -> +inf
    return _PLD(eps0=pld.eps0 + lo * pld.h, h=pld.h, mass=kept,
                p_inf=pld.p_inf + dropped_hi)


def _convolve(a: _PLD, b: _PLD) -> _PLD:
    """Exact linear convolution of two PLDs on a shared lattice (FFT)."""
    n = a.mass.size + b.mass.size - 1
    size = 1 << max(1, (n - 1)).bit_length()
    conv = np.fft.irfft(np.fft.rfft(a.mass, size) * np.fft.rfft(b.mass, size),
                        n=size)[:n]
    conv[conv < 0.0] = 0.0
    want = (1.0 - a.p_inf) * (1.0 - b.p_inf)
    tot = float(conv.sum())
    if tot > 0.0:
        drift = abs(tot - want) / max(want, 1e-300)
        if drift > 1e-9:
            raise AssertionError(
                f"FFT convolution mass drift {drift:.3e} (L={n}); refuse to certify.")
        conv *= want / tot                            # round-off correction only
    return _PLD(eps0=a.eps0 + b.eps0, h=a.h, mass=conv, p_inf=1.0 - want)


def _compose_pld(step: _PLD, n: int, tol: float = 1e-18) -> _PLD:
    """n-fold self-composition by binary exponentiation with conservative truncation.

    O(log n) FFT convolutions, each followed by `_truncate`, keep the arrays at the size
    of the actual support (mean +/- O(sqrt(n)) sd) instead of n times the step support.
    A single FFT power rfft(M)**n would amplify round-off until the composed mass
    drifts by about 1e-6 at n ~ 1e4.
    """
    if n < 1:
        raise ValueError("n_steps must be >= 1")
    if n == 1:
        return step
    result: Optional[_PLD] = None
    base = _truncate(step, tol)
    k = int(n)
    while k:
        if k & 1:
            result = base if result is None else _truncate(_convolve(result, base), tol)
        k >>= 1
        if k:
            base = _truncate(_convolve(base, base), tol)
    assert result is not None
    return result


# --------------------------------------------------------------------------- #
#  Accountant
# --------------------------------------------------------------------------- #
class SubsampledGaussianAccountant:
    """Certified accountant for C_q(G_mu0)^{otimes n} (substitution adjacency, WOR).

    The step PLD and the composed PLDs are built lazily and cached per n. Every delta
    and eps it returns is an upper bound on the true value.

    Args:
        mu0: per-step GDP parameter 2C / (B * sigma).
        q: record-level subsampling rate B / m_star.
        h: requested lattice spacing. The spacing actually used is
            min(h, mu0 / 40), see `_effective_h`.
        eps_lo: lower end of the step lattice. None sizes it adaptively.
        eps_hi: upper end of the step lattice. None sizes it adaptively.
        adaptive: size the lattice to the essential support. Sound either way.
        grid_tail: Gaussian tail mass used to size the lattice.

    Raises:
        ValueError: if mu0 < 0.

    Example:
        >>> acc = SubsampledGaussianAccountant.from_params(C=1.0, B=32, m_star=400,
        ...                                                sigma=1.5)
        >>> eps = acc.epsilon(target_delta=1e-5, n_steps=1000)
    """


    def __init__(self, mu0: float, q: float, *, h: float = 0.01,
                 eps_lo: Optional[float] = None, eps_hi: Optional[float] = None,
                 adaptive: bool = True, grid_tail: float = 1e-16):
        """See the class docstring."""
        if mu0 < 0:
            raise ValueError("mu0 must be >= 0")
        self.mu0 = float(mu0)
        self.q = float(q)
        self.h = _effective_h(self.mu0, float(h))   # tied to the scale of the loss
        self._adaptive = bool(adaptive)

        if self._adaptive and (eps_lo is None or eps_hi is None):
            lo, hi = _adaptive_grid_bounds(self.mu0, self.q, self.h,
                                           tail=grid_tail)
        else:
            lo, hi = _EPS_LO_MIN, _EPS_HI_MAX
        self._eps_lo = lo if eps_lo is None else float(eps_lo)
        self._eps_hi = hi if eps_hi is None else float(eps_hi)

        self._step_pld: Optional[_PLD] = None
        self._composed: Dict[int, _PLD] = {}

    @classmethod
    def from_params(cls, C: float, B: int, m_star: int, sigma: float, **kw
                    ) -> "SubsampledGaussianAccountant":
        """Build from the mechanism parameters, mu0 = 2C/(B sigma) and q = B/m_star."""
        return cls(mu0_from_clip(C, B, sigma), subsampling_rate(B, m_star), **kw)

    # -- construction ------------------------------------------------------- #
    def _step(self) -> _PLD:
        """Cached step PLD, with the top of the lattice widened until p_inf <= 1e-14."""
        if self._step_pld is None:
            lo, hi = self._eps_lo, self._eps_hi
            pld = _build_step_pld(self.mu0, self.q, lo, hi, self.h)
            # A non-negligible p_inf is sound but loose: widen the top and rebuild.
            guard = 0
            while pld.p_inf > 1e-14 and hi < _EPS_HI_MAX and guard < 8:
                hi = min(_EPS_HI_MAX, hi + max(2.0, 0.5 * (hi - lo)))
                pld = _build_step_pld(self.mu0, self.q, lo, hi, self.h)
                guard += 1
            self._eps_hi = hi
            self._step_pld = pld
        return self._step_pld

    def _composed_pld(self, n: int) -> _PLD:
        """Cached n-fold composition of the step PLD."""
        n = int(n)
        if n not in self._composed:
            self._composed[n] = _compose_pld(self._step(), n)
        return self._composed[n]

    # -- readout ------------------------------------------------------------ #
    def noise_floor(self, n_steps: int) -> float:
        """Smallest delta this accountant can resolve after n_steps compositions.

        This is the composed p_inf. Since delta(eps) = E[(1 - e^{eps - ell})_+] + p_inf,
        no delta below p_inf is representable at any eps. p_inf collects the upper
        tails folded by `_truncate` and the top cut of the step lattice. It is an
        over-estimate, so it is safe, but a target delta at or below it is meaningless.

        Args:
            n_steps: number of composed steps.

        Returns:
            The composed p_inf.
        """
        return float(self._composed_pld(n_steps).p_inf)

    def delta(self, eps: float, n_steps: int) -> float:
        """Upper bound on delta(eps) after n_steps compositions.

        Returns 0 when mu0 = 0 (infinite noise) and 1 when mu0 = inf (no noise).
        """
        if self.mu0 == 0.0:
            return 0.0
        if not math.isfinite(self.mu0):
            return 1.0
        return self._composed_pld(n_steps).delta(float(eps))

    def epsilon(self, target_delta: float, n_steps: int,
                eps_hi: float = 500.0) -> float:
        """Smallest eps whose certified delta(eps) is <= target_delta, after n_steps.

        Args:
            target_delta: target delta.
            n_steps: number of composed steps.
            eps_hi: upper end of the root-finding bracket.

        Returns:
            The certified eps. 0 when delta(0) already meets the target, and +inf when
            no eps <= eps_hi does. This covers a target_delta below `noise_floor`, and
            the tiny sigma probed during calibration.

        Raises:
            ValueError: if target_delta is between 1 and 10 times `noise_floor`, where
                the answer would only reflect numerical slack. Between 10 and 100 times
                the floor the returned eps is still a valid upper bound, accurate to a
                few percent. Only PriRFed's per-round delta split can land there:
                `orchestration.setup_dp` refuses any target below 100 times the floor.
        """
        if self.mu0 == 0.0:
            return 0.0
        if not math.isfinite(self.mu0):
            return float("inf")
        pld = self._composed_pld(n_steps)
        # Degenerate cases first: calibration probes extreme sigma, where no finite eps
        # exists. That must return +inf rather than trip the noise-floor guard.
        if pld.delta(eps_hi) > target_delta:
            return float("inf")
        if pld.delta(0.0) <= target_delta:
            return 0.0

        # Between 10x and 100x the floor the answer is valid but only accurate to a few
        # percent (see Raises). Below 10x it is refused.
        floor = self.noise_floor(n_steps)
        if target_delta < 10.0 * floor:
            raise ValueError(
                f"target_delta={target_delta:.2e} is below 10x this accountant's "
                f"numerical noise floor ({floor:.2e}) at n_steps={n_steps}: the FFT "
                "composition cannot resolve it in double precision, so the returned "
                "eps would be round-off. Raise delta, reduce n_steps, or move to an "
                "arbitrary-precision PLD.")
        eps = float(brentq(lambda e: pld.delta(e) - target_delta, 0.0, eps_hi,
                           xtol=1e-6))
        # brentq may stop just below the crossing: step up until delta(eps) <= target,
        # so the reported eps is never optimistic.
        while pld.delta(eps) > target_delta:
            eps += 1e-6
        return eps

    # -- self-checks -------------------------------------------------------- #
    def certify(self, n_probe: int = 4001) -> Dict[str, float]:
        """Check numerically that the step PLD dominates the exact profile off-lattice.

        Property (b) of the module docstring, checked on `n_probe` points spanning the
        lattice. This also catches lattice-offset bugs.

        Returns:
            Diagnostics: min, max and mean gap (PLD minus exact), p_inf, the mass at
            eps_lo, the total mass and the lattice size.

        Raises:
            AssertionError: if the PLD undershoots the exact profile by more than 1e-11.
        """
        pld = self._step()
        lo = pld.eps0
        hi = pld.eps0 + (pld.mass.size - 1) * pld.h
        probe = np.linspace(lo, hi, n_probe)
        exact = step_profile(probe, self.mu0, self.q)
        got = np.array([pld.delta(float(e)) for e in probe])
        gap = got - exact                       # must be >= 0
        worst = float(gap.min())
        if worst < -1e-11:
            raise AssertionError(
                f"per-step PLD does NOT dominate the exact profile "
                f"(min gap {worst:.3e}); the composed bound is not certified.")
        return {
            "min_gap": worst,
            "max_gap": float(gap.max()),
            "mean_gap": float(gap.mean()),
            "p_inf": float(pld.p_inf),
            "mass_at_eps_lo": float(pld.mass[0]),
            "total_mass": float(pld.total),
            "support": int(pld.mass.size),
        }


# --------------------------------------------------------------------------- #
#  sigma calibration (root-finding on the accountant)
# --------------------------------------------------------------------------- #
def calibrate_sigma(*, C: float, B: int, m_star: int, n_steps: int,
                    target_eps: float, target_delta: float,
                    sigma_lo: float = 1e-3, sigma_hi: float = 1e4,
                    tol: float = 1e-4, **acc_kw) -> Tuple[float, float]:
    """Smallest noise std sigma meeting (target_eps, target_delta) after n_steps.

    eps decreases with sigma, so the root is found by bracketing and then by `brentq`
    in log(sigma), warm-started from the un-subsampled Gaussian composition. sigma is
    then nudged up until the certified eps honours the target. C and sigma enter only
    through mu0 = 2C / (B * sigma), so sigma is linear in C.

    Args:
        C: clip constant.
        B: minibatch size.
        m_star: smallest client training-set size (sets q = B / m_star).
        n_steps: number of composed steps T * tau.
        target_eps: target epsilon.
        target_delta: target delta.
        sigma_lo: lower end of the search range.
        sigma_hi: upper end of the search range.
        tol: tolerance on log(sigma).
        **acc_kw: forwarded to `SubsampledGaussianAccountant` (in practice `h`).

    Returns:
        (sigma, realized_eps), with realized_eps <= target_eps the certified eps at
        that sigma.

    Raises:
        ValueError: if the target is infeasible at sigma_hi, or if B > m_star.
    """
    q = subsampling_rate(B, m_star)

    def eps_at(sigma: float) -> float:
        acc = SubsampledGaussianAccountant(mu0_from_clip(C, B, sigma), q, **acc_kw)
        try:
            return acc.epsilon(target_delta, n_steps)
        except AssertionError as err:
            # At extreme mu0 the composition loses mass to round-off: count the point
            # as infeasible, so the search moves to more noise.
            if "mass drift" in str(err):
                return float("inf")
            raise

    # Warm start from the un-subsampled composition mu0 * sqrt(n), which overstates the
    # loss, so sigma_est lies just above the answer.
    try:
        mu_target = brentq(lambda mu: gaussian_profile(target_eps, mu) - target_delta,
                           1e-6, 1e3, xtol=1e-8)
        sigma_est = 2.0 * C * math.sqrt(max(n_steps, 1)) / (B * mu_target)
        sigma_est = min(max(sigma_est, sigma_lo * 1.001), sigma_hi * 0.999)
    except Exception:
        sigma_est = math.sqrt(sigma_lo * sigma_hi)

    if eps_at(sigma_est) > target_eps:                 # too little noise -> grow
        lo, hi = sigma_est, sigma_est
        while eps_at(hi) > target_eps and hi < sigma_hi:
            lo, hi = hi, min(hi * 2.0, sigma_hi)
        if eps_at(hi) > target_eps:
            raise ValueError(f"target_eps={target_eps} infeasible at sigma_hi="
                             f"{sigma_hi}; raise sigma_hi.")
    else:                                              # enough noise -> shrink
        lo, hi = sigma_est, sigma_est
        while eps_at(lo) <= target_eps and lo > sigma_lo:
            lo, hi = max(lo * 0.5, sigma_lo), lo
        if eps_at(lo) <= target_eps:
            return lo, eps_at(lo)

    u = float(brentq(lambda uu: eps_at(math.exp(uu)) - target_eps,
                     math.log(lo), math.log(hi), xtol=tol))
    sigma = math.exp(u)

    # brentq may land slightly on the infeasible side. Bump sigma until the target holds.
    realized = eps_at(sigma)
    guard = 0
    while realized > target_eps and guard < 128:
        sigma *= 1.0 + 4.0 * tol
        realized = eps_at(sigma)
        guard += 1
    if realized > target_eps:
        raise RuntimeError(f"sigma calibration did not reach target_eps={target_eps} "
                           f"(eps={realized:.6g} at sigma={sigma:.6g})")
    return sigma, realized
