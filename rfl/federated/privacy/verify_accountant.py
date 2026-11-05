#!/usr/bin/env python3
"""Sign-off checks for the privacy accountant. Run after any change to accountant.py.

    python -m rfl.federated.privacy.verify_accountant

Exits with code 0 if and only if every check passes. Uses numpy and scipy only, and
takes a few minutes on a CPU.

[1] Certification. The step PLD dominates the exact profile on a dense probe grid
    (`SubsampledGaussianAccountant.certify`). Every reported eps relies on this.
[2] q = 1 identity. Without subsampling, G_mu0^{otimes n} = G_{mu0 sqrt(n)}. The
    accountant must match it from above, within the lattice resolution. This catches
    lattice-offset and composition bugs.
[3] Monotonicity. eps decreases in sigma and increases in n_steps.
[4] Calibration. `calibrate_sigma` returns a sigma whose certified eps is <= target.
[5] Linearity in C. C and sigma enter only through mu0 = 2C/(B sigma), so
    sigma(C) = C * sigma(1). Both `orchestration.setup_dp` and the PriRFed comparison
    rely on it.
[6] Noise floor. A target delta below the composed p_inf is unachievable. The
    accountant must raise or return +inf, never a finite eps.
"""

from __future__ import annotations

import math
import sys

from rfl.federated.privacy.accountant import (
    SubsampledGaussianAccountant,
    calibrate_sigma,
    gaussian_profile,
    mu0_from_clip,
    subsampling_rate,
)

# Test cells spanning a range of batch sizes, client sizes and step counts. Like the
# paper's EEG runs, they compose thousands of subsampled steps.
CELLS = [
    dict(C=1.0, B=24, m_star=1200, n=8000),
    dict(C=2.0, B=32, m_star=800, n=12000),
    dict(C=0.5, B=16, m_star=600, n=5000),
]
SIGMAS = [0.3, 0.6, 1.0, 2.0, 4.0]
DELTA = 1e-5
H = 0.01

# Calibration is costly (many epsilon() calls). Its properties hold at any n_steps,
# so checks [4] and [5] use a smaller composition count.
CALIB_CELLS = [
    dict(C=1.0, B=24, m_star=1200, n=2000),
    dict(C=2.0, B=32, m_star=800, n=2000),
]
TARGET_EPS = [1.0, 3.0]


def _ok(msg: str) -> bool:
    """Print a passed check and return True."""
    print(f"  ok    {msg}")
    return True


def _fail(msg: str) -> bool:
    """Print a failed check and return False."""
    print(f"  FAIL  {msg}")
    return False


def check_certification() -> bool:
    """[1] The step PLD dominates the exact profile."""
    print("[1] per-step PLD dominates the exact profile (the certificate)")
    worst = math.inf
    for cell in CELLS:
        q = subsampling_rate(cell["B"], cell["m_star"])
        for sigma in SIGMAS:
            mu0 = mu0_from_clip(cell["C"], cell["B"], sigma)
            try:
                gap = SubsampledGaussianAccountant(mu0, q, h=H).certify()["min_gap"]
            except AssertionError as e:
                return _fail(f"certify() raised: {e}")
            worst = min(worst, gap)
    # Zero gaps are expected at the lattice points. Only a shortfall beyond round-off fails.
    if worst < -1e-11:
        return _fail(f"PLD undershoots the exact profile (min gap {worst:.2e})")
    return _ok(f"worst min_gap = {worst:.2e}  (>= -1e-11 required)")


def check_identity() -> bool:
    """[2] q = 1 composition matches G_{mu0 sqrt(n)} from above."""
    print("[2] q = 1 analytic identity:  G_mu0^{otimes n} == G_{mu0 sqrt(n)}")
    worst_over = 0.0
    worst_under = 0.0
    for mu0 in (0.03, 0.05, 0.1):
        for n in (200, 400, 800):
            acc = SubsampledGaussianAccountant(mu0, 1.0, h=H)
            mu_total = mu0 * math.sqrt(n)
            for eps in (0.1, 0.5, 1.0, 2.0, 3.0, 5.0):
                diff = acc.delta(eps, n) - gaussian_profile(eps, mu_total)
                worst_over = max(worst_over, diff)
                worst_under = min(worst_under, diff)
    if worst_under < -1e-9:
        return _fail(f"accountant reports delta BELOW the analytic truth "
                     f"({worst_under:.2e}): the bound is not an upper bound")
    if worst_over > 5e-3:
        return _fail(f"accountant is loose by {worst_over:.2e} (> 5e-3); tighten h")
    return _ok(f"upper bound, slack in [0, {worst_over:.2e}]")


def check_monotonicity() -> bool:
    """[3] eps decreases in sigma and increases in n_steps."""
    print("[3] epsilon decreases in sigma, increases in n_steps")
    for cell in CELLS:
        q = subsampling_rate(cell["B"], cell["m_star"])
        eps_by_sigma = []
        for sigma in SIGMAS:
            mu0 = mu0_from_clip(cell["C"], cell["B"], sigma)
            eps_by_sigma.append(
                SubsampledGaussianAccountant(mu0, q, h=H).epsilon(DELTA, cell["n"]))
        if any(a < b - 1e-6 for a, b in zip(eps_by_sigma, eps_by_sigma[1:])):
            return _fail(f"epsilon not decreasing in sigma: {eps_by_sigma}")

        mu0 = mu0_from_clip(cell["C"], cell["B"], 1.0)
        acc = SubsampledGaussianAccountant(mu0, q, h=H)
        eps_by_n = [acc.epsilon(DELTA, n) for n in (1000, 2000, 4000, 8000)]
        if any(a > b + 1e-6 for a, b in zip(eps_by_n, eps_by_n[1:])):
            return _fail(f"epsilon not increasing in n_steps: {eps_by_n}")
    return _ok("monotone in both arguments")


def check_calibration() -> bool:
    """[4] calibrate_sigma honours the target eps."""
    print("[4] calibrate_sigma honours the target (realized eps <= target)")
    worst_overshoot = 0.0
    for cell in CALIB_CELLS:
        for target in TARGET_EPS:
            sigma, realized = calibrate_sigma(
                C=cell["C"], B=cell["B"], m_star=cell["m_star"], n_steps=cell["n"],
                target_eps=target, target_delta=DELTA, h=H)
            if realized > target + 1e-6:
                return _fail(f"realized eps {realized:.6f} > target {target} "
                             f"at sigma={sigma:.4f}")
            worst_overshoot = max(worst_overshoot, target - realized)
    return _ok(f"every realized eps <= target (worst slack {worst_overshoot:.2e})")


def check_c_linearity() -> bool:
    """[5] sigma(C) = C * sigma(1)."""
    print("[5] sigma is linear in C  (orchestration and the PriRFed noise ratio depend on this)")
    worst = 0.0
    for cell in CALIB_CELLS:
        for target in TARGET_EPS:
            s_unit, _ = calibrate_sigma(
                C=1.0, B=cell["B"], m_star=cell["m_star"], n_steps=cell["n"],
                target_eps=target, target_delta=DELTA, h=H)
            s_real, _ = calibrate_sigma(
                C=cell["C"], B=cell["B"], m_star=cell["m_star"], n_steps=cell["n"],
                target_eps=target, target_delta=DELTA, h=H)
            rel = abs(s_unit * cell["C"] - s_real) / max(s_real, 1e-12)
            worst = max(worst, rel)
    if worst > 1e-6:
        return _fail(f"sigma(C) != C * sigma(1): worst relative error {worst:.2e}")
    return _ok(f"sigma(C) == C * sigma(1) to {worst:.2e} relative")


def check_noise_floor() -> bool:
    """[6] A delta below the noise floor is refused."""
    print("[6] a target delta below the numerical noise floor is REFUSED")
    cell = CELLS[0]
    q = subsampling_rate(cell["B"], cell["m_star"])
    acc = SubsampledGaussianAccountant(
        mu0_from_clip(cell["C"], cell["B"], 1.0), q, h=H)
    floor = acc.noise_floor(cell["n"])
    if floor <= 0.0:
        return _ok("noise floor is exactly 0 (nothing to refuse)")

    # Raising or returning +inf are both correct. A finite eps would be meaningless.
    try:
        eps = acc.epsilon(floor / 10.0, cell["n"])
    except ValueError:
        return _ok(f"delta below the floor ({floor:.2e}) raises, as it must")
    if math.isinf(eps):
        return _ok(f"delta below the floor ({floor:.2e}) returns +inf: unachievable "
                   "at any finite eps, which is the correct refusal")
    return _fail(f"an unresolvable delta ({floor / 10:.2e} < floor {floor:.2e}) "
                 f"returned a FINITE eps = {eps:.4f}; that number is round-off")


def main(argv=None) -> int:
    """Run all checks. Returns 0 if they all pass, 1 otherwise."""
    import argparse
    argparse.ArgumentParser(
        allow_abbrev=False,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    print("=" * 72)
    print("Accountant sign-off  (certification, identity, monotonicity, calibration)")
    print("=" * 72)
    checks = [
        check_certification,
        check_identity,
        check_monotonicity,
        check_calibration,
        check_c_linearity,
        check_noise_floor,
    ]
    results = [check() for check in checks]
    print("-" * 72)
    if all(results):
        print("ALL CHECKS PASSED -- the reported epsilon is a certified upper bound.")
        return 0
    print("SOME CHECKS FAILED -- do NOT use these numbers.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
