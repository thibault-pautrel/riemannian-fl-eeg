"""DP run setup: fix the clip constant, calibrate the noise, report the budget.

`setup_dp` is called once by the runner, before any training.

  1. m_star = min_i m_i and m_max = max_i m_i, the smallest and largest client
     training-set sizes. The guarantee is stated for a given train/validation/test
     split: the split is stratified over all trials, so it depends on every label, and
     the sizes are fixed once it is drawn (README, Limitations).
  2. q = B / m_star, the largest subsampling rate any record faces.
  3. n = T * tau with tau = local_epochs * ceil(m_max / B), the largest number of steps
     any record is composed over. Each client actually runs local_epochs * floor(m_i / B)
     steps per round, so (max q, max n) bounds every client.
  4. C is `config.dp_clip_C`, fixed a priori (C = 1 by default).
  5. sigma is the smallest noise std meeting (target_eps, target_delta) for
     C_q(G_mu0)^{otimes n}, with mu0 = 2C / (B * sigma).
  6. (C, sigma) are written onto the config, where `DPFLClient` reads them.

The accountant is called with exactly the (C, sigma) pair the clients apply. The
guarantee covers the training records. C uses no data and the models carry no
normalisation layer with running statistics, so the private phase touches only the
training records. Client participation is observable by an honest-but-curious server,
so no amplification by client sampling is claimed.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn

from rfl.federated.privacy.accountant import (
    SubsampledGaussianAccountant,
    calibrate_sigma,
    mu0_from_clip,
    n_steps_conservative,
    subsampling_rate,
)

logger = logging.getLogger(__name__)

__all__ = ["client_dataset_sizes", "setup_dp"]


def client_dataset_sizes(clients: Sequence[Any]) -> Tuple[int, int]:
    """(m_star, m_max), the smallest and largest client training-set sizes.

    Raises:
        ValueError: if `clients` is empty.
    """
    sizes = [int(c.n_samples) for c in clients]
    if not sizes:
        raise ValueError("No clients to size for DP accounting.")
    return min(sizes), max(sizes)


def setup_dp(config: Any, clients: Sequence[Any], model: nn.Module,
             criterion: nn.Module, device: torch.device,
             verbose: bool = True) -> Dict[str, Any]:
    """Calibrate sigma for the fixed C, write it onto `config` and return the budget.

    With `config.dp_accounting == 'prirfed'` sigma is calibrated by PriRFed's Theorem 7
    and `realized_eps_ours` records what this accountant certifies at that sigma
    (Section 7.7 of the DP-RFedProj paper). Otherwise it is calibrated by
    `accountant.calibrate_sigma`. With `config.dp_compare_prirfed`, the PriRFed noise for
    the same target is added to the budget for comparison.

    Args:
        config: FLConfig with dp_enabled. Reads batch_size, fl_rounds, fl_local_epochs,
            dp_target_eps, dp_target_delta, dp_accountant_h, dp_clip_C and
            dp_accounting. Writes dp_sigma.
        clients: the FL clients, with data prepared.
        model: the initial global model (unused, kept for the call signature).
        criterion: loss function (unused, kept for the call signature).
        device: torch device (unused, kept for the call signature).
        verbose: print the budget.

    Returns:
        The budget dict saved in fl_results.json: C, sigma, mu0, B, m_star, m_max, q,
        T, local_epochs, tau, n_steps, target_eps, delta, realized_eps, sigma_source,
        the accountant self-check (`certify`, `noise_floor`, `h_effective`) and, for
        calibrated runs with `config.dp_compare_prirfed`, the PriRFed comparison under
        `prirfed`.

    Raises:
        ValueError: if B > m_star, or if target_delta is within 100 times the
            accountant's noise floor.
    """
    # ---- accounting inputs ---------------------------------------------------
    B = int(config.batch_size)
    m_star, m_max = client_dataset_sizes(clients)
    if B > m_star:
        raise ValueError(
            f"batch_size B={B} > m_star={m_star} (the smallest client). A DP step "
            "consumes exactly B records and the subsampling rate is q = B/m_star, so B "
            "must not exceed the smallest client's training set.")
    q = subsampling_rate(B, m_star)
    # The guarantee holds for each client's dataset, of size m_i: delta must stay well
    # below 1/m_i for every client, so below 1/m_max.
    if float(config.dp_target_delta) >= 1.0 / m_max:
        import warnings
        warnings.warn(
            f"dp_target_delta={config.dp_target_delta:g} >= 1/m_max = {1.0 / m_max:.2g} "
            f"(largest client: {m_max} training records). A delta of this size allows "
            "the release of a whole record: take delta well below 1/m_max.",
            stacklevel=2)

    T = int(config.fl_rounds)
    local_epochs = int(config.fl_local_epochs)
    tau = local_epochs * int(math.ceil(m_max / B))
    n_steps = n_steps_conservative(T, local_epochs, m_max, B)

    target_eps = float(config.dp_target_eps)
    target_delta = float(config.dp_target_delta)
    h = float(getattr(config, "dp_accountant_h", 0.01))

    # ---- 1. the clip constant, fixed a priori ----------------------------------
    C = float(config.dp_clip_C)
    if not (math.isfinite(C) and C > 0):
        raise ValueError(f"dp_clip_C must be a positive finite number, got {C}")

    realized_eps_ours = None            # set only on a PriRFed-accounted run

    # ---- 2. the noise --------------------------------------------------------
    if str(getattr(config, "dp_accounting", "ours")) == "prirfed":
        # PriRFed baseline: same mechanism, sigma set by Theorem 7 at rho = 1.
        # realized_eps is PriRFed's claim, realized_eps_ours is our certified eps.
        from rfl.federated.privacy.prirfed_accountant import calibrate_sigma_prirfed
        sigma, realized_eps, _info = calibrate_sigma_prirfed(
            C=C, B=B, m_star=m_star, tau=tau, T=T,
            target_eps=target_eps, target_delta=target_delta, rho=1.0, h=h)
        realized_eps_ours = SubsampledGaussianAccountant(
            mu0_from_clip(C, B, sigma), q, h=h).epsilon(target_delta, n_steps)
        sigma_source = "prirfed_thm7"
        if verbose:
            logger.info(f"[DP] PriRFed accounting: sigma = {sigma:.4g} for eps = "
                  f"{realized_eps:.4g}; our accountant certifies eps = "
                  f"{realized_eps_ours:.4g} at this sigma")
    else:
        sigma, realized_eps = calibrate_sigma(
            C=C, B=B, m_star=m_star, n_steps=n_steps,
            target_eps=target_eps, target_delta=target_delta, h=h)
        sigma_source = "calibrated"

    # ---- 3. publish ----------------------------------------------------------
    # DPFLClient clips to C and adds N(0, sigma^2) to the averaged gradient.
    config.dp_clip_C = C
    config.dp_sigma = sigma

    mu0 = mu0_from_clip(C, B, sigma)

    budget: Dict[str, Any] = {
        "dp_enabled": True,
        "C": C, "sigma": sigma, "mu0": mu0,
        "B": B, "m_star": m_star, "m_max": m_max, "q": q,
        "T": T, "local_epochs": local_epochs, "tau": tau, "n_steps": n_steps,
        "target_eps": target_eps, "delta": target_delta,
        "realized_eps": realized_eps,
        "sigma_source": sigma_source,
        "dp_accounting": str(getattr(config, "dp_accounting", "ours")),
        "realized_eps_ours": realized_eps_ours,
        "accounting": (
            "record-level f-DP, substitution adjacency; per-step subsampled Gaussian "
            "C_q(G_mu0) with q = B/m_star and mu0 = 2C/(B*sigma), composed over "
            "n_steps = T * local_epochs * ceil(m_max/B) by a dominating PLD (FFT); "
            "certified upper bound. Client i runs local_epochs * floor(m_i/B) <= tau "
            "steps per round. The clip constant is fixed a priori and the private "
            "phase touches only training records. Client participation is observable, "
            "so no client-subsampling amplification is claimed."),
    }

    if math.isfinite(mu0) and mu0 > 0:
        acc = SubsampledGaussianAccountant(mu0, q, h=h)
        # certify() checks that the step PLD dominates the exact profile.
        budget["certify"] = acc.certify()
        budget["noise_floor"] = acc.noise_floor(n_steps)
        budget["h_requested"] = h
        budget["h_effective"] = acc.h

        # The composed delta is only meaningful well above the noise floor.
        if target_delta < 100.0 * budget["noise_floor"]:
            raise ValueError(
                f"target_delta={target_delta:.2e} is within 100x of the accountant's "
                f"numerical noise floor ({budget['noise_floor']:.2e}) at "
                f"n_steps={n_steps}: the composed delta cannot be resolved in double "
                "precision. Raise delta, or reduce n_steps.")

    # ---- 4. PriRFed comparison (reporting only, the trained sigma is unchanged) ---
    # rho = 1 is sound under observable participation. rho = s/N grants PriRFed its
    # client-sampling amplification. Only computed for calibrated runs, on request.
    if sigma_source == "calibrated" and bool(getattr(config, "dp_compare_prirfed", False)):
        try:
            from rfl.federated.privacy.prirfed_accountant import compare_accountants
            cmp = compare_accountants(
                C=C, B=B, m_star=m_star, m_max=m_max,
                T=T, local_epochs=local_epochs,
                target_eps=target_eps, target_delta=target_delta,
                n_clients=len(clients),
                client_fraction=float(getattr(config, "fl_client_fraction", 1.0)),
                h=h)
            budget["prirfed"] = {
                "n_clients": cmp.n_clients, "s_clients": cmp.s_clients, "rho": cmp.rho,
                "sigma_A_rho1": cmp.sigma_prirfed,
                "realized_eps_A_rho1": cmp.realized_eps_prirfed,
                "branch_rho1": cmp.prirfed_branch,
                "sigma_ratio_rho1": cmp.sigma_ratio,        # sigma_A(1) / sigma_B
                "sigma_A_rho": cmp.sigma_prirfed_rho,
                "realized_eps_A_rho": cmp.realized_eps_prirfed_rho,
                "branch_rho": cmp.prirfed_branch_rho,
                "sigma_ratio_rho": cmp.sigma_ratio_rho,     # sigma_A(rho) / sigma_B
                "note": ("rho < 1 amplification (their Thm 7) is sound only if "
                         "participation is hidden; under an observable-participation "
                         "threat model rho = 1 is the primary comparison."),
            }
            if verbose:
                logger.info(f"[DP][PriRFed] rho={cmp.rho:.3f} "
                      f"(s={cmp.s_clients}/{cmp.n_clients})  "
                      f"sigma_A(rho=1)={cmp.sigma_prirfed:.4g} "
                      f"(noise ratio x{cmp.sigma_ratio:.2f})"
                      + (f"  sigma_A(rho)={cmp.sigma_prirfed_rho:.4g} "
                         f"(x{cmp.sigma_ratio_rho:.2f})"
                         if cmp.sigma_prirfed_rho is not None
                         else "  [full participation: no rho row]"))
        except Exception as e:
            budget["prirfed"] = {"error": f"{type(e).__name__}: {e}"}
            if verbose:
                logger.info(f"[DP][PriRFed] comparison skipped: {type(e).__name__}: {e}")

    if verbose:
        logger.info(f"[DP] m_star={m_star} m_max={m_max} B={B} q={q:.4f} "
              f"tau={tau} n_steps={n_steps}")
        logger.info(f"[DP] C={C:.4g}  sigma={sigma:.4g}  mu0={mu0:.4g}  ->  "
              f"(eps={realized_eps:.4g}, delta={target_delta:.0e}) "
              f"[certified upper bound]")

    return budget
