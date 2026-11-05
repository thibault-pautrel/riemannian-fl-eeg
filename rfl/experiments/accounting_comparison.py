"""Noise required by our accountant and by PriRFed's (Theorem 7), across target epsilons.

Both accountants certify the same mechanism: per-sample gradients clipped to C,
averaged over a minibatch of size B drawn without replacement from the smallest client
(q = B / m_star), Gaussian noise of std sigma on the averaged gradient, and
tau = local_epochs * ceil(m_max / B) local steps in each of T rounds. They differ only
in the outer accounting:

  * ours composes the T * tau steps as one tensor power C_q(G_mu0)^{otimes T tau}
    (rfl/federated/privacy/accountant.py);
  * PriRFed summarises each round by an (eps_r, delta_r) pair, computed here from our
    per-round profile C_q(G_mu0)^{otimes tau} (its best case), and composes the T rounds
    by the better of sequential and advanced composition
    (rfl/federated/privacy/prirfed_accountant.py).

For each target epsilon the script calibrates the smallest sigma under each accountant
and reports sigma_ours, sigma_PriRFed and their ratio. The ratio does not depend on C.
No dataset is needed: the federation enters only through (B, T, m_star, m_max,
local_epochs).

Presets reproduce the PriRFed comparison of the DP-RFedProj paper: batch size
and number of rounds of RetAvg at epsilon = 20, and m_star, m_max the smallest and
largest client training sizes over the 10 seeds (Table of federated settings). Every
preset value can be overridden, and --dataset custom takes all of them from the
command line.

Usage (from any folder once the repository is installed, see README):
    # The three datasets of the paper, epsilon from 0.5 to 20
    python -m rfl.experiments.accounting_comparison --dataset all --out results/accounting

    # One dataset, another batch size and horizon
    python -m rfl.experiments.accounting_comparison --dataset Cho2017 --batch-size 128 \
        --rounds 300 --eps 1 5 10

    # A federation of your own
    python -m rfl.experiments.accounting_comparison --dataset custom --batch-size 64 \
        --rounds 500 --m-star 400 --m-max 450 --eps 2 8

    # Also write a figure (matplotlib) and pgfplots coordinates
    python -m rfl.experiments.accounting_comparison --dataset all --plot --pgfplots

Outputs in --out: accounting_comparison.csv, and with the flags
accounting_comparison.png and accounting_comparison_pgfplots.tex.

Runtime: a few seconds per (dataset, epsilon) pair when tau is small, up to a few
minutes and a few GB of memory when T * tau is in the tens of thousands. Runs on CPU, needs numpy and scipy
(matplotlib for --plot).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict, List


from rfl.federated.privacy.prirfed_accountant import compare_accountants

# dataset -> mechanism of the PriRFed comparison in the paper.
#   B, T:          batch size and rounds of SPDNet RetAvg at epsilon = 20
#   m_star, m_max: smallest and largest client training sizes over the 10 seeds
#   n_clients:     number of clients (subjects grouped per client)
PRESETS: Dict[str, Dict[str, int]] = {
    "BNCI2014_001": dict(B=256, T=200, m_star=1257, m_max=1326, n_clients=3),
    "Cho2017":      dict(B=64,  T=600, m_star=277,  m_max=358,  n_clients=26),
    "Weibo2014":    dict(B=32,  T=450, m_star=762,  m_max=873,  n_clients=5),
}
DEFAULT_EPS = (0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0)

FIELDS = ["dataset", "eps", "delta", "C", "B", "T", "local_epochs", "tau", "n_steps",
          "m_star", "m_max", "q", "sigma_ours", "eps_ours", "sigma_prirfed",
          "eps_prirfed", "prirfed_branch", "ratio", "rho", "sigma_prirfed_rho",
          "ratio_rho", "seconds"]


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line options. See the module docstring for examples."""
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="all",
                   choices=["all", "custom", *PRESETS],
                   help="preset federation, 'all' presets, or 'custom' (default: all)")
    p.add_argument("--eps", type=float, nargs="+", default=list(DEFAULT_EPS),
                   help="target epsilons")
    p.add_argument("--delta", type=float, default=1e-5, help="target delta")
    p.add_argument("--clip-C", type=float, default=1.0,
                   help="clip constant C (sigma is linear in C, the ratio does not depend on it)")
    p.add_argument("--batch-size", type=int, default=None, help="override B")
    p.add_argument("--rounds", type=int, default=None, help="override T")
    p.add_argument("--local-epochs", type=int, default=1,
                   help="local epochs; tau = local_epochs * ceil(m_max / B)")
    p.add_argument("--m-star", type=int, default=None,
                   help="override the smallest client training size")
    p.add_argument("--m-max", type=int, default=None,
                   help="override the largest client training size")
    p.add_argument("--n-clients", type=int, default=None, help="override the number of clients")
    p.add_argument("--client-fraction", type=float, default=1.0,
                   help="fraction of clients per round. Below 1, also reports PriRFed with "
                        "amplification by client sampling, which is only sound when the "
                        "server cannot see who participates")
    p.add_argument("--h", type=float, default=0.01,
                   help="lattice spacing of the privacy-loss-distribution accountant")
    p.add_argument("--out", default="results/accounting_comparison", help="output directory")
    p.add_argument("--plot", action="store_true", help="write a PNG of the ratio (matplotlib)")
    p.add_argument("--pgfplots", action="store_true",
                   help="write pgfplots \\addplot coordinates of the ratio")
    return p.parse_args(argv)


def federations(a: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    """Mechanism parameters per dataset, presets overridden by the command line.

    Args:
        a: parsed options.

    Returns:
        dataset -> dict with keys B, T, m_star, m_max and n_clients.

    Exits with a message if a custom federation is incomplete, if m_max < m_star or if
    B exceeds m_star.
    """
    if a.dataset == "custom":
        missing = [n for n, v in (("--batch-size", a.batch_size), ("--rounds", a.rounds),
                                  ("--m-star", a.m_star), ("--m-max", a.m_max)) if v is None]
        if missing:
            sys.exit(f"--dataset custom needs {' '.join(missing)}")
        base = {"custom": dict(B=a.batch_size, T=a.rounds, m_star=a.m_star,
                               m_max=a.m_max, n_clients=a.n_clients or 1)}
    else:
        names = list(PRESETS) if a.dataset == "all" else [a.dataset]
        base = {n: dict(PRESETS[n]) for n in names}
        for cfg in base.values():
            for key, val in (("B", a.batch_size), ("T", a.rounds), ("m_star", a.m_star),
                             ("m_max", a.m_max), ("n_clients", a.n_clients)):
                if val is not None:
                    cfg[key] = val
    for name, cfg in base.items():
        if cfg["m_max"] < cfg["m_star"]:
            sys.exit(f"{name}: m_max ({cfg['m_max']}) < m_star ({cfg['m_star']})")
        if cfg["B"] > cfg["m_star"]:
            sys.exit(f"{name}: B ({cfg['B']}) exceeds m_star ({cfg['m_star']})")
    return base


def run(a: argparse.Namespace) -> List[dict]:
    """Calibrate sigma under both accountants for every (dataset, target epsilon).

    Args:
        a: parsed options.

    Returns:
        One row per (dataset, epsilon), with the keys of FIELDS.
    """
    rows = []
    for name, cfg in federations(a).items():
        tau = a.local_epochs * math.ceil(cfg["m_max"] / cfg["B"])
        print(f"\n{name}: B={cfg['B']} T={cfg['T']} tau={tau} m_star={cfg['m_star']} "
              f"m_max={cfg['m_max']} q={cfg['B'] / cfg['m_star']:.4f} delta={a.delta:g}")
        print(f"  {'eps':>6} {'sigma_ours':>11} {'sigma_PriRFed':>14} {'ratio':>6}")
        for eps in sorted(a.eps):
            t0 = time.time()
            r = compare_accountants(
                C=a.clip_C, B=cfg["B"], m_star=cfg["m_star"], m_max=cfg["m_max"],
                T=cfg["T"], local_epochs=a.local_epochs, target_eps=eps,
                target_delta=a.delta, n_clients=cfg["n_clients"],
                client_fraction=a.client_fraction, h=a.h)
            rows.append(dict(
                dataset=name, eps=eps, delta=a.delta, C=a.clip_C, B=r.B, T=r.T,
                local_epochs=a.local_epochs, tau=r.tau, n_steps=r.n_steps,
                m_star=r.m_star, m_max=cfg["m_max"], q=r.q,
                sigma_ours=r.sigma_ours, eps_ours=r.realized_eps_ours,
                sigma_prirfed=r.sigma_prirfed, eps_prirfed=r.realized_eps_prirfed,
                prirfed_branch=r.prirfed_branch, ratio=r.sigma_ratio, rho=r.rho,
                sigma_prirfed_rho=r.sigma_prirfed_rho, ratio_rho=r.sigma_ratio_rho,
                seconds=round(time.time() - t0, 1)))
            print(f"  {eps:6g} {r.sigma_ours:11.4g} {r.sigma_prirfed:14.4g} "
                  f"{r.sigma_ratio:6.2f}", flush=True)
    return rows


def write_outputs(rows: List[dict], a: argparse.Namespace) -> None:
    """Write the CSV, and the pgfplots coordinates and the PNG when asked.

    Args:
        rows: output of `run`.
        a: parsed options (out, plot, pgfplots).
    """
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "accounting_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out / 'accounting_comparison.csv'}")
    names = list(dict.fromkeys(r["dataset"] for r in rows))

    if a.pgfplots:
        lines = []
        for n in names:
            pts = " ".join(f"({r['eps']:g},{r['ratio']:.4f})" for r in rows if r["dataset"] == n)
            lines.append(f"% {n}\n\\addplot coordinates {{{pts}}};")
        (out / "accounting_comparison_pgfplots.tex").write_text("\n".join(lines) + "\n")
        print(f"wrote {out / 'accounting_comparison_pgfplots.tex'}")

    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed, --plot skipped")
            return
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for n in names:
            pts = [(r["eps"], r["ratio"]) for r in rows if r["dataset"] == n]
            ax.plot(*zip(*pts), marker="o", ms=3, label=n)
        ax.set_xscale("log")
        ax.set_xlabel(r"target $\varepsilon$")
        ax.set_ylabel(r"$\sigma_{\mathrm{PriRFed}} / \sigma_{\mathrm{ours}}$")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "accounting_comparison.png", dpi=150)
        print(f"wrote {out / 'accounting_comparison.png'}")


def main(argv=None) -> None:
    """Entry point."""
    a = parse_args(argv)
    write_outputs(run(a), a)


if __name__ == "__main__":
    main()
