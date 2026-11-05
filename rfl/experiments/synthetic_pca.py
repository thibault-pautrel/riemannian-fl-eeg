"""Synthetic experiments of the DP-RFedProj paper: private federated PCA on Gr(d, p).

The convergence bound of DP-RFedProj charges privacy through a term proportional to
eta sigma^2 d_M / k, with eta = alpha / tau the local stepsize: the noise only counts
along the d_M directions of the manifold, not along the D = d (d + 1) / 2 directions of
the ambient space where it is drawn, and the server averages it over the k clients of a
round. The figure checks this scaling: its dashed lines are the term itself. On log
axes, proportional curves are parallel. The cost of privacy is the optimality gap
F(P_T) - F* of the last iterate of a private run minus that of the same run without
noise (same data, initial point, sampled clients, minibatches and stepsize).

  dimension  Two sweeps at epsilon = 1, with ProjAvg and with RLAvg. (a) D = 820 fixed
             (d = 40) and p from 1 to 8, so d_M grows from 39 to 256. (b) d_M = 144
             fixed and D from 351 to 10585, which needs d and p to change together:
             (d, p) from (26, 8) to (145, 1).
  clients    Cost against the number k of clients per round, at epsilon = 1, with
             ProjAvg and with RLAvg. sigma does not depend on k.
  privacy    Optimality gap against epsilon for DP-RFedProj with ProjAvg, RLAvg and
             RetAvg (our accounting) and for PriRFed (RetAvg, with the noise of its
             Theorem 7 composition over rounds, our exact accountant inside each round,
             and no amplification by client sampling). Each arm has a stepsize tuned on
             3 separate runs.

Clip constant C = 1/(2 sqrt 2), the bound on ||grad f(P; z)|| for ||z|| <= 1. It is the
smallest C that never clips any record of the unit ball, and it uses no data.

10 runs per point (seeds 0 to 9, tuning on seeds 100 to 102). The figure shows the
median and the interquartile range. The tables also print the mean and standard
deviation.

Usage (from any folder once the repository is installed, see README):
    python -m rfl.experiments.synthetic_pca all --jobs 2        # about 2 h on 2 cores
    python -m rfl.experiments.synthetic_pca privacy --jobs 2    # one experiment
    python -m rfl.experiments.synthetic_pca all --quick         # smoke test, about 1 min, writes to results/synthetic_pca_quick
    python -m rfl.experiments.synthetic_pca all --plot-only     # redraw from the CSV files

Outputs in results/synthetic_pca/: <experiment>.csv, fig_synthetic.tex (pgfplots,
for the paper), fig_synthetic.png, and tables on stdout.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")          # one BLAS thread per worker
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# The imports follow the thread settings above, which NumPy reads at import time.
import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List  # noqa: E402

import numpy as np  # noqa: E402

from rfl.analysis.tikz import grid_figure  # noqa: E402
from rfl.geometry import grassmann as gr  # noqa: E402
from rfl.synthetic_pca.algorithms import Config, dp_term, run  # noqa: E402
from rfl.synthetic_pca.data import make_federation  # noqa: E402
from rfl.federated.privacy.accountant import calibrate_sigma  # noqa: E402
from rfl.federated.privacy.prirfed_accountant import calibrate_sigma_prirfed  # noqa: E402

# Setting. Every client holds M records, so m_star = M and sigma is fixed by
# (C, B, M, T, TAU, eps, DELTA) alone: it does not depend on d, p, n or k.
N, M, D, P = 50, 5000, 40, 4
B, TAU, T, K = 500, 10, 200, 10         # TAU = M / B local steps per round
C = 1 / (2 * np.sqrt(2))                # bound on ||grad f||: smallest C that never clips
DELTA, EPS = 1e-5, 1.0
ALPHA = 2.0                             # constant stepsize of the dimension and clients runs

GRID_P = [(40, 1), (40, 2), (40, 4), (40, 8)]       # D = 820 fixed, d_M from 39 to 256
GRID_DM = [(26, 8), (40, 4), (74, 2), (145, 1)]     # d_M = 144 fixed, D from 351 to 10585
K_VALUES = [1, 2, 5, 10, 25, 50]
N_CLIENTS = N
EPS_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0]
ALPHA_GRID = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
# Panel (d): DP-RFedProj with each aggregator and our accounting, and PriRFed (RetAvg with
# its own accounting).
ARMS = [("projavg", "ours"), ("rlavg", "ours"), ("retavg", "ours"), ("retavg", "prirfed")]
# Aggregators of the dimension and clients sweeps (our accounting).
THEORY_AGGS = ["projavg", "rlavg"]
# Plot style and label of each (aggregator, accounting) pair.
LOOK = {("projavg", "ours"): ("ours", "ProjAvg"), ("rlavg", "ours"): ("rlavg", "RLAvg"),
        ("retavg", "ours"): ("retavg", "RetAvg"), ("retavg", "prirfed"): ("prirfed", "PriRFed")}
# Horizontal shift of the markers of each arm on the log x axis, so that arms with the
# same values stay visible. The ticks keep the true values.
DODGE = {("projavg", "ours"): 0.94, ("rlavg", "ours"): 1.0, ("retavg", "ours"): 1.06,
         ("retavg", "prirfed"): 1.0}
THEORY_DODGE = {"projavg": 0.96, "rlavg": 1.04}
SEEDS = list(range(10))
TUNING_SEEDS = [100, 101, 102]

FIELDS = ["exp", "grid", "aggregator", "accounting", "d", "p", "d_M", "D", "n", "k", "m",
          "B", "tau", "T", "alpha", "C", "eps", "sigma", "seed", "gap", "gap_np", "cost",
          "clip_fraction", "dp_term", "grad2", "sin2", "seconds"]
TEXT = ("exp", "grid", "aggregator", "accounting")


def sigma_ours(eps: float, clip: float = C) -> float:
    """Noise of our accountant (subsampled Gaussian, all local steps composed) for (eps,
    DELTA) and the clip constant `clip`."""
    return calibrate_sigma(C=clip, B=B, m_star=M, n_steps=T * TAU,
                           target_eps=eps, target_delta=DELTA)[0]


def sigma_prirfed(eps: float, clip: float = C) -> float:
    """Noise that PriRFed's accounting (Theorem 7) needs for the same mechanism and target."""
    return calibrate_sigma_prirfed(C=clip, B=B, m_star=M, tau=TAU, T=T,
                                   target_eps=eps, target_delta=DELTA)[0]


def job(s: Dict) -> Dict:
    """One run. With `twin`, also the same run without noise, which shares the data, the
    initial point, the sampled clients and the minibatches."""
    t0 = time.time()
    fed = make_federation(n=s["n"], m=M, d=s["d"], p=s["p"], seed=s["seed"])
    cfg = Config(aggregator=s["aggregator"], T=s["T"], k=s["k"], tau=TAU, B=B,
                 alpha=s["alpha"], C=s["C"], sigma=s["sigma"], seed=s["seed"])
    h = run(fed, cfg)
    gap_np = float("nan")
    if s["twin"]:
        cfg.sigma = 0.0
        gap_np = run(fed, cfg).gap[-1]
    d_M = gr.intrinsic_dim(s["d"], s["p"])
    row = {key: v for key, v in s.items() if key != "twin"}
    row.update(d_M=d_M, D=gr.ambient_dim(s["d"]), m=M, B=B, tau=TAU, gap=h.gap[-1],
               gap_np=gap_np, cost=h.gap[-1] - gap_np,
               clip_fraction=h.clip_fraction,
               dp_term=dp_term(s["alpha"], s["sigma"], d_M, s["k"], TAU),
               grad2=h.grad2[-1], sin2=h.sin2[-1], seconds=round(time.time() - t0, 1))
    return row


def run_jobs(specs: List[Dict], jobs: int) -> List[Dict]:
    rows = []
    with Pool(jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(job, specs), 1):
            print(f"  [{i}/{len(specs)}] {r['exp']:10s} {r['aggregator']:7s} d={r['d']:3d} "
                  f"p={r['p']} k={r['k']:2d} eps={r['eps']:g} alpha={r['alpha']:g} "
                  f"seed={r['seed']} gap={r['gap']:.2e} ({r['seconds']}s)", flush=True)
            rows.append(r)
    return rows


def spec(exp, *, aggregator="projavg", accounting="ours", d=D, p=P, n=N, k=K, grid="",
         alpha=ALPHA, clip=C, eps=EPS, sigma, seed, twin=True) -> Dict:
    # T travels with the job: under the 'spawn' start method (macOS, Windows) the
    # workers re-import this module and would not see a T changed by --quick.
    return dict(exp=exp, grid=grid, aggregator=aggregator, accounting=accounting, d=d,
                p=p, n=n, k=k, T=T, alpha=alpha, C=clip, eps=eps, sigma=sigma, seed=seed,
                twin=twin)


def exp_dimension(seeds, jobs):
    s = sigma_ours(EPS)
    print(f"dimension: eps={EPS:g} sigma={s:.4g}")
    return run_jobs([spec("dimension", aggregator=agg, d=d, p=p, grid=g, sigma=s, seed=sd)
                     for agg in THEORY_AGGS
                     for g, grid in (("p", GRID_P), ("dM", GRID_DM)) for d, p in grid
                     for sd in seeds], jobs)


def exp_clients(seeds, jobs):
    s = sigma_ours(EPS)
    print(f"clients: eps={EPS:g} sigma={s:.4g} n={N_CLIENTS}")
    return run_jobs([spec("clients", aggregator=agg, n=N_CLIENTS, k=k, sigma=s, seed=sd)
                     for agg in THEORY_AGGS for k in K_VALUES for sd in seeds], jobs)


def exp_privacy(seeds, jobs):
    sig = {}
    for eps in EPS_VALUES:
        sig[(eps, "ours")], sig[(eps, "prirfed")] = sigma_ours(eps), sigma_prirfed(eps)
        print(f"privacy: eps={eps:g} sigma_ours={sig[(eps, 'ours')]:.4g} "
              f"sigma_PriRFed={sig[(eps, 'prirfed')]:.4g}", flush=True)
    tune = run_jobs([spec("tuning", aggregator=agg, accounting=acc, alpha=a, eps=eps,
                          sigma=sig[(eps, acc)], seed=sd, twin=False)
                     for eps in EPS_VALUES for agg, acc in ARMS for a in ALPHA_GRID
                     for sd in TUNING_SEEDS], jobs)
    best = {}
    for eps in EPS_VALUES:
        for agg, acc in ARMS:
            means = {a: np.mean([r["gap"] for r in tune if r["eps"] == eps
                                 and r["aggregator"] == agg and r["accounting"] == acc
                                 and r["alpha"] == a])
                     for a in ALPHA_GRID}
            best[(eps, agg, acc)] = a_best = min(means, key=means.get)
            edge = " (grid edge)" if a_best in (ALPHA_GRID[0], ALPHA_GRID[-1]) else ""
            print(f"  tuned alpha eps={eps:g} {agg} {acc}: {a_best}{edge}")
    final = [spec("privacy", aggregator=agg, accounting=acc, alpha=best[(eps, agg, acc)],
                  eps=eps, sigma=sig[(eps, acc)], seed=sd, twin=False)
             for eps in EPS_VALUES for agg, acc in ARMS for sd in seeds]
    return tune + run_jobs(final, jobs)


EXPERIMENTS = {"dimension": exp_dimension, "clients": exp_clients, "privacy": exp_privacy}


def write_csv(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def read_csv(path: Path) -> List[Dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in r:
            if key not in TEXT:
                r[key] = float(r[key]) if r[key] not in ("", "nan") else float("nan")
        r.setdefault("aggregator", "projavg")
        r.setdefault("accounting", "ours")
    return rows


def stats(values) -> Dict:
    q25, med, q75 = np.percentile(values, [25, 50, 75])
    return dict(median=float(med), q25=float(q25), q75=float(q75),
                mean=float(np.mean(values)), std=float(np.std(values)), runs=len(values))


def point(x, z, style) -> Dict:
    return dict(style=style, x=[x], y=[z["median"]], err_lo=[z["median"] - z["q25"]],
                err_hi=[z["q75"] - z["median"]])


def _range(values) -> tuple:
    v = [x for x in values if x > 0]
    return 10 ** np.floor(np.log10(min(v))), 10 ** np.ceil(np.log10(max(v)))


def _with_bars(style, xs, zs) -> Dict:
    return dict(style=style, x=xs, y=[z["median"] for z in zs],
                err_lo=[z["median"] - z["q25"] for z in zs],
                err_hi=[z["q75"] - z["median"] for z in zs])


def report(dim, cli, prv, fig_dir: Path) -> None:
    """Figure of the paper (panels a to d) and its tables on stdout."""
    sigma = dim[0]["sigma"]
    # The dashed lines are the DP term itself. Proportional means a constant ratio.
    panels = []
    # (a) and (b): the two dimension sweeps.
    for g, key, title, xlabel in (
            ("p", "d_M", "(a) $D=820$, $d_{\\mathcal M}$ varies",
             "manifold dimension $d_{\\mathcal M}$"),
            ("dM", "D", "(b) $d_{\\mathcal M}=144$, $D$ varies",
             "ambient dimension $D$")):
        series, vals = [], []
        for agg in THEORY_AGGS:
            print(f"\n{title}, {agg}: cost of privacy, eps={EPS:g}, sigma={sigma:.4g}, n={N}")
            print(f"  {'d':>4} {'p':>3} {'d_M':>5} {'D':>6} {'median':>9} {'mean ± std':>20} "
                  f"{'dp term':>10} {'ratio':>6}")
            sub = [r for r in dim if r["grid"] == g and r["aggregator"] == agg]
            xs, zs, ps = [], [], []
            for (d, p) in sorted({(r["d"], r["p"]) for r in sub}):
                rr = [r for r in sub if r["d"] == d and r["p"] == p]
                z = stats([r["cost"] for r in rr])
                xs.append(rr[0][key]); zs.append(z); ps.append(rr[0]["dp_term"])
                print(f"  {d:4.0f} {p:3.0f} {rr[0]['d_M']:5.0f} {rr[0]['D']:6.0f} "
                      f"{z['median']:9.2e} {z['mean']:10.2e} ± {z['std']:.1e} {ps[-1]:10.2e} "
                      f"{z['median'] / ps[-1]:6.3f}")
            order = np.argsort(xs)
            xs, zs, ps = [xs[i] for i in order], [zs[i] for i in order], [ps[i] for i in order]
            series.append(_with_bars(LOOK[(agg, "ours")][0],
                                     [x * THEORY_DODGE[agg] for x in xs], zs))
            vals += [z["q25"] for z in zs] + [z["q75"] for z in zs] + ps
        series.append(dict(style="prediction", x=xs, y=ps))
        panels.append(dict(title=title, xlabel=xlabel, ylabel="cost of privacy", xticks=xs,
                           series=series, vals=vals))
    # (a) and (b) share the y range, so that the flat curve of (b) reads as flat.
    lo, hi = _range(panels[0]["vals"] + panels[1]["vals"])
    for pan in panels:
        pan["ymin"], pan["ymax"] = lo, hi
    # (c) cost against k.
    series, vals = [], []
    ks = sorted({r["k"] for r in cli})
    pk = [next(r["dp_term"] for r in cli if r["k"] == k) for k in ks]
    for agg in THEORY_AGGS:
        print(f"\n(c) cost of privacy against k, {agg}, n={N_CLIENTS}, eps={EPS:g}")
        sub = [r for r in cli if r["aggregator"] == agg]
        zs = [stats([r["cost"] for r in sub if r["k"] == k]) for k in ks]
        for k, z, p_ in zip(ks, zs, pk):
            print(f"  k={k:3.0f} median {z['median']:.2e}  mean {z['mean']:.2e} ± "
                  f"{z['std']:.1e}  dp term {p_:.2e}  ratio {z['median'] / p_:.3f}")
        series.append(_with_bars(LOOK[(agg, "ours")][0],
                                 [k * THEORY_DODGE[agg] for k in ks], zs))
        vals += [z["q25"] for z in zs] + [z["q75"] for z in zs]
    series.append(dict(style="prediction", x=ks, y=pk))
    yc = _range(vals + pk)
    panels.append(dict(title="(c) clients per round", xlabel=f"$k$ (out of $n={N_CLIENTS}$)",
                       ylabel="cost of privacy", xticks=ks, ymin=yc[0], ymax=yc[1],
                       series=series))
    # (d) gap against epsilon.
    fin = [r for r in prv if r["exp"] == "privacy"]
    print("\n(d) optimality gap of the last iterate against epsilon, tuned stepsize")
    series, vals = [], []
    for agg, acc in ARMS:
        style, name = LOOK[(agg, acc)]
        sub = [r for r in fin if r["aggregator"] == agg and r["accounting"] == acc]
        es = sorted({r["eps"] for r in sub})
        zs = [stats([r["gap"] for r in sub if r["eps"] == e]) for e in es]
        for e, z in zip(es, zs):
            r0 = next(r for r in sub if r["eps"] == e)
            print(f"  {name:22s} eps={e:4g} sigma={r0['sigma']:.4g} alpha={r0['alpha']:g} "
                  f"median {z['median']:.2e}  mean {z['mean']:.2e} ± {z['std']:.1e}")
        series.append(_with_bars(style, [e * DODGE[(agg, acc)] for e in es], zs))
        vals += [z["q25"] for z in zs] + [z["q75"] for z in zs]
    es = sorted({r["eps"] for r in fin})
    yd = _range(vals)
    panels.append(dict(title="(d) DP-RFedProj vs PriRFed", xlabel="$\\varepsilon$",
                       ylabel="optimality gap", xticks=es, ymin=yd[0], ymax=yd[1],
                       series=series))
    legend = [(LOOK[arm][1], LOOK[arm][0]) for arm in ARMS] + [
        ("$\\eta\\,\\sigma_{\\mathrm{dp}}^2 d_{\\mathcal M}/k$", "prediction")]
    fig_dir.mkdir(parents=True, exist_ok=True)
    grid_figure(fig_dir / "fig_synthetic.tex", panels, legend, cols=2,
                comment="Synthetic study. Written by rfl/experiments/synthetic_pca.py")
    _png(fig_dir / "fig_synthetic.png", panels, legend, cols=2)
    print(f"wrote {fig_dir / 'fig_synthetic.tex'} and .png")


def _png(path: Path, panels, legend, cols) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    look = {"ours": dict(color="#1B6CA8", marker="o", ms=4),
            "prirfed": dict(color="#A02C2C", marker="^", ms=5),
            "rlavg": dict(color="#2C8B57", marker="s", ms=4, mfc="white"),
            "retavg": dict(color="#E1701A", marker="D", ms=4),
            "prediction": dict(color="#444444", ls="--"),
            "nonprivate": dict(color="#444444", ls=":")}

    def plain(text):
        for a, b in (("$", ""), ("\\varepsilon", "epsilon"), ("d_{\\mathcal M}", "d_M"),
                     ("\\approx", "~"), ("=144", " = 144")):
            text = text.replace(a, b)
        return text

    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8, 3.0 * rows))
    for ax, pan in zip(axes.ravel(), panels):
        for s in pan["series"]:
            if "err_lo" in s:
                ax.errorbar(s["x"], s["y"], yerr=[s["err_lo"], s["err_hi"]], capsize=2,
                            **look[s["style"]])
            else:
                ax.plot(s["x"], s["y"], **look[s["style"]])
        ax.set_xscale("log")
        ax.set_yscale("log" if pan.get("ylog", True) else "linear")
        ax.set_xticks(pan["xticks"], [f"{t:g}" for t in pan["xticks"]])
        ax.minorticks_off()
        ax.set_ylim(pan["ymin"], pan["ymax"])
        ax.set_xlabel(plain(pan["xlabel"]))
        ax.set_ylabel(pan["ylabel"])
        ax.set_title(plain(pan["title"]), fontsize=9)
        ax.grid(alpha=0.3)
    handles = [Line2D([], [], **look[st]) for _, st in legend]
    labels = [plain(lab).replace("\\eta\\,\\sigma_{\\mathrm{dp}}^2 d_M/k",
                                 "η σ² d_M / k") for lab, _ in legend]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    global T, GRID_DM, GRID_P, K_VALUES, EPS_VALUES, ALPHA_GRID, TUNING_SEEDS
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["all", *EXPERIMENTS])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--jobs", type=int, default=2, help="parallel processes")
    ap.add_argument("--out", default="results/synthetic_pca", help="folder of the CSV files")
    ap.add_argument("--figures", default="results/synthetic_pca", help="folder of the figures")
    ap.add_argument("--plot-only", action="store_true", help="redraw from the CSV files")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 20 rounds, one seed, reduced grids")
    a = ap.parse_args(argv)
    if a.quick:                          # never overwrite the results of a full run
        if a.out == ap.get_default("out"):
            a.out += "_quick"
        if a.figures == ap.get_default("figures"):
            a.figures += "_quick"
        T, a.seeds, TUNING_SEEDS = 20, a.seeds[:1], TUNING_SEEDS[:1]
        GRID_DM, GRID_P, K_VALUES = GRID_DM[:2], GRID_P[:2], [1, 10]
        EPS_VALUES, ALPHA_GRID = [1.0, 5.0], [1.0, 2.0]
    out = Path(a.out)
    names = list(EXPERIMENTS) if a.experiment == "all" else [a.experiment]
    if not a.plot_only:
        for name in names:
            write_csv(EXPERIMENTS[name](a.seeds, a.jobs), out / f"{name}.csv")
    if all((out / f"{n}.csv").exists() for n in EXPERIMENTS):
        report(*(read_csv(out / f"{n}.csv") for n in EXPERIMENTS), Path(a.figures))
    else:
        print("the figure needs every experiment: " + ", ".join(EXPERIMENTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
