"""The aggregation study (mi_agg_calib, then mi_agg) on a small synthetic federation.

A demonstration of the whole pipeline and of its outputs, in about an hour on a laptop:
same steps, same arms and same selection rules, at a reduced scale.

* Data: 8 clients of 100 SPD matrices (12 x 12, 2 classes, rfl/data/synthetic.py),
  'subj' = non-IID clients (label skew and channel gains), 'iid' = IID clients.
* Floors: computed from the spectra of the pooled training covariances, for every pair
  (q1, q2) in {5, 15, 30} (the synthetic data have no floors file, so --epsilons).
* Calibration (central, 2 seeds): lr, then eps1 x eps2, then BN momentum, for the
  candidate architectures 8,6 and 10,6. The best architecture serves every arm.
* Study: 4 rules x 2 client types x participation {10, 50, 100} % x E {1, 10}, FedPer +
  FedBN, lr tuned per arm (2 values x 2 seeds), 3 final seeds. Patiences and budgets
  scaled as in mi_agg, from a reference rule of 5 / 15 / 60 rounds.

    python -m rfl.experiments.pipeline run agg_synth --jobs 2
    python -m rfl.analysis.aggregation $RFL_RESULTS/agg_synth/final \\
        --sweep $RFL_RESULTS/agg_synth/sweep1 --out agg_synth
"""

import json
import math
import subprocess
import sys

import numpy as np

from rfl.experiments.grid import REPO, results_root
from rfl.experiments.pipeline import Command
from rfl.experiments.protocol import Protocol, Variant

N_CLIENTS = 8
ARCHS = ["8,6", "10,6"]
QUANTILES = [15, 5, 30]
TRAINING = ("--optimizer", "sgd", "--stiefel", "trivialization", "--lr-schedule", "plateau",
            "--lr-plateau-factor", "0.5", "--selection-metric", "val_loss",
            "--report", "selected")
REFERENCE = {"plateau": 5, "stop": 15, "budget": 60}
MIN_PARTICIPATIONS = {"plateau": 2, "stop": 4, "budget": 12}
RULES = ("projavg", "rlavg", "retavg", "pseavg")
PARTICIPATION = (10, 50, 100)
EPOCHS = (1, 10)
EXTEND = {"lr": {"factor": 3, "min": 1e-3, "max": 1.0}}
NAME = "agg_synth"


def _root():
    return results_root() / NAME


def data(kind: str) -> str:
    return str(_root() / f"synth_{kind}.npz")


def _floors_path():
    return _root() / "floors.json"


def prepare() -> None:
    """The two federations and the floors of every candidate architecture."""
    _root().mkdir(parents=True, exist_ok=True)
    for kind, mode in (("subj", "noniid"), ("iid", "iid")):
        subprocess.check_call(
            [sys.executable, "-m", "rfl.data.synthetic", "--n-clients", str(N_CLIENTS),
             "--samples-per-client", "100", "--n-channels", "12", "--n-classes", "2",
             "--mode", mode, "--signal-strength", "0.2", "--seed", "0",
             "--output", data(kind)], cwd=str(REPO), stdout=subprocess.DEVNULL)
    from rfl.data.synthetic import load_dataset
    from rfl.experiments.analyze_covariances import architecture_floor_grid, percentile_key
    covs = np.concatenate([c["X_train"] for c in load_dataset(data("subj"))["clients"]])
    out = {a: {percentile_key(q): r["floors"] for q, r in
               architecture_floor_grid(covs, [int(x) for x in a.split(",")], QUANTILES,
                                       True).items()} for a in ARCHS}
    _floors_path().write_text(json.dumps(out, indent=1))


def _ready() -> bool:
    return _floors_path().exists() and all(
        __import__("os").path.exists(data(k)) for k in ("subj", "iid"))


def _eps(arch: str, q1: float, q2: float) -> str:
    from rfl.experiments.analyze_covariances import percentile_key
    floors = json.loads(_floors_path().read_text())[arch][percentile_key([q1, q2])]
    return ",".join(f"{f:.4g}" for f in floors)


def calibration() -> Protocol:
    if not _ready():
        raise SystemExit("no synthetic data yet: run the prepare step first")
    pairs = [(a, b) for a in QUANTILES for b in QUANTILES]
    variants = {f"a{a.replace(',', '_')}_bn": Variant(
        args=("--hidden-dims", a, "--use-batch-norm"),
        space={"--epsilons": [_eps(a, *q) for q in pairs], "--bn-momentum": [0.1, 0.3]})
        for a in ARCHS}
    return Protocol(
        dataset="synthetic", data_args=["--synthetic-clients", data("subj")],
        models=("spdnet",), aggregations={"spdnet": ("projavg",)}, kinds=("central",),
        model_args={"spdnet": TRAINING + ("--lr-plateau-patience", "5",
                                          "--early-stop-patience", "15")},
        variants=variants,
        space={"nodp": {"lr": [0.05, 0.2], "batch_size": [32], "horizon": [60]}, "dp": {}},
        search="staged", stage_keys=[["lr", "batch_size"], ["--epsilons"], ["--bn-momentum"]],
        sweep_seeds=(1, 2), final_seeds=(101,), name=f"{NAME}_calib")


def calibrated() -> tuple:
    cal = calibration()
    best = max(((cal.selection(f"central_spdnet_a{a.replace(',', '_')}_bn"), a) for a in ARCHS),
               key=lambda t: t[0]["mean"])
    hp = best[0]["hp"]
    return ("--hidden-dims", best[1], "--use-batch-norm", "--epsilons", hp["--epsilons"],
            "--bn-momentum", f"{hp['--bn-momentum']:g}")


def schedule(percent: int, epochs: int) -> dict:
    m = max(1, math.ceil(N_CLIENTS * (percent / 100)))
    pe = m / N_CLIENTS
    return {k: max(math.ceil(REFERENCE[k] / (pe * epochs)),
                   math.ceil(MIN_PARTICIPATIONS[k] / pe)) for k in REFERENCE}


def study() -> Protocol:
    variants = {}
    for c in ("subj", "iid"):
        for pct in PARTICIPATION:
            for e in EPOCHS:
                s = schedule(pct, e)
                variants[f"{c}_p{pct}_e{e}"] = Variant(
                    args=("--synthetic-clients", data(c))
                    + (("--client-fraction", f"{pct / 100:g}") if pct < 100 else ())
                    + (("--local-epochs", str(e)) if e > 1 else ())
                    + ("--lr-plateau-patience", str(s["plateau"]),
                       "--early-stop-patience", str(s["stop"])),
                    space={"horizon": [s["budget"]]})
    return Protocol(
        dataset="synthetic", data_args=[], common_args=(), models=("spdnet",),
        aggregations={"spdnet": RULES}, kinds=("fed_fedper",),
        model_args={"spdnet": TRAINING + calibrated()}, variants=variants,
        space={"nodp": {"lr": [0.05, 0.2], "batch_size": [32]}, "dp": {}},
        search="staged", stage_keys=[["lr", "batch_size"]],
        sweep_seeds=(1, 2), final_seeds=(101, 102, 103), name=NAME)


STAGES = {"sweep1": lambda: study().sweep(0), "final": lambda: study().final()}
PIPELINE = [Command("prepare", [sys.executable, "-c",
                                "from rfl.experiments.plans.agg_synth import prepare; prepare()"],
                    done=_ready),
            ("agg_synth_calib", "sweep1"), ("agg_synth_calib", "sweep2"),
            ("agg_synth_calib", "sweep3"),
            ("agg_synth", "sweep1"), ("agg_synth", "final")]
