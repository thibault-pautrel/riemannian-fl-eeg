"""Step 1 of the aggregation study: the two-block SPDNet of each dataset, set centrally.

Architecture, two blocks with a gently decreasing dimension (d -> h1 -> h2, h1 >= h2):

    [BiMap(W1) -> ReEig(eps1) -> SPD BN] -> [BiMap(W2) -> ReEig(eps2) -> SPD BN]
        -> LogEig -> Vech -> Linear

The protocol is that of the batch-norm study (mi_spdnet_bn_calib), with one ReEig
percentile per block, so that eps1 and eps2 are tuned separately:

0. Floors (`analyze_covariances --architectures ... --per-block-percentiles`, no
   training). For every candidate architecture and every pair (q1, q2) of
   percentiles, eps1 is the q1-th percentile of the spectrum entering ReEig 1 and eps2
   the q2-th percentile of the spectrum entering ReEig 2 (after the batch norm of
   block 1). Written to $RFL_RESULTS/reeig_floors/<dataset>.json, next to the floors of
   the other studies.
1. Calibration (this plan), on the centralised SPDNet, by mean validation accuracy over
   3 seeds, for each candidate architecture:
     sweep1  learning rate x batch size, at q1 = q2 = 15 and momentum 0.1
     sweep2  q1 x q2 (eps1 and eps2)
     sweep3  momentum of the running mean of the batch norms
   The architecture with the best mean is then kept for every arm of mi_agg.

Training as in the batch-norm study: plain SGD, BiMap weights by the library's
orthogonal parametrization, learning rate halved after 20 epochs without a lower
validation loss, stop after 150, within 3000 epochs, model of the lowest validation
loss. A choice on the edge of its grid extends the grid by itself (EXTEND).
Runs per dataset: 4 architectures x (36 + 27 + 9) = 288, plus the extensions.
"""

import json
import os

from rfl.experiments.analyze_covariances import PERCENTILES
from rfl.experiments.protocol import Protocol, Variant

DATASET = os.environ.get("RFL_DATASET", "BNCI2014_001")

# Training to convergence, as in mi_spdnet_bn_calib (copied, so that a change there never
# changes the grids of this study after it started).
TRAINING = ("--optimizer", "sgd", "--stiefel", "trivialization")
PLATEAU = ("--lr-schedule", "plateau", "--lr-plateau-factor", "0.5",
           "--selection-metric", "val_loss", "--report", "selected")
CENTRAL = TRAINING + PLATEAU + ("--lr-plateau-patience", "20",
                                "--early-stop-patience", "150")
BUDGET = 3000                                # epochs of the centralised runs, at most
BN = ("--use-batch-norm",)                   # GAH mean, after each ReEig
LR = [1e-2, 3e-2, 1e-1, 3e-1]                # SGD
MOMENTUM = [0.1, 0.01, 0.3]                  # running mean; 0.01: the library default
QUANTILES = [15, 5, 30]                      # share (%) of eigenvalues a ReEig clamps
SWEEP_SEEDS = (1, 2, 3)
FINAL_SEEDS = tuple(range(101, 111))

# Per dataset: candidate batch sizes (the first one is used while the batch size is not
# tuned), the largest batch an extension may reach (about the smallest client training
# set), as in mi_spdnet_bn_calib, and the candidate two-block architectures.
STUDY = {
    "BNCI2014_001": {"channels": 22, "batches": [64, 32, 128], "batch_max": 256,
                     "architectures": ["16,8", "16,12", "20,12", "12,8"]},
    "Weibo2014": {"channels": 60, "batches": [64, 32, 128], "batch_max": 256,
                  "architectures": ["32,16", "40,20", "32,12", "48,24"]},
    "Cho2017": {"channels": 64, "batches": [32, 16, 64], "batch_max": 128,
                "architectures": ["32,16", "48,24", "32,12", "48,16"]},
    "PhysionetMI": {"channels": 64, "batches": [32, 16, 64], "batch_max": 64,
                    "architectures": ["32,16", "48,24", "32,12", "48,16"]},
    "Schirrmeister2017": {"channels": 128, "batches": [64, 32, 128], "batch_max": 256,
                          "architectures": ["64,32", "64,16", "32,16", "32,8"]},
}
if DATASET not in STUDY:
    raise SystemExit(f"RFL_DATASET={DATASET}: the aggregation study covers {sorted(STUDY)} "
                     "(add an entry to STUDY in rfl/experiments/plans/mi_agg_calib.py)")
SETTINGS = STUDY[DATASET]

EXTEND = {
    "lr": {"factor": 3, "min": 1e-4, "max": 1.0},
    "batch_size": {"factor": 2, "int": True, "min": 8, "max": SETTINGS["batch_max"]},
    "--bn-momentum": {"factor": 3, "min": 1e-3, "max": 0.9},
    "--reeig-percentile": {"values": list(PERCENTILES)},
    "--reeig-percentile-2": {"values": list(PERCENTILES)},
}


def arch_name(arch: str) -> str:
    """Variant prefix of an architecture: '32,16' -> 'a32_16'."""
    return "a" + arch.replace(",", "_")


def arch_args(arch: str) -> tuple:
    return ("--hidden-dims", arch)


def floors_ready(dataset: str = DATASET) -> bool:
    """The floors file has one floor pair per (q1, q2) for every candidate."""
    from rfl.experiments.analyze_covariances import floors_file, percentile_key
    path = floors_file(dataset)
    if not os.path.exists(path):
        return False
    with open(path) as f:
        archs = json.load(f).get("architectures", {})
    want = {percentile_key([a, b]) for a in PERCENTILES for b in PERCENTILES}
    return all(want <= set(archs.get(a, {}).get("bn", {}))
               for a in STUDY[dataset]["architectures"])


def floors_command(dataset: str = DATASET) -> list:
    return ["python", "-m", "rfl.experiments.analyze_covariances", "--datasets", dataset,
            "--max-subjects", "0", "--per-block-percentiles",
            "--architectures", *STUDY[dataset]["architectures"]]


def calibration(dataset: str = DATASET) -> Protocol:
    """The calibration Protocol of `dataset` (module docstring)."""
    if not floors_ready(dataset):
        raise SystemExit(f"no per-block ReEig floors for {dataset} yet. The DAG computes "
                         "them (bash condor/submit_dag.sh mi_agg). By hand, with the data "
                         "downloaded:\n  " + " ".join(floors_command(dataset)))
    st = STUDY[dataset]
    variants = {f"{arch_name(a)}_bn": Variant(
        args=arch_args(a) + BN,
        space={"--reeig-percentile": QUANTILES, "--reeig-percentile-2": QUANTILES,
               "--bn-momentum": MOMENTUM}) for a in st["architectures"]}
    return Protocol(
        dataset=dataset, models=("spdnet",), aggregations={"spdnet": ("projavg",)},
        kinds=("central",), model_args={"spdnet": CENTRAL}, variants=variants,
        space={"nodp": {"lr": LR, "batch_size": st["batches"], "horizon": [BUDGET]},
               "dp": {}},
        search="staged",
        stage_keys=[["lr", "batch_size"], ["--reeig-percentile", "--reeig-percentile-2"],
                    ["--bn-momentum"]],
        sweep_seeds=SWEEP_SEEDS, final_seeds=FINAL_SEEDS, name=f"mi_{dataset}_agg_calib")


def calibrated(dataset: str = DATASET) -> tuple:
    """Arguments of every arm of mi_agg: the best architecture and its floors and
    momentum, by mean validation accuracy at the last stage of each architecture."""
    cal = calibration(dataset)
    scored = []
    for arch in STUDY[dataset]["architectures"]:
        arm = f"central_spdnet_{arch_name(arch)}_bn"
        if any(cal._failed(arm, f"sweep{i + 1}") for i in cal._stages_of(arm)):
            continue                     # every cell diverged: the architecture is out
        sel = cal.selection(arm)
        scored.append((sel["mean"], arch, sel["hp"]))
    if not scored:
        raise SystemExit(f"{NAME}: every candidate architecture failed its calibration")
    _, arch, hp = max(scored, key=lambda t: t[0])
    return arch_args(arch) + BN + (
        "--reeig-percentile", f"{hp['--reeig-percentile']:g}",
        "--reeig-percentile-2", f"{hp['--reeig-percentile-2']:g}",
        "--bn-momentum", f"{hp['--bn-momentum']:g}")


NAME = f"mi_{DATASET}_agg_calib"
# Built when a stage is requested: the protocol needs the floors file.
STAGES = {f"sweep{k + 1}": (lambda k=k: calibration().sweep(k)) for k in range(3)}
