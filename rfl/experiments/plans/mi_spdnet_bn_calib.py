"""Step 1 of the batch-norm study: calibrate the dataset-level settings, centrally.

The protocol of the study (docs/options.md, SPD batch norm, Study):

0. ReEig floors (`analyze_covariances --architectures`, no training). For every
   candidate architecture, with and without batch norm, and every percentile q, each
   floor is the q-th percentile of the spectrum its ReEig acts on (after the BiMap
   congruences, and after the batch norm of the previous block when there is one).
   Written to $RFL_RESULTS/reeig_floors/<dataset>.json, read by the runners
   (--reeig-percentile q).
1. Calibration (this plan), on the centralised SPDNet, by mean validation accuracy over
   3 seeds: stage 1 tunes the learning rate and the batch size of each candidate
   architecture (q = 15), stage 2 the percentile q and, with batch norm, the momentum of
   the running mean. The best architecture is then kept, with and without batch norm
   (Schirrmeister2017: four two-block candidates; the other datasets keep one block).
2. Study (mi_spdnet_bn). Every arm, centralised or federated, personalised or not, gets
   the same tuning (learning rate x batch size, then horizon) with these settings fixed.

A grid edge extends by itself (EXTEND, rfl/experiments/pipeline.py). The whole study
runs as one HTCondor DAG: `bash condor/submit_dag.sh mi_spdnet_bn` (see mi_spdnet_bn).
Runs: 108 per dataset (72 + 36), 432 for Schirrmeister2017, plus the extensions.
`STUDY` below holds the settings of each dataset, shared with mi_spdnet_bn.
"""

import json
import os

from rfl.experiments.analyze_covariances import PERCENTILES
from rfl.experiments.protocol import Protocol, Variant

DATASET = os.environ.get("RFL_DATASET", "BNCI2014_001")

# Training of every arm, as in Gallet et al. (On batch normalization for SPDnet, 2027):
# plain SGD, BiMap weights by the library's orthogonal parametrization.
TRAINING = ("--optimizer", "sgd", "--stiefel", "trivialization")
BN = ("--use-batch-norm",)                   # GAH mean, FedBN (the defaults)
LR = [1e-2, 3e-2, 1e-1, 3e-1]                # SGD (the paper used 0.01 to 0.5)
HORIZON = [200, 100, 300]                    # rounds or epochs; 200 during stage 1
MOMENTUM = [0.1, 0.01, 0.3]                  # running mean; 0.01: the library default
QUANTILES = [15, 5, 30]                      # share (%) of eigenvalues each ReEig clamps
SWEEP_SEEDS = (1, 2, 3)
FINAL_SEEDS = tuple(range(101, 111))

# Per dataset: the candidate batch sizes (the first one is used while the batch size is
# not tuned), the largest batch an extension may reach (about the smallest client
# training set), and the candidate architectures (block dimensions). The single blocks are
# those of rfl/data/registry.py (PhysionetMI: 16, as the other 64-channel dataset).
# Schirrmeister2017 (128 channels) tries two blocks.
STUDY = {
    "BNCI2014_001": {"batches": [64, 32, 128], "batch_max": 256, "architectures": ["16"]},
    "Weibo2014": {"batches": [64, 32, 128], "batch_max": 256, "architectures": ["20"]},
    "Cho2017": {"batches": [32, 16, 64], "batch_max": 128, "architectures": ["16"]},
    "PhysionetMI": {"batches": [32, 16, 64], "batch_max": 64, "architectures": ["16"]},
    "Schirrmeister2017": {"batches": [64, 32, 128], "batch_max": 256,
                          "architectures": ["64,32", "64,16", "32,16", "32,8"]},
}
if DATASET not in STUDY:
    raise SystemExit(f"RFL_DATASET={DATASET}: the batch-norm study covers {sorted(STUDY)} "
                     "(add an entry to STUDY in rfl/experiments/plans/mi_spdnet_bn_calib.py)")
SETTINGS = STUDY[DATASET]


def arch_name(arch: str) -> str:
    """Variant prefix of an architecture: '64,32' -> 'a64_32'."""
    return "a" + arch.replace(",", "_")


# One more value beyond a grid edge (rfl/experiments/grid.py:next_value). The horizon
# only grows: a short horizon at which the validation accuracy has stopped improving is
# the converged choice.
EXTEND = {
    "lr": {"factor": 3, "min": 1e-4, "max": 3.0},
    "batch_size": {"factor": 2, "int": True, "min": 8, "max": SETTINGS["batch_max"]},
    # No practical cap: an arm still improving at its longest horizon gets a longer one.
    # (A cap of 1000 stopped four arms of BNCI2014_001 at 675 rounds, still improving.)
    "horizon": {"factor": 1.5, "int": True, "max": 20000, "down": False},
    "--bn-momentum": {"factor": 3, "min": 1e-3, "max": 0.9},
    "--reeig-percentile": {"values": list(PERCENTILES)},
}


def floors_ready(dataset: str = DATASET) -> bool:
    """The floors file exists and covers every candidate architecture."""
    from rfl.experiments.analyze_covariances import floors_file
    path = floors_file(dataset)
    if not os.path.exists(path):
        return False
    with open(path) as f:
        archs = json.load(f).get("architectures", {})
    return all(a in archs for a in STUDY[dataset]["architectures"])


def floors_command(dataset: str = DATASET) -> list:
    return ["python", "-m", "rfl.experiments.analyze_covariances", "--datasets", dataset,
            "--max-subjects", "0", "--architectures", *STUDY[dataset]["architectures"]]


def arch_args(arch: str) -> tuple:
    return ("--hidden-dims", arch)


def calibration(dataset: str = DATASET) -> Protocol:
    """The calibration Protocol of `dataset` (module docstring)."""
    if not floors_ready(dataset):
        raise SystemExit(f"no ReEig floors for {dataset} yet. The DAG computes them "
                         "(bash condor/submit_dag.sh ...). By hand, with the data "
                         "downloaded:\n  "
                         + " ".join(floors_command(dataset)))
    st = STUDY[dataset]
    variants = {}
    for arch in st["architectures"]:
        a = arch_name(arch)
        variants[f"{a}_nobn"] = Variant(args=arch_args(arch),
                                        space={"--reeig-percentile": QUANTILES})
        variants[f"{a}_bn"] = Variant(args=arch_args(arch) + BN,
                                      space={"--reeig-percentile": QUANTILES,
                                             "--bn-momentum": MOMENTUM})
    return Protocol(
        dataset=dataset,
        models=("spdnet",),
        aggregations={"spdnet": ("projavg",)},
        kinds=("central",),
        model_args={"spdnet": TRAINING},
        variants=variants,
        space={"nodp": {"lr": LR, "batch_size": st["batches"], "horizon": HORIZON[:1]},
               "dp": {}},
        search="staged",
        stage_keys=[["lr", "batch_size"], ["--reeig-percentile", "--bn-momentum"]],
        sweep_seeds=SWEEP_SEEDS,
        final_seeds=FINAL_SEEDS,
        name=f"mi_{dataset}_bn_calib",
    )


NAME = f"mi_{DATASET}_bn_calib"
# The protocol is built when a stage is requested: it needs the floors file.
STAGES = {"sweep1": lambda: calibration().sweep(0),
          "sweep2": lambda: calibration().sweep(1)}
