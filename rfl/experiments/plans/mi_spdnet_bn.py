"""Step 2 of the batch-norm study: does the SPD batch norm help federated EEG decoding?

Factors, for SPDNet on one dataset (RFL_DATASET):

    normalisation   none | SPD batch norm kept on each client (FedBN) | shared batch norm
    scope           shared head ('classical' federated) | personal head (FedPer) | local only
    clients         one subject per client (non-IID) | IID clients (the same trials
                    pooled and dealt at random to the same number of clients)

plus a centralised reference with and without batch norm: 18 arms. The IID clients are
the control: FedBN and a shared batch norm should then give the same result. The batch
norm uses the GAH mean. DP refuses the layer, so the study is non-private. Every arm
trains with plain SGD and the library's orthogonal parametrization of the BiMap weights,
and the federated arms aggregate with ProjAvg.

One protocol for every arm:

    floors        analyze_covariances --architectures (mi_spdnet_bn_calib)
    calibration   mi_spdnet_bn_calib: architecture, ReEig percentile and momentum of the
                  running mean, chosen on the centralised SPDNet, then fixed
    sweep1        learning rate x batch size, 12 cells x 3 seeds, horizon 200
    sweep2        horizon (200, 100, 300) at the stage-1 choice: the shortest horizon
                  within one standard error of the best, so that the runs converge
    final         the selected values on 10 seeds, the same splits for every arm

A choice on the edge of its grid extends the grid by itself (EXTEND), and the whole
study runs as one HTCondor DAG (PIPELINE below, rfl/experiments/pipeline.py):

    export RFL_DATASET=BNCI2014_001
    bash condor/submit_dag.sh mi_spdnet_bn --memory 12GB
    python -m rfl.experiments.pipeline status mi_spdnet_bn
    python -m rfl.analysis.report $RFL_RESULTS/mi_BNCI2014_001_bn/final \\
        --by clients scope arch --metric client_mean_acc
    python -m rfl.analysis.stats $RFL_RESULTS/mi_BNCI2014_001_bn/final \\
        --factor arch --by kind clients scope

Runs per dataset: 648 + 162 + 180, plus the extensions.

Arms, named <arm kind>_<variant>:

    central_spdnet_{nobn,bn}
    fed_spdnet_projavg_{shared,fedper}_{nobn,bn,bnshared,iid_nobn,iid_bn,iid_bnshared}
    fed_spdnet_local_{nobn,bn,iid_nobn,iid_bn}

The protocol is built when a stage is requested, from the floors file and the
calibration, which must be finished and selected first.
"""

from typing import Tuple

from rfl.experiments.pipeline import Command
from rfl.experiments.plans.mi_spdnet_bn_calib import EXTEND as _EXTEND
from rfl.experiments.plans.mi_spdnet_bn_calib import (
    BN, DATASET, FINAL_SEEDS, HORIZON, LR, SETTINGS, SWEEP_SEEDS, TRAINING,
    arch_args, arch_name, calibration, floors_command, floors_ready)
from rfl.experiments.protocol import Protocol, Variant

IID = ("--client-partition", "iid")
FEDERATED = ("fed_shared", "fed_fedper")       # a batch norm can be shared
WITH_CLIENTS = FEDERATED + ("fed_local",)      # the client partition matters
NAME = f"mi_{DATASET}_bn"


def _best(cal: Protocol, norm: str) -> Tuple[str, dict]:
    """(architecture, selected values) of the best calibrated arm with `norm`."""
    scored = []
    for arch in SETTINGS["architectures"]:
        sel = cal.selection(f"central_spdnet_{arch_name(arch)}_{norm}")
        scored.append((sel["mean"], arch, sel["hp"]))
    _, arch, hp = max(scored, key=lambda t: t[0])
    return arch, hp


def calibrated() -> Tuple[tuple, tuple]:
    """(arguments without batch norm, arguments with batch norm) of every arm."""
    cal = calibration()
    arch, hp = _best(cal, "nobn")
    nobn = arch_args(arch) + ("--reeig-percentile", f"{hp['--reeig-percentile']:g}")
    arch, hp = _best(cal, "bn")
    bn = arch_args(arch) + BN + ("--reeig-percentile", f"{hp['--reeig-percentile']:g}",
                                 "--bn-momentum", f"{hp['--bn-momentum']:g}")
    return nobn, bn


def study() -> Protocol:
    """The Protocol of the study, with the calibrated settings."""
    nobn, bn = calibrated()
    shared_bn = bn + ("--no-fedbn",)
    return Protocol(
        dataset=DATASET,
        models=("spdnet",),
        aggregations={"spdnet": ("projavg",)},
        model_args={"spdnet": TRAINING},
        kinds=("central", "fed_shared", "fed_fedper", "fed_local"),
        variants={
            "nobn": Variant(args=nobn),
            "bn": Variant(args=bn),
            "bnshared": Variant(args=shared_bn, kinds=FEDERATED),
            "iid_nobn": Variant(args=IID + nobn, kinds=WITH_CLIENTS),
            "iid_bn": Variant(args=IID + bn, kinds=WITH_CLIENTS),
            "iid_bnshared": Variant(args=IID + shared_bn, kinds=FEDERATED),
        },
        space={"nodp": {"lr": LR, "batch_size": SETTINGS["batches"], "horizon": HORIZON},
               "dp": {}},
        search="staged",
        stage_keys=[["lr", "batch_size"], ["horizon"]],
        sweep_seeds=SWEEP_SEEDS,
        final_seeds=FINAL_SEEDS,
        name=NAME,
    )


STAGES = {"sweep1": lambda: study().sweep(0),
          "sweep2": lambda: study().sweep(1),
          "final": lambda: study().final()}

# Grid extensions, the same rules as the calibration (rfl/experiments/pipeline.py).
EXTEND = _EXTEND

# The horizon is the shortest one whose validation accuracy is within one standard
# error of the best (rfl/experiments/grid.py:select).
SELECTION = {"prefer_smaller": ["horizon"]}

# Every step of the study, for rfl/experiments/pipeline.py and condor/submit_dag.sh.
PIPELINE = [
    Command("floors", floors_command(), done=floors_ready),
    ("mi_spdnet_bn_calib", "sweep1"), ("mi_spdnet_bn_calib", "sweep2"),
    ("mi_spdnet_bn", "sweep1"), ("mi_spdnet_bn", "sweep2"), ("mi_spdnet_bn", "final"),
]
