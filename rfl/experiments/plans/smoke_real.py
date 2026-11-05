"""Smoke test on real EEG: do the worker nodes read the downloaded data, offline?

Five short runs on RFL_DATASET (default BNCI2014_001), downloaded first with
condor/fetch_data.sh: SPDNet federated (batch norm, recalibration, reduced subjects),
SPDNet centralised, EEGNet local, EEGNet shared batch norm with a head re-fit, and the
TS+LR baseline. Two rounds or epochs each: minutes. The accuracies mean nothing.

    export RFL_DATASET=BNCI2014_001
    bash condor/submit.sh smoke_real --short
    python -m rfl.experiments.grid smoke_real --status

A job that fails on a worker but runs on the user interface points to data read over
the network: see README, Data.
"""

import os

from rfl.experiments.grid import grid

DATASET = os.environ.get("RFL_DATASET", "BNCI2014_001")
NAME = f"smoke_real_{DATASET}"
D = ["--dataset", DATASET]
RED = ["--train-trials-per-class", "5", "--reduced-subjects", "2"]
SGD = ["--optimizer", "sgd"]
JOBS = [
    ("fed_spdnet_bn", "fed", D + ["--model", "spdnet", "--aggregation", "projavg",
                                  "--use-batch-norm", "--bn-recalibrate"] + SGD + RED),
    ("central_spdnet", "central", D + ["--model", "spdnet"] + SGD),
    ("fed_eegnet_local", "fed", D + ["--model", "eegnet", "--local-only",
                                     "--eegnet-norm-type", "batch"] + SGD + RED),
    ("fed_eegnet_bnshared", "fed", D + ["--model", "eegnet", "--eegnet-norm-type", "batch",
                                        "--no-fedbn", "--bn-recalibrate",
                                        "--head-refit-epochs", "5"] + SGD),
]


def stage():
    jobs = []
    for arm, entry, args in JOBS:
        jobs += grid(arm, entry, args, [1], lrs=[0.1], batch_sizes=[32], horizons=[2])
    jobs += grid("tslr_local", "tslr", D + RED + ["--mode", "local"], [1])
    return jobs


STAGES = {"run": stage}
