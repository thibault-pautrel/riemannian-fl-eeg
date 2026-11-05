"""Data quantity on synthetic subjects: the protocol of mi_datasize, on SPD matrices.

10 synthetic subjects of 12 channels and 3 classes whose discriminant directions differ
(rfl/data/synthetic.py, --pattern-rotation 0.45, signal strength between 0.3 and 0.9):
with these settings a model per subject beats a model on pooled re-centred subjects with
many trials, and loses with few. SPDNet arms (one block of 6, fixed floor and momentum,
no calibration) and the TS+LR baseline; EEGNet needs raw EEG.

    python -m rfl.experiments.pipeline run synth_datasize --jobs 4
    bash condor/submit_dag.sh synth_datasize --short
    python -m rfl.analysis.datasize $RFL_RESULTS/synth_datasize*/final --out synth

About 1,500 short runs (1,008 in the first sweep): minutes on the cluster, hours on a
laptop. RFL_QUICK=1 keeps one N, the batch-norm arms and one sweep seed, in folders of
their own (synth_datasize_quick): a check of the wiring, under an hour on 2 cores. The
accuracies then mean nothing.
"""

import os

from rfl.experiments.datasize import (ALL_KINDS, FEDERATED, SELECTION, extend_rules,
                                      scenarios, variants)
from rfl.experiments.grid import results_root
from rfl.experiments.pipeline import Command
from rfl.experiments.protocol import Protocol

QUICK = os.environ.get("RFL_QUICK") == "1"
DATA = results_root() / "synth_datasize" / "subjects.npz"
DATA_ARGS = ["--synthetic-clients", str(DATA)]
GENERATE = ["python", "-m", "rfl.data.synthetic", "--n-clients", "10",
            "--samples-per-client", "300", "--n-channels", "12", "--n-classes", "3",
            "--label-skew-alpha", "1000", "--channel-gain-std", "0.6",
            "--pattern-rotation", "0.45", "--signal-strength-range", "0.3,0.9",
            "--val-ratio", "0.05", "--test-ratio", "0.25", "--seed", "0",
            "--output", str(DATA)]
NS, HETERO = ([5], [5]) if QUICK else ([5, 10, 20, 40], [5, 20])
SCENARIOS = scenarios(NS, HETERO, k_reduced=3, full=not QUICK)
TRAINING = ("--hidden-dims", "6", "--epsilons", "0.005", "--optimizer", "sgd",
            "--stiefel", "trivialization")
BN = ("--use-batch-norm", "--bn-momentum", "0.1", "--bn-recalibrate")
NORMS = ({"bn": (BN, None), "bnshared": (BN + ("--no-fedbn",), FEDERATED)} if QUICK else
         {"nobn": ((), None), "bn": (BN, None),
          "bnshared": (BN + ("--no-fedbn",), FEDERATED)})
HORIZON = [10] if QUICK else [30, 15, 45]
SWEEP_SEEDS, FINAL_SEEDS = ((1,), (101,)) if QUICK else ((1, 2), (101, 102, 103))
NAME = "synth_datasize_quick" if QUICK else "synth_datasize"
EXTEND = extend_rules({"lr": {"factor": 3, "min": 1e-3, "max": 3.0},
                       "batch_size": {"factor": 2, "int": True, "min": 2, "max": 128},
                       "horizon": {"factor": 1.5, "int": True, "max": 400, "down": False}},
                      3)
SELECTION = SELECTION


def study() -> Protocol:
    return Protocol(
        dataset="synthetic", data_args=DATA_ARGS, common_args=(), models=("spdnet",),
        aggregations={"spdnet": ("projavg",)}, model_args={"spdnet": TRAINING},
        kinds=ALL_KINDS, variants=variants(NORMS, SCENARIOS, HORIZON, [32, 16, 64], 3),
        space={"nodp": {"lr": [0.3] if QUICK else [0.1, 0.3, 0.03],
                        "batch_size": [32], "horizon": HORIZON}, "dp": {}},
        search="staged", stage_keys=[["lr", "batch_size"], ["horizon"]],
        sweep_seeds=SWEEP_SEEDS, final_seeds=FINAL_SEEDS, name=NAME)


STAGES = {"sweep1": lambda: study().sweep(0), "sweep2": lambda: study().sweep(1),
          "final": lambda: study().final()}
PIPELINE = [Command("data", GENERATE, done=DATA.exists),
            ("synth_datasize", "sweep1"), ("synth_datasize", "sweep2"),
            ("synth_datasize", "final"), ("synth_datasize_tslr", "final")]
