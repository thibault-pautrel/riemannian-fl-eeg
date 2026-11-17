"""Data quantity: when does federated training beat local training? SPDNet arms.

The question: with N calibration trials per class and subject, from which N does a
subject gain by joining a federation? The scenarios (rfl/experiments/datasize.py):
U<N>, every subject has N trials per class, and full, no reduction. The test trials are
the same for every N. The heterogeneous scenarios H<N> (a third of the subjects reduced)
are set aside: their reduced subjects, drawn at random per seed, cover the subjects
unevenly.

Arms, for each scenario: every combination of head (shared, shared + re-fit, personal
FedPer, local) and normalisation (none, SPD batch norm kept on each client = FedBN,
shared SPD batch norm), so both levels of personalisation (head and batch norm) are
covered. The architecture, ReEig percentile and momentum of the running mean come from
the calibration of the batch-norm study (mi_spdnet_bn_calib, centralised, all data).
With few trials the running mean has few updates, so every batch-norm arm sets it
exactly after training (--bn-recalibrate, rfl/models/bn_recalibration.py).

Each arm is tuned on its own (learning rate x batch size), on the validation accuracy
of the reduced clients, and trains to convergence (mi_spdnet_bn_calib, SCHEDULE). Companion plans, same dataset:
mi_datasize_eegnet (EEGNet arms) and mi_datasize_tslr (tangent space + logistic
regression, local and pooled re-centred). The whole study is one DAG:

    export RFL_DATASET=BNCI2014_001
    bash condor/submit_dag.sh mi_datasize --memory 12GB
    python -m rfl.analysis.datasize $RFL_RESULTS/mi_BNCI2014_001_datasize*/final \\
        --out datasize_BNCI2014_001

Cost per dataset (BNCI2014_001, 5 scenarios): 40 SPDNet arms x (36 sweep runs + 10
final seeds), 1,840 runs, plus the extensions. Trim DATASIZE below to go faster.
"""

import os

from rfl.data.registry import DATASET_CONFIGS
from rfl.experiments.datasize import (ALL_KINDS, FEDERATED, SELECTION, extend_rules,
                                      scenarios, variants)
from rfl.experiments.pipeline import Command
from rfl.experiments.plans.mi_spdnet_bn_calib import (
    EXTEND as _EXTEND, BUDGET, FINAL_SEEDS, LR, SETTINGS, SWEEP_SEEDS, TRAINING,
    floors_command, floors_ready)
from rfl.experiments.protocol import Protocol

DATASET = os.environ.get("RFL_DATASET", "BNCI2014_001")

# Per dataset: N values of the uniform scenario (below the calibration pool of a
# subject, about 0.75 x trials per class). No heterogeneous scenario (docstring).
DATASIZE = {
    "BNCI2014_001": {"ns": [5, 10, 20, 40], "hetero": []},
    "Weibo2014": {"ns": [5, 10, 20, 40], "hetero": []},
    "Cho2017": {"ns": [5, 10, 20, 40], "hetero": []},
    "PhysionetMI": {"ns": [3, 5, 10], "hetero": []},
    "Schirrmeister2017": {"ns": [5, 10, 20, 40], "hetero": []},
}
SETTINGS_DS = DATASIZE[DATASET]
N_SUBJECTS = len(DATASET_CONFIGS[DATASET]["subjects"])
K_REDUCED = max(1, round(N_SUBJECTS / 3))          # reduced subjects in H<N>
N_CLASSES = DATASET_CONFIGS[DATASET]["n_classes"]
SCENARIOS = scenarios(SETTINGS_DS["ns"], SETTINGS_DS["hetero"], K_REDUCED)
NAME = f"mi_{DATASET}_datasize"

# The rules of the calibration, the batch down to 2.
EXTEND = extend_rules({**_EXTEND,
                       "batch_size": {"factor": 2, "int": True, "min": 2,
                                      "max": SETTINGS["batch_max"]}},
                      N_CLASSES)
SELECTION = SELECTION


def study() -> Protocol:
    from rfl.experiments.plans.mi_spdnet_bn import calibrated
    nobn, bn = calibrated()
    bn = tuple(bn) + ("--bn-recalibrate",)
    norms = {"nobn": (nobn, None), "bn": (bn, None),
             "bnshared": (bn + ("--no-fedbn",), FEDERATED)}
    return Protocol(
        dataset=DATASET, models=("spdnet",), aggregations={"spdnet": ("projavg",)},
        model_args={"spdnet": TRAINING}, kinds=ALL_KINDS,
        variants=variants(norms, SCENARIOS, SETTINGS["batches"], N_CLASSES),
        space={"nodp": {"lr": LR, "batch_size": SETTINGS["batches"], "horizon": BUDGET},
               "dp": {}},
        search="staged", stage_keys=[["lr", "batch_size"]],
        sweep_seeds=SWEEP_SEEDS, final_seeds=FINAL_SEEDS, name=NAME)


STAGES = {"sweep1": lambda: study().sweep(0), "final": lambda: study().final()}
PIPELINE = [
    Command("floors", floors_command(), done=floors_ready),
    ("mi_spdnet_bn_calib", "sweep1"), ("mi_spdnet_bn_calib", "sweep2"),
    ("mi_datasize", "sweep1"), ("mi_datasize", "final"),
    ("mi_datasize_eegnet", "sweep1"), ("mi_datasize_eegnet", "final"),
    ("mi_datasize_tslr", "final"),
]
