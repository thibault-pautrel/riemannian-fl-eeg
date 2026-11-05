"""Data quantity, EEGNet arms (see mi_datasize). Same scenarios and selection.

EEGNet trains with plain SGD, as SPDNet does, on the same learning-rate grid, so that
the two models differ only by their architecture and aggregation. Plain SGD has no
state, so the new optimiser of each federated round changes nothing. Its
normalisation is the BatchNorm of the original architecture: none (nobn), kept on each
client (bn, FedBN) or shared (bnshared). As for SPDNet, the running statistics are set
exactly after training (--bn-recalibrate), so that both models are treated alike.
"""

from rfl.experiments.datasize import ALL_KINDS, FEDERATED, SELECTION, variants
from rfl.experiments.plans.mi_datasize import DATASET, EXTEND, N_CLASSES, SCENARIOS
from rfl.experiments.plans.mi_spdnet_bn_calib import (FINAL_SEEDS, HORIZON, LR, SETTINGS,
                                                      SWEEP_SEEDS)
from rfl.experiments.protocol import Protocol

NAME = f"mi_{DATASET}_datasize_eegnet"
EXTEND = EXTEND                 # the rules of mi_datasize
SELECTION = SELECTION
BN = ("--eegnet-norm-type", "batch", "--bn-recalibrate")
NORMS = {"nobn": (("--eegnet-norm-type", "none"), None),
         "bn": (BN, None),
         "bnshared": (BN + ("--no-fedbn",), FEDERATED)}


def study() -> Protocol:
    return Protocol(
        dataset=DATASET, models=("eegnet",), aggregations={"eegnet": ("fedavg",)},
        model_args={"eegnet": ("--optimizer", "sgd")}, kinds=ALL_KINDS,
        variants=variants(NORMS, SCENARIOS, HORIZON, SETTINGS["batches"], N_CLASSES),
        space={"nodp": {"lr": LR, "batch_size": SETTINGS["batches"], "horizon": HORIZON},
               "dp": {}},
        search="staged", stage_keys=[["lr", "batch_size"], ["horizon"]],
        sweep_seeds=SWEEP_SEEDS, final_seeds=FINAL_SEEDS, name=NAME)


STAGES = {"sweep1": lambda: study().sweep(0), "sweep2": lambda: study().sweep(1),
          "final": lambda: study().final()}
