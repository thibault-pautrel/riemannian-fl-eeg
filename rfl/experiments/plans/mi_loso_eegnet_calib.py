"""Step 1 of the inter-subject study (EEGNet): calibration of the momentum, per fold.

The protocol of mi_eegnet_bn_calib on the sources of each fold (mi_loso_calib): stage 1
tunes the learning rate and the batch size of the centralised EEGNet with BatchNorm,
stage 2 the momentum of its running statistics. EEGNet has no ReEig floors.

Runs per fold: 36 + 9, plus the extensions.
"""

from rfl.experiments import loso
from rfl.experiments.plans.mi_eegnet_bn_calib import BN, TRAINING
from rfl.experiments.plans.mi_loso_calib import FOLDS
from rfl.experiments.plans.mi_spdnet_bn_calib import (
    BUDGET, DATASET, EXTEND, FINAL_SEEDS, LR, MOMENTUM, SETTINGS, SWEEP_SEEDS)
from rfl.experiments.protocol import Protocol, Variant

NAME = f"mi_{DATASET}_loso_eegnet_calib"
EXTEND = EXTEND


def calibration() -> Protocol:
    variants = {f"bn_f{k}": Variant(args=BN + loso.fold_args(DATASET, k, floors=False),
                                    space={"--bn-momentum": MOMENTUM})
                for k in range(len(FOLDS))}
    return Protocol(
        dataset=DATASET, models=("eegnet",), aggregations={"eegnet": ("fedavg",)},
        kinds=("central",), model_args={"eegnet": TRAINING}, variants=variants,
        space={"nodp": {"lr": LR, "batch_size": SETTINGS["batches"], "horizon": BUDGET},
               "dp": {}},
        search="staged", stage_keys=[["lr", "batch_size"], ["--bn-momentum"]],
        sweep_seeds=SWEEP_SEEDS, final_seeds=FINAL_SEEDS, name=NAME)


STAGES = {"sweep1": lambda: calibration().sweep(0),
          "sweep2": lambda: calibration().sweep(1)}
