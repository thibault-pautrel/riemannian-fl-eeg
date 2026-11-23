"""Step 2 of the inter-subject study (EEGNet): the arms of mi_loso, with EEGNet.

Same folds, arms, tuning, seeds and target adaptation as mi_loso, FedAvg, the momentum
of the BatchNorm from mi_loso_eegnet_calib.

    export RFL_DATASET=BNCI2014_001
    bash condor/submit_dag.sh mi_loso_eegnet --memory 8GB
"""

from rfl.data.registry import DATASET_CONFIGS
from rfl.experiments import loso
from rfl.experiments.plans.mi_eegnet_bn_calib import BN, TRAINING
from rfl.experiments.plans.mi_loso_calib import FOLDS, NS
from rfl.experiments.plans.mi_loso_eegnet_calib import calibration
from rfl.experiments.plans.mi_spdnet_bn_calib import (
    BUDGET, DATASET, EXTEND as _EXTEND, FINAL_SEEDS, LR, SETTINGS, SWEEP_SEEDS)
from rfl.experiments.protocol import Protocol

NAME = f"mi_{DATASET}_loso_eegnet"
N_CLASSES = DATASET_CONFIGS[DATASET]["n_classes"]
EXTEND = loso.extend_rules(_EXTEND, N_CLASSES)


def calibrated(k: int) -> tuple:
    hp = calibration().selected_hp(f"central_eegnet_bn_f{k}")
    return (BN + ("--bn-momentum", f"{hp['--bn-momentum']:g}")
            + loso.fold_args(DATASET, k, floors=False))


def study() -> Protocol:
    variants = {}
    for k in range(len(FOLDS)):
        variants.update(loso.fold_variants(k, calibrated(k), NS, N_CLASSES))
    return Protocol(
        dataset=DATASET, models=("eegnet",), aggregations={"eegnet": ("fedavg",)},
        model_args={"eegnet": TRAINING},
        kinds=("central", "fed_shared", "fed_fedper", "fed_local"), variants=variants,
        space={"nodp": {"lr": LR, "batch_size": SETTINGS["batches"], "horizon": BUDGET},
               "dp": {}},
        search="staged", stage_keys=[["lr", "batch_size"]], refit_args=(),
        sweep_seeds=SWEEP_SEEDS, final_seeds=FINAL_SEEDS, name=NAME)


STAGES = {"sweep1": lambda: study().sweep(0),
          "final": lambda: loso.with_target_ns(study().final(), NS)}
PIPELINE = [("mi_loso_eegnet_calib", "sweep1"), ("mi_loso_eegnet_calib", "sweep2"),
            ("mi_loso_eegnet", "sweep1"), ("mi_loso_eegnet", "final")]
