"""Step 1 of the inter-subject study (SPDNet): ReEig floors and calibration, per fold.

Every subject is a target once (rfl/experiments/loso.py: one subject per fold, or 10
groups for PhysionetMI). Nothing of a fold depends on its targets: each fold has its own
ReEig floors (analyze_covariances without the targets) and its own calibration, the
protocol of mi_spdnet_bn_calib on the sources of the fold, by mean validation accuracy
of the sources over 3 seeds:

    floors   python -m rfl.experiments.loso floors (one floors file per fold)
    sweep1   learning rate x batch size of the centralised SPDNet with batch norm, q = 15
    sweep2   ReEig percentile q x momentum of the running mean

The architecture is that of the batch-norm study (one block, STUDY in
mi_spdnet_bn_calib). The study itself is mi_loso, and the whole study one DAG:

    export RFL_DATASET=BNCI2014_001
    bash condor/submit_dag.sh mi_loso --memory 12GB

Runs per fold: 36 + 27, plus the extensions.
"""

from rfl.experiments import loso
from rfl.experiments.pipeline import Command
from rfl.experiments.plans.mi_spdnet_bn_calib import (
    BN, BUDGET, DATASET, EXTEND, FINAL_SEEDS, LR, MOMENTUM, QUANTILES, SETTINGS,
    SWEEP_SEEDS, TRAINING, arch_args)
from rfl.experiments.protocol import Protocol, Variant

FOLDS = loso.folds(DATASET)
NS = loso.NS[DATASET]
if len(SETTINGS["architectures"]) != 1:
    raise SystemExit(f"{DATASET}: the inter-subject study takes the single architecture "
                     "of the batch-norm study (STUDY in mi_spdnet_bn_calib)")
ARCH = SETTINGS["architectures"][0]
NAME = f"mi_{DATASET}_loso_calib"
EXTEND = EXTEND                              # the rules of mi_spdnet_bn_calib


def floors_step() -> Command:
    return Command("loso_floors", ["python", "-m", "rfl.experiments.loso", "floors",
                                   "--dataset", DATASET, "--architectures", ARCH],
                   done=lambda: loso.floors_ready(DATASET, [ARCH]))


def calibration() -> Protocol:
    """The calibration Protocol, one variant per fold (module docstring)."""
    if not loso.floors_ready(DATASET, [ARCH]):
        raise SystemExit(f"no ReEig floors per fold for {DATASET} yet. The DAG computes "
                         "them; by hand: " + " ".join(floors_step().argv))
    variants = {f"bn_f{k}": Variant(args=arch_args(ARCH) + BN
                                    + loso.fold_args(DATASET, k, floors=True),
                                    space={"--reeig-percentile": QUANTILES,
                                           "--bn-momentum": MOMENTUM})
                for k in range(len(FOLDS))}
    return Protocol(
        dataset=DATASET, models=("spdnet",), aggregations={"spdnet": ("projavg",)},
        kinds=("central",), model_args={"spdnet": TRAINING}, variants=variants,
        space={"nodp": {"lr": LR, "batch_size": SETTINGS["batches"], "horizon": BUDGET},
               "dp": {}},
        search="staged",
        stage_keys=[["lr", "batch_size"], ["--reeig-percentile", "--bn-momentum"]],
        sweep_seeds=SWEEP_SEEDS, final_seeds=FINAL_SEEDS, name=NAME)


STAGES = {"sweep1": lambda: calibration().sweep(0),
          "sweep2": lambda: calibration().sweep(1)}
