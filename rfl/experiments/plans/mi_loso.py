"""Step 2 of the inter-subject study (SPDNet): train on the sources, adapt to the targets.

For each fold (mi_loso_calib), with its own floors and calibration, the arms of the
batch-norm study that a new subject can use, then the local reference:

    central_spdnet_bn_f<k>                         centralised on the sources
    fed_spdnet_projavg_{shared,fedper}_bn_f<k>     FedBN
    fed_spdnet_projavg_{shared,fedper}_bnshared_f<k>   shared batch norm
    fed_spdnet_local_bn_U<N>_f<k>                  each target alone, N trials per class
                                                   (local from scratch, N > 0)

Every arm is tuned on its fold (learning rate x batch size, 3 seeds), on the validation
of its own training subjects: the sources, or the N trials of the targets for the local
arms (their batch grid follows N, as in the data-quantity study). The final runs (10
seeds) of the source arms adapt their model to each target in the same job
(--target-ns, rfl/experiments/loso.py: batch-norm statistics, shared or new head, for
every N). Training to convergence, plain SGD and ProjAvg, as in the batch-norm study.

    export RFL_DATASET=BNCI2014_001
    bash condor/submit_dag.sh mi_loso --memory 12GB
    python -m rfl.experiments.pipeline status mi_loso
    python -m rfl.analysis.paper BNCI2014_001          # section "Inter-subject"

Runs per fold: 5 source arms x 36 + 3 local arms x 36 in sweep1, 80 final runs, plus the
extensions (PhysionetMI: N in 3, 5, 10).
"""

from rfl.data.registry import DATASET_CONFIGS
from rfl.experiments import loso
from rfl.experiments.plans.mi_loso_calib import (ARCH, FOLDS, NS, calibration,
                                                 floors_step)
from rfl.experiments.plans.mi_spdnet_bn_calib import (
    BN, BUDGET, DATASET, EXTEND as _EXTEND, FINAL_SEEDS, LR, SETTINGS, SWEEP_SEEDS,
    TRAINING, arch_args)
from rfl.experiments.protocol import Protocol

NAME = f"mi_{DATASET}_loso"
N_CLASSES = DATASET_CONFIGS[DATASET]["n_classes"]
# The rules of the calibration, the batch of a local arm bounded by its training set.
EXTEND = loso.extend_rules(_EXTEND, N_CLASSES)


def calibrated(k: int) -> tuple:
    """The arguments of every arm of fold k: architecture, calibrated settings, fold."""
    hp = calibration().selected_hp(f"central_spdnet_bn_f{k}")
    return (arch_args(ARCH) + BN
            + ("--reeig-percentile", f"{hp['--reeig-percentile']:g}",
               "--bn-momentum", f"{hp['--bn-momentum']:g}")
            + loso.fold_args(DATASET, k, floors=True))


def study() -> Protocol:
    variants = {}
    for k in range(len(FOLDS)):
        variants.update(loso.fold_variants(k, calibrated(k), NS, N_CLASSES))
    return Protocol(
        dataset=DATASET, models=("spdnet",), aggregations={"spdnet": ("projavg",)},
        model_args={"spdnet": TRAINING},
        kinds=("central", "fed_shared", "fed_fedper", "fed_local"), variants=variants,
        space={"nodp": {"lr": LR, "batch_size": SETTINGS["batches"], "horizon": BUDGET},
               "dp": {}},
        search="staged", stage_keys=[["lr", "batch_size"]], refit_args=(),
        sweep_seeds=SWEEP_SEEDS, final_seeds=FINAL_SEEDS, name=NAME)


STAGES = {"sweep1": lambda: study().sweep(0),
          "final": lambda: loso.with_target_ns(study().final(), NS)}
PIPELINE = [floors_step(), ("mi_loso_calib", "sweep1"), ("mi_loso_calib", "sweep2"),
            ("mi_loso", "sweep1"), ("mi_loso", "final")]
