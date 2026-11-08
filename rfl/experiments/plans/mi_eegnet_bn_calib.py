"""Step 1 of the batch-norm study for EEGNet: calibrate the momentum of the BatchNorm.

The study of mi_spdnet_bn_calib + mi_spdnet_bn, with EEGNet in place of SPDNet. The only
dataset-level setting of EEGNet that the batch norm brings is the momentum of its
running mean and variance (--bn-momentum), used at evaluation. It is chosen as for
SPDNet, on the centralised model with batch norm, by mean validation accuracy over 3
seeds: stage 1 tunes the learning rate and the batch size (momentum 0.1), stage 2 the
momentum. The study (mi_eegnet_bn) then fixes it for every arm with batch norm.

EEGNet has no ReEig and no architecture to choose, so there is no floors step and no
calibration without batch norm. Every other setting is the one of the SPDNet study:
plain SGD, the same learning-rate, batch, horizon and momentum grids, the same seeds and
the same grid extensions.

    export RFL_DATASET=BNCI2014_001
    bash condor/submit_dag.sh mi_eegnet_bn --memory 8GB

Runs: 36 + 9 per dataset, plus the extensions.
"""

from rfl.experiments.plans.mi_spdnet_bn_calib import (
    DATASET, EXTEND, FINAL_SEEDS, HORIZON, LR, MOMENTUM, STUDY, SWEEP_SEEDS)
from rfl.experiments.protocol import Protocol, Variant

TRAINING = ("--optimizer", "sgd")            # as SPDNet, without the BiMap weights
NOBN = ("--eegnet-norm-type", "none")
BN = ("--eegnet-norm-type", "batch")         # BatchNorm2d, FedBN (the default)
EXTEND = EXTEND                              # the rules of mi_spdnet_bn_calib


def calibration(dataset: str = DATASET) -> Protocol:
    """The calibration Protocol of `dataset` (module docstring)."""
    st = STUDY[dataset]
    return Protocol(
        dataset=dataset,
        models=("eegnet",),
        aggregations={"eegnet": ("fedavg",)},
        kinds=("central",),
        model_args={"eegnet": TRAINING},
        variants={"bn": Variant(args=BN, space={"--bn-momentum": MOMENTUM})},
        space={"nodp": {"lr": LR, "batch_size": st["batches"], "horizon": HORIZON[:1]},
               "dp": {}},
        search="staged",
        stage_keys=[["lr", "batch_size"], ["--bn-momentum"]],
        sweep_seeds=SWEEP_SEEDS,
        final_seeds=FINAL_SEEDS,
        name=f"mi_{dataset}_bn_eegnet_calib",
    )


NAME = f"mi_{DATASET}_bn_eegnet_calib"
STAGES = {"sweep1": lambda: calibration().sweep(0),
          "sweep2": lambda: calibration().sweep(1)}
