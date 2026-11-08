"""Step 2 of the batch-norm study for EEGNet: the 18 arms of mi_spdnet_bn, with EEGNet.

Same factors, tuning, selection and seeds as mi_spdnet_bn, on the same dataset
(RFL_DATASET):

    normalisation   none | BatchNorm kept on each client (FedBN) | shared BatchNorm
    scope           shared head | personal head (FedPer) | local only
    clients         one subject per client | IID clients

plus the centralised reference with and without batch norm. Plain SGD, and FedAvg for
the federated arms (EEGNet has no Stiefel weights). The momentum of the BatchNorm comes
from mi_eegnet_bn_calib. As in the SPDNet study, the running statistics used at
evaluation are the momentum averages of training (no --bn-recalibrate). The shared
BatchNorm averages the running means and variances of the clients with the other
weights, as FedAvg does.

    sweep1   learning rate x batch size, 12 cells x 3 seeds, horizon 200
    sweep2   horizon (200, 100, 300) at the stage-1 choice, the shortest within one
             standard error of the best
    final    the selected values on 10 seeds, the same splits as the SPDNet arms

    export RFL_DATASET=BNCI2014_001
    bash condor/submit_dag.sh mi_eegnet_bn --memory 8GB
    python -m rfl.experiments.pipeline status mi_eegnet_bn
    python -m rfl.analysis.report $RFL_RESULTS/mi_BNCI2014_001_bn/final \\
        $RFL_RESULTS/mi_BNCI2014_001_bn_eegnet/final \\
        --by model clients scope arch --metric client_mean_acc

Runs per dataset: 648 + 162 + 180, plus the extensions.

Arms, named <arm kind>_<variant>:

    central_eegnet_{nobn,bn}
    fed_eegnet_fedavg_{shared,fedper}_{nobn,bn,bnshared,iid_nobn,iid_bn,iid_bnshared}
    fed_eegnet_local_{nobn,bn,iid_nobn,iid_bn}
"""

from rfl.experiments.plans.mi_eegnet_bn_calib import BN, NOBN, TRAINING, calibration
from rfl.experiments.plans.mi_spdnet_bn import (FEDERATED, IID, SELECTION, WITH_CLIENTS)
from rfl.experiments.plans.mi_spdnet_bn import EXTEND as _EXTEND
from rfl.experiments.plans.mi_spdnet_bn_calib import (DATASET, FINAL_SEEDS, HORIZON, LR,
                                                      SETTINGS, SWEEP_SEEDS)
from rfl.experiments.protocol import Protocol, Variant

NAME = f"mi_{DATASET}_bn_eegnet"
EXTEND = _EXTEND                              # the rules of mi_spdnet_bn
SELECTION = SELECTION                         # the shortest horizon within one SE


def calibrated() -> tuple:
    """The arguments of the arms with batch norm: BatchNorm and its calibrated momentum."""
    hp = calibration().selected_hp("central_eegnet_bn")
    return BN + ("--bn-momentum", f"{hp['--bn-momentum']:g}")


def study() -> Protocol:
    """The Protocol of the study, with the calibrated momentum."""
    bn = calibrated()
    shared_bn = bn + ("--no-fedbn",)
    return Protocol(
        dataset=DATASET,
        models=("eegnet",),
        aggregations={"eegnet": ("fedavg",)},
        model_args={"eegnet": TRAINING},
        kinds=("central", "fed_shared", "fed_fedper", "fed_local"),
        variants={
            "nobn": Variant(args=NOBN),
            "bn": Variant(args=bn),
            "bnshared": Variant(args=shared_bn, kinds=FEDERATED),
            "iid_nobn": Variant(args=IID + NOBN, kinds=WITH_CLIENTS),
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

# Every step of the study, for rfl/experiments/pipeline.py and condor/submit_dag.sh.
PIPELINE = [
    ("mi_eegnet_bn_calib", "sweep1"), ("mi_eegnet_bn_calib", "sweep2"),
    ("mi_eegnet_bn", "sweep1"), ("mi_eegnet_bn", "sweep2"), ("mi_eegnet_bn", "final"),
]
