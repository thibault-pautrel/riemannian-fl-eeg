"""Synthetic version of the batch-norm study, step 2 (the 18 arms), and its pipeline.

See synth_bn_calib. Same arms, tuning and selection as mi_spdnet_bn: learning rate x
batch size, then the horizon (shortest within one standard error of the best, extended
while the best is the longest), then the final seeds. Scaled down: 2 sweep seeds, 3
final seeds, horizon 10 to 60.

    python -m rfl.experiments.pipeline run synth_bn --jobs 2
    python -m rfl.analysis.report $RFL_RESULTS/synth_bn/final --by arm
    python -m rfl.analysis.report $RFL_RESULTS/synth_bn/final --by arm --metric client_min_acc

Expected: on subject clients FedBN helps and a shared batch norm does not, on IID
clients the three normalisations are close.
"""
from rfl.experiments.pipeline import Command
from rfl.experiments.plans.synth_bn_calib import (
    BN, EXTEND as _EXTEND, FINAL_SEEDS, IID, SPACE, SUBJECT, SWEEP_SEEDS, TRAINING,
    calibration, data_ready)
from rfl.experiments.protocol import Protocol, Variant

NAME = "synth_bn"
EXTEND = _EXTEND
SELECTION = {"prefer_smaller": ["horizon"]}


def study() -> Protocol:
    cal = calibration()
    hn, hb = cal.selected_hp("central_spdnet_nobn"), cal.selected_hp("central_spdnet_bn")
    nobn = ("--epsilons", hn["--epsilons"])
    bn = BN + ("--epsilons", hb["--epsilons"], "--bn-momentum", f"{hb['--bn-momentum']:g}")
    sh = bn + ("--no-fedbn",)
    subj, iid = ("--synthetic-clients", str(SUBJECT)), ("--synthetic-clients", str(IID))
    F, W = ("fed_shared", "fed_fedper"), ("fed_shared", "fed_fedper", "fed_local")
    return Protocol(
        "synthetic", data_args=(), common_args=(), models=("spdnet",),
        aggregations={"spdnet": ("projavg",)}, model_args={"spdnet": TRAINING},
        kinds=("central", "fed_shared", "fed_fedper", "fed_local"),
        variants={"nobn": Variant(args=subj + nobn), "bn": Variant(args=subj + bn),
                  "bnshared": Variant(args=subj + sh, kinds=F),
                  "iid_nobn": Variant(args=iid + nobn, kinds=W),
                  "iid_bn": Variant(args=iid + bn, kinds=W),
                  "iid_bnshared": Variant(args=iid + sh, kinds=F)},
        space={"nodp": {**SPACE, "horizon": [10, 5, 15]}, "dp": {}}, search="staged",
        stage_keys=[["lr", "batch_size"], ["horizon"]], sweep_seeds=SWEEP_SEEDS,
        final_seeds=FINAL_SEEDS, name="synth_bn")


STAGES = {"sweep1": lambda: study().sweep(0), "sweep2": lambda: study().sweep(1),
          "final": lambda: study().final()}
PIPELINE = [
    Command("data", ["python", "-m", "rfl.experiments.plans.synth_bn_calib", "--make-data"],
            done=data_ready),
    ("synth_bn_calib", "sweep1"), ("synth_bn_calib", "sweep2"),
    ("synth_bn", "sweep1"), ("synth_bn", "sweep2"), ("synth_bn", "final"),
]
