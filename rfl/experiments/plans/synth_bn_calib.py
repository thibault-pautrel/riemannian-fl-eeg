"""Synthetic version of the batch-norm study, step 1 (data, floors, calibration).

The protocol of mi_spdnet_bn_calib + mi_spdnet_bn on a small synthetic federation, to
check the whole pipeline and the expected effects without EEG data: 6 clients of 12
channels, 3 classes, one block of 6. Each client has its own channel gains (one
'subject' per client). The IID control re-deals the same trials, stratified by class,
to 6 clients (`make_data`).

    python -m rfl.experiments.pipeline run synth_bn --jobs 2      # about 2 h on 2 cores
    bash condor/submit_dag.sh synth_bn --short                     # or on HTCondor

The floors are those of the study (the q-th percentile of the spectrum after the BiMap
congruence), computed here from the synthetic training matrices, q in (15, 5, 30).
"""
import os
import sys

import numpy as np

from rfl.experiments.grid import results_root
from rfl.experiments.protocol import Protocol, Variant

ROOT = results_root() / "synth_bn"
SUBJECT, IID = ROOT / "subject.npz", ROOT / "iid.npz"
TRAINING = ("--hidden-dims", "6", "--optimizer", "sgd", "--stiefel", "trivialization")
BN = ("--use-batch-norm",)
SPACE = {"lr": [0.1, 0.03, 0.3], "batch_size": [32, 16, 64]}
EXTEND = {"lr": {"factor": 3, "min": 1e-3, "max": 3.0},
          "batch_size": {"factor": 2, "int": True, "min": 8, "max": 128},
          "horizon": {"factor": 1.5, "int": True, "max": 60, "down": False},
          "--bn-momentum": {"factor": 3, "min": 1e-3, "max": 0.9}}
SWEEP_SEEDS, FINAL_SEEDS = (1, 2), (101, 102, 103)


def make_data() -> None:
    """subject.npz (non-IID, channel gains per client) and iid.npz (the same trials)."""
    from rfl.data.synthetic import generate_synthetic_spd, save_dataset
    d = generate_synthetic_spd(n_clients=6, samples_per_client=150, n_channels=12,
                               n_classes=3, signal_strength=0.6, mode="noniid",
                               label_skew_alpha=100.0, channel_gain_std=0.6, seed=0)
    save_dataset(d, str(SUBJECT))
    rng = np.random.default_rng(0)
    k = len(d["clients"])
    iid = {"clients": [dict() for _ in range(k)],
           "class_scale_matrices": d["class_scale_matrices"],
           "metadata": {**d["metadata"], "mode": "iid-redeal"}}
    for split in ("train", "val", "test"):
        X = np.concatenate([c[f"X_{split}"] for c in d["clients"]])
        y = np.concatenate([c[f"y_{split}"] for c in d["clients"]])
        order = np.concatenate([rng.permutation(np.where(y == c)[0]) for c in np.unique(y)])
        for i in range(k):
            iid["clients"][i][f"X_{split}"] = X[order[i::k]]
            iid["clients"][i][f"y_{split}"] = y[order[i::k]]
    save_dataset(iid, str(IID))


def data_ready() -> bool:
    return SUBJECT.exists() and IID.exists()


def floors() -> list:
    """Floors at q = 15, 5, 30 (first: the value of stage 1)."""
    from rfl.data.synthetic import load_dataset
    from rfl.experiments.analyze_covariances import architecture_floors
    X = np.concatenate([c["X_train"] for c in load_dataset(str(SUBJECT))["clients"]])
    return [f"{architecture_floors(X, [6], q, False)['floors'][0]:g}" for q in (15, 5, 30)]


def calibration() -> Protocol:
    if not data_ready():
        raise SystemExit(f"no synthetic data in {ROOT} yet. `python -m "
                         "rfl.experiments.pipeline run synth_bn` and the DAG write them, "
                         "or: python -m rfl.experiments.plans.synth_bn_calib --make-data")
    eps = floors()
    return Protocol(
        "synthetic", data_args=["--synthetic-clients", str(SUBJECT)], common_args=(),
        models=("spdnet",), aggregations={"spdnet": ("projavg",)}, kinds=("central",),
        model_args={"spdnet": TRAINING},
        variants={"nobn": Variant(space={"--epsilons": eps}),
                  "bn": Variant(args=BN, space={"--epsilons": eps,
                                                "--bn-momentum": [0.1, 0.01, 0.3]})},
        space={"nodp": {**SPACE, "horizon": [10]}, "dp": {}}, search="staged",
        stage_keys=[["lr", "batch_size"], ["--epsilons", "--bn-momentum"]],
        sweep_seeds=SWEEP_SEEDS, final_seeds=FINAL_SEEDS, name="synth_bn_calib")


NAME = "synth_bn_calib"
STAGES = {"sweep1": lambda: calibration().sweep(0), "sweep2": lambda: calibration().sweep(1)}

if __name__ == "__main__" and "--make-data" in sys.argv:
    os.makedirs(ROOT, exist_ok=True)
    make_data()
    print(f"written: {SUBJECT} {IID}")
