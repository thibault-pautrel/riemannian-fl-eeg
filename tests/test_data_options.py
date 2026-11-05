"""Data options that must match across entry points: band-pass, task and split seed."""
import numpy as np
import pandas as pd
import pytest

from rfl.data.data_utils import filter_binary_task
from rfl.utils.config import ExperimentConfig


class _Stop(Exception):
    pass


def test_task_defaults_to_all_classes():
    X = np.zeros((6, 2, 4))
    y = np.array([0, 1, 2, 3, 0, 1])
    meta = pd.DataFrame({"subject": [1] * 6})
    _, y2, _ = filter_binary_task(X, y, meta, verbose=False)
    assert np.array_equal(y2, y)
    assert ExperimentConfig(device="cpu").task == "all_classes"
    assert ExperimentConfig(device="cpu").cov_estimator == "scm"


def test_centralized_runner_uses_the_configured_band(monkeypatch):
    from rfl.experiments import run_centralized_experiment as rc

    seen = {}

    def fake_load(**kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(rc, "load_moabb_dataset", fake_load)
    cfg = ExperimentConfig(device="cpu", fmin=4.0, fmax=40.0)
    with pytest.raises(_Stop):
        rc.prepare_centralized_data("BNCI2014_001", [1], config=cfg)
    assert (seen["fmin"], seen["fmax"]) == (4.0, 40.0)

    args = rc._build_arg_parser().parse_args(["--dataset", "BNCI2014_001"])
    assert (args.fmin, args.fmax, args.task, args.cov_estimator) == (
        8.0, 32.0, "all_classes", "scm")


def test_federated_runner_band_defaults():
    from rfl.experiments.run_federated_experiment import _build_arg_parser

    args = _build_arg_parser().parse_args(["--dataset", "BNCI2014_001"])
    assert (args.fmin, args.fmax, args.task, args.cov_estimator) == (
        8.0, 32.0, "all_classes", "scm")


@pytest.mark.parametrize("ds,d,reeig", [("BNCI2014_001", 16, 8e-5),
                                        ("Cho2017", 16, 6e-5),
                                        ("Weibo2014", 20, 2e-5),
                                        ("Schirrmeister2017", 16, 4e-6)])
def test_federated_runner_defaults(ds, d, reeig):
    from rfl.experiments.run_federated_experiment import _build_arg_parser, apply_defaults

    args = apply_defaults(
        _build_arg_parser().parse_args(["--dataset", ds, "--learning-rate", "0.1"]), ds)
    assert (args.subjects_per_client, args.hidden_dim, args.epsilon) == (1, d, reeig)
    assert args.aggregation == "projavg" and args.local_epochs == 1
    assert args.lr_schedule == "cosine" and args.lr_min == pytest.approx(1e-3)
    assert args.report == "last" and args.early_stop_patience == 0
    assert args.dp_clip_C == 1.0 and args.dp_target_delta == 1e-5
    assert args.eegnet_norm_type == "none"
    assert not (args.dp_enabled or args.local_fc or args.local_only or args.freeze_trunk)
    assert args.local_head_mode == "train" and args.head_refit_epochs == ""

    eeg = apply_defaults(
        _build_arg_parser().parse_args(["--dataset", ds, "--model", "eegnet"]), ds)
    assert eeg.aggregation == "fedavg" and eeg.hidden_dim is None


def test_centralized_runner_defaults():
    from rfl.experiments.run_centralized_experiment import _build_arg_parser, apply_defaults

    args = apply_defaults(
        _build_arg_parser().parse_args(["--dataset", "Weibo2014", "--learning-rate",
                                        "1e-3"]), "Weibo2014")
    assert (args.hidden_dim, args.epsilon) == (20, 2e-5)
    assert args.lr_schedule == "cosine" and args.lr_min == pytest.approx(1e-5)
    assert args.report == "last" and args.early_stopping_patience == 0
    assert args.eegnet_norm_type == "none" and not args.freeze_trunk


@pytest.mark.slow
def test_entry_points_parse_help():
    """Every entry point imports and builds its parser."""
    import subprocess
    import sys
    for mod in ("rfl.experiments.fetch_dataset", "rfl.experiments.run_federated_experiment",
                "rfl.experiments.run_centralized_experiment",
                "rfl.experiments.accounting_comparison", "rfl.experiments.analyze_covariances",
                "rfl.experiments.grid", "rfl.experiments.synthetic_pca", "rfl.data.synthetic",
                "rfl.federated.privacy.verify_accountant", "rfl.analysis.report", "rfl.analysis.stats"):
        r = subprocess.run([sys.executable, "-m", mod, "--help"], capture_output=True,
                           text=True)
        assert r.returncode == 0, (mod, r.stderr[-500:])
