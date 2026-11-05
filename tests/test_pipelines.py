"""End-to-end runs of the two entry points on a tiny synthetic SPD federation.

These are smoke tests: they check that a run completes, writes its result files, and
that the reporting tools read them. Accuracy is not checked.
"""
import json
import os
import sys

import pytest

pytest.importorskip("yetanotherspdnet")

from rfl.analysis.results import load_runs  # noqa: E402
from rfl.data.synthetic import generate_synthetic_spd, save_dataset  # noqa: E402
from rfl.experiments import run_centralized_experiment, run_federated_experiment  # noqa: E402

COMMON = ["--model", "spdnet", "--hidden-dim", "4", "--batch-size", "16",
          "--learning-rate", "0.05", "--device", "cpu", "--seed", "1"]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "synth.npz")
    save_dataset(generate_synthetic_spd(n_clients=3, samples_per_client=60,
                                        n_channels=8, n_classes=2, seed=0), path)
    return path


def _run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module.__name__] + argv)
    module.main()


def test_federated_private_and_centralized_runs(synth, tmp_path, monkeypatch):
    _run(run_federated_experiment,
         ["--synthetic-clients", synth, "--aggregation", "projavg", "--rounds", "2",
          "--local-epochs", "1", "--save-dir", str(tmp_path / "fed")] + COMMON, monkeypatch)
    _run(run_federated_experiment,
         ["--synthetic-clients", synth, "--aggregation", "rlavg", "--rounds", "2",
          "--local-epochs", "1", "--dp-enabled", "--dp-target-eps", "10",
          "--dp-clip-C", "1.0", "--save-dir", str(tmp_path / "dp")] + COMMON, monkeypatch)
    _run(run_centralized_experiment,
         ["--synthetic", synth, "--epochs", "2", "--save-dir", str(tmp_path / "central")]
         + COMMON, monkeypatch)

    with open(tmp_path / "dp" / "fl_results.json") as f:
        dp = json.load(f)
    assert dp["dp_budget"]["realized_eps"] <= 10.0 + 1e-9
    assert len(dp["round_history"]["round"]) == 2
    assert os.path.exists(tmp_path / "fed" / "global_model_final.pt")

    df = load_runs(str(tmp_path))
    assert len(df) == 3
    assert df["test_acc"].between(0, 1).all()
    priv = df[df["eps"] == 10.0]
    assert len(priv) == 1 and priv["aggregation"].iloc[0] == "rlavg"


def test_sgd_and_trivialization_run_both_pipelines(synth, tmp_path, monkeypatch):
    """SGD with the library orthogonal parametrization, federated (batch norm, FedPer)
    and centralised. The saved global BiMap weight is orthonormal and stored as W."""
    import torch
    extra = ["--optimizer", "sgd", "--momentum", "0.9", "--stiefel", "trivialization"]
    _run(run_federated_experiment,
         ["--synthetic-clients", synth, "--rounds", "2", "--use-batch-norm", "--local-fc",
          "--save-dir", str(tmp_path / "fed")] + COMMON + extra, monkeypatch)
    _run(run_centralized_experiment,
         ["--synthetic-clients", synth, "--epochs", "2",
          "--save-dir", str(tmp_path / "cen")] + COMMON + extra, monkeypatch)
    g = torch.load(tmp_path / "fed" / "global_model_final.pt", weights_only=False)
    W = g["model_state_dict"]["bimap.weight"].double()
    assert torch.allclose(W.T @ W, torch.eye(W.shape[1], dtype=W.dtype), atol=1e-5)
    assert not any("parametrizations.weight" in k for k in g["model_state_dict"])
    cfg = json.load(open(tmp_path / "fed" / "config.json"))
    assert cfg["optimizer"] == "sgd" and cfg["spdnet_stiefel"] == "trivialization"
