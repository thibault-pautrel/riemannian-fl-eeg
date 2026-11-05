"""Tests for rfl/utils/config.py and rfl/utils/reproducibility.py."""
import json
import random

import numpy as np
import pytest
import torch

from rfl.utils.config import ExperimentConfig
from rfl.utils.reproducibility import get_git_info, log_environment, set_seed


def _cfg(**kw):
    kw.setdefault("run_name", "r")
    return ExperimentConfig(**kw)


def test_hash_ignores_run_name_and_save_paths():
    a = _cfg(run_name="a", save_root="/tmp/x")
    b = _cfg(run_name="b", save_root="/tmp/x")
    assert a.config_hash == b.config_hash and len(a.config_hash) == 8


def test_hash_changes_with_hyperparameters():
    assert _cfg(learning_rate=1e-3).config_hash != _cfg(learning_rate=3e-3).config_hash


def test_hash_follows_protocol_fields():
    base = _cfg()
    assert base.get_hash() == _cfg(selection_metric="val_acc").get_hash()
    assert base.get_hash() != _cfg(selection_metric="val_loss").get_hash()
    assert base.get_hash() == _cfg(eegnet_norm_type="none").get_hash()
    assert base.get_hash() != _cfg(eegnet_norm_type="batch").get_hash()


def test_experiment_dir():
    cfg = _cfg(experiment_name="e", run_name="r", save_root="out")
    assert cfg.experiment_dir.replace("\\", "/") == "out/e/r"


@pytest.mark.parametrize("bad", [
    dict(batch_size=0), dict(learning_rate=0.0), dict(model_type="resnet"),
    dict(normalization_type="robust"), dict(eegnet_norm_type="layer"),
    dict(selection_metric="test_acc"), dict(report_rule="best"),
    dict(val_size=0.5, test_size=0.5), dict(lr_plateau_factor=1.0),
])
def test_invalid_values_raise(bad):
    with pytest.raises(ValueError):
        _cfg(**bad)


def test_model_params_drop_none():
    eeg = _cfg(model_type="eegnet").get_model_params()
    assert "F2" not in eeg and "kern_length" not in eeg and eeg["norm_type"] == "none"
    spd = _cfg(model_type="spdnet", spdnet_hidden_dim=8).get_model_params()
    assert spd == {"hidden_dim": 8, "use_batch_norm": False, "epsilon": 8e-5,
                   "trace_normalize": True, "stiefel": "projection"}               # BNCI2014_001, registry
    syn = _cfg(model_type="spdnet", dataset="synthetic", data_type="spd").get_model_params()
    assert syn == {"use_batch_norm": False, "epsilon": 1e-4, "trace_normalize": True,
                   "stiefel": "projection"}


def test_save_and_load_roundtrip_ignores_unknown_keys(tmp_path):
    cfg = _cfg(learning_rate=3e-3, subjects=[1, 2])
    path = cfg.save(str(tmp_path))
    with open(path) as f:
        d = json.load(f)
    d["unknown_field"] = 0.5                  # not a dataclass field
    with open(path, "w") as f:
        json.dump(d, f)
    back = ExperimentConfig.load(str(tmp_path))
    assert back.learning_rate == 3e-3 and back.subjects == [1, 2]
    assert back.config_hash == cfg.config_hash


def test_load_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ExperimentConfig.load(str(tmp_path))


def test_update_recomputes_hash():
    cfg = _cfg()
    h = cfg.config_hash
    with pytest.warns(UserWarning):
        cfg.update(learning_rate=5e-3, not_a_field=1)
    assert cfg.learning_rate == 5e-3 and cfg.config_hash != h


def test_set_seed_reproduces_all_generators():
    def draw():
        return random.random(), float(np.random.rand()), float(torch.rand(1))
    set_seed(7, deterministic=False, warn=False)
    a = draw()
    set_seed(7, deterministic=False, warn=False)
    assert draw() == a


def test_log_environment_writes_json(tmp_path):
    info = log_environment(str(tmp_path))
    assert (tmp_path / "environment.json").exists()
    assert set(info) == {"environment", "git"}
    assert set(get_git_info()) == {"commit_hash", "branch", "has_uncommitted_changes",
                                   "remote_url"}


def test_hash_values_are_stable():
    """Pinned values: a change here changes the identifiers of new runs."""
    assert _cfg().config_hash == "287cc176"
    assert _cfg(model_type="spdnet", learning_rate=3e-3,
                eegnet_norm_type="none").config_hash == "d5376b85"
