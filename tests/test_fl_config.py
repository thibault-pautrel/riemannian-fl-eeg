"""FLConfig validation, hashing and client partitioning."""

import pytest

from rfl.federated.fl_config import FLConfig, create_client_subject_mapping


def _cfg(**kw):
    base = dict(experiment_name="t", run_name="t", dataset="synthetic", subjects=[],
                data_type="spd", model_type="spdnet", device="cpu")
    base.update(kw)
    return FLConfig(**base)


def test_subject_mapping_last_client_is_smaller():
    assert create_client_subject_mapping([1, 2, 3, 4, 5, 6, 7, 8], 3) == \
        [[1, 2, 3], [4, 5, 6], [7, 8]]
    assert create_client_subject_mapping([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
    with pytest.raises(ValueError):
        create_client_subject_mapping([1, 2], 0)


def test_defaults_are_valid():
    cfg = _cfg()
    # The defaults are those of the runner (tests/test_cli_checks.py).
    assert cfg.fl_aggregation == "projavg" and not cfg.dp_enabled
    assert _cfg(model_type="eegnet", data_type="raw").fl_aggregation == "fedavg"


@pytest.mark.parametrize("kw", [
    dict(fl_client_fraction=0.0),
    dict(lr_schedule="linear"),
    dict(dp_enabled=True, dp_target_delta=1.0),
    dict(dp_enabled=True, dp_accounting="other"),
    dict(dp_enabled=True, spdnet_use_batch_norm=True),
    dict(dp_enabled=True, model_type="eegnet", data_type="raw", eegnet_norm_type="batch"),
    dict(dp_enabled=True, dp_clip_C=float("inf")),
    dict(dp_enabled=True, dp_clip_C=0.0),
    dict(dp_enabled=True, model_type="eegnet", data_type="raw",
         normalization_type="global"),
    dict(fl_aggregation="other"),
    dict(spdnet_hidden_dims="16,8", spdnet_epsilons="1e-2,1e-2"),
    dict(spdnet_hidden_dims="16,8", spdnet_epsilons="1e-4"),
])
def test_invalid_combinations_raise(kw):
    with pytest.raises(ValueError):
        _cfg(**kw)


def test_hash_ignores_dp_sigma_and_run_name():
    a = _cfg(dp_enabled=True)
    b = _cfg(dp_enabled=True, run_name="other")
    b.dp_sigma = 3.0
    assert a.get_hash() == b.get_hash()
    assert a.get_hash() != _cfg(dp_enabled=True, dp_target_eps=5.0).get_hash()


def test_dp_defaults():
    cfg = _cfg(dp_enabled=True)
    assert cfg.dp_clip_C == 1.0 and cfg.dp_target_delta == 1e-5
    assert cfg.eegnet_norm_type == "none"
    assert not cfg.fl_local_fc and not cfg.head_refit_enabled
    assert _cfg(fl_local_fc=True).get_hash() != cfg.get_hash()


def test_local_only_needs_full_participation():
    with pytest.raises(ValueError, match="fl_client_fraction"):
        _cfg(fl_local_only=True, fl_client_fraction=0.5)


def test_spdnet_takes_a_riemannian_rule_and_eegnet_fedavg():
    with pytest.raises(ValueError, match="projavg, rlavg or retavg"):
        _cfg(fl_aggregation="fedavg")
    for agg in ("projavg", "rlavg", "retavg"):
        _cfg(fl_aggregation=agg)
        with pytest.raises(ValueError, match="Stiefel"):
            _cfg(model_type="eegnet", data_type="raw", fl_aggregation=agg)


def test_dp_and_frozen_trunk_need_the_projection():
    with pytest.raises(ValueError, match="--stiefel projection"):
        FLConfig(dataset="BNCI2014_001", dp_enabled=True, spdnet_stiefel="trivialization")
    with pytest.raises(ValueError, match="--stiefel projection"):
        FLConfig(dataset="BNCI2014_001", freeze_trunk=True, fl_local_fc=True,
                 spdnet_stiefel="trivialization")
    FLConfig(dataset="BNCI2014_001", spdnet_stiefel="trivialization", optimizer="sgd",
             sgd_momentum=0.9)
    with pytest.raises(ValueError, match="sgd_momentum"):
        FLConfig(dataset="BNCI2014_001", sgd_momentum=0.9)
