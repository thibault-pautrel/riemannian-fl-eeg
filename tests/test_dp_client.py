"""End-to-end tests of the private client and of the DP setup.

Covers rfl/federated/privacy/dp_client.py and rfl/federated/privacy/orchestration.py on top of
the real FLClient / FLServer stack, with synthetic data (no MOABB download):
  * with DP off the private client reproduces the non-private client bit for bit,
  * client i runs local_epochs * floor(m_i / B) <= tau steps of exactly B records,
  * batch normalisation is refused under DP,
  * setup_dp reports a certified budget at the fixed C = 1, which does not depend on the
    aggregation rule, and PriRFed's accounting needs more noise for the same target,
  * private rounds with ProjAvg and RetAvg keep the BiMap weight on the Stiefel manifold.
"""
import math

import numpy as np
import pytest
import torch

from rfl.federated.client import FLClient
from rfl.federated.fl_config import FLConfig
from rfl.federated.privacy.dp_client import DPFLClient
from rfl.federated.privacy.orchestration import client_dataset_sizes, setup_dp
from rfl.models.eegnet import create_eegnet
from rfl.utils.reproducibility import set_seed

CRIT = torch.nn.CrossEntropyLoss()
CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
#  Builders
# --------------------------------------------------------------------------- #
def _config(model_type="eegnet", **kw):
    base = dict(experiment_name="t", run_name="t", dataset="synthetic", subjects=[],
                data_type="raw" if model_type == "eegnet" else "spd",
                model_type=model_type, batch_size=8, learning_rate=1e-3,
                fl_rounds=1, fl_local_epochs=2, device="cpu", seed=0, data_split_seed=0)
    if model_type == "spdnet":
        base["spdnet_hidden_dim"] = 6
    base.update(kw)
    return FLConfig(**base)


def _set(cfg, **kw):
    """Write runtime fields (normally set by setup_dp) after validation."""
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _raw_splits(n=48, ch=8, T=64, k=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, ch, T).astype("float32")
    y = rng.randint(0, k, size=n).astype("int64")
    a, b = int(0.7 * n), int(0.85 * n)
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


def _spd_splits(n=60, d=12, k=4, seed=0):
    rng = np.random.RandomState(seed)
    A = rng.randn(n, d, d).astype("float32")
    X = np.einsum("nij,nkj->nik", A, A) + 0.1 * np.eye(d, dtype="float32")[None]
    y = rng.randint(0, k, size=n).astype("int64")
    a, b = int(0.7 * n), int(0.85 * n)
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


def _client(cls, cfg, splits, cid=0):
    (Xtr, ytr), (Xv, yv), (Xte, yte) = splits
    c = cls(client_id=cid, subjects=[cid], dataset_name="synthetic", config=cfg,
            device=CPU, use_fedbn=True)
    c.prepare_data_from_splits(Xtr, ytr, Xv, yv, Xte, yte, verbose=False)
    return c


def _eegnet():
    return create_eegnet(n_classes=3, n_channels=8, n_timepoints=64, sampling_rate=128)


def _spdnet(seed=0):
    pytest.importorskip("yetanotherspdnet")
    from rfl.models.spdnet import create_spdnet
    return create_spdnet(n_classes=4, n_channels=12, hidden_dim=6, seed=seed)


def _train(cls, cfg, model_fn, splits, train_seed, is_spdnet=False):
    set_seed(0, warn=False)
    model = model_fn()
    c = _client(cls, cfg, splits)
    c.set_model(model)
    set_seed(train_seed, warn=False)
    return c, c.train_local(2, CRIT, 1e-3, is_spdnet=is_spdnet)


# --------------------------------------------------------------------------- #
#  DPFLClient
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_type", ["eegnet", "spdnet"])
def test_dp_off_is_bit_for_bit(model_type):
    is_spd = model_type == "spdnet"
    splits = _spd_splits() if is_spd else _raw_splits()
    model_fn = _spdnet if is_spd else _eegnet
    _, (sd0, _) = _train(FLClient, _config(model_type), model_fn, splits, 123,
                         is_spdnet=is_spd)
    _, (sd1, _) = _train(DPFLClient, _config(model_type), model_fn, splits, 123,
                         is_spdnet=is_spd)
    assert set(sd0) == set(sd1)
    for k in sd0:
        assert torch.equal(sd0[k], sd1[k]), k


def test_private_round_runs_and_differs():
    splits = _raw_splits()
    _, (sd_ref, _) = _train(FLClient, _config(), _eegnet, splits, 7)
    cfg = _set(_config(dp_enabled=True), dp_clip_C=1.0, dp_sigma=0.5)
    _, (sd_dp, metrics) = _train(DPFLClient, cfg, _eegnet, splits, 7)
    assert metrics["dp_local_steps"] == 2 * (33 // 8)
    assert all(torch.isfinite(v).all() for v in sd_dp.values())
    assert any(not torch.equal(sd_dp[k], sd_ref[k]) for k in sd_ref
               if sd_ref[k].dtype.is_floating_point)


def test_floor_steps_of_exactly_B_records(monkeypatch):
    splits = _raw_splits(n=50)                 # 35 training records, B = 8
    cfg = _set(_config(dp_enabled=True), dp_clip_C=1.0, dp_sigma=0.1)
    sizes = []
    orig = DPFLClient._sample_wor_batch

    def spy(self, B):
        x, y = orig(self, B)
        sizes.append(len(y))
        return x, y

    monkeypatch.setattr(DPFLClient, "_sample_wor_batch", spy)
    _, (_, metrics) = _train(DPFLClient, cfg, _eegnet, splits, 1)
    assert metrics["dp_local_steps"] == 2 * (35 // 8) == len(sizes)
    assert set(sizes) == {8}


def test_batch_larger_than_client_is_refused():
    cfg = _set(_config(dp_enabled=True, batch_size=64), dp_clip_C=1.0, dp_sigma=0.1)
    with pytest.raises(ValueError):
        _train(DPFLClient, cfg, _eegnet, _raw_splits(), 1)


def test_batchnorm_is_refused_under_dp():
    with pytest.raises(ValueError):
        _config(dp_enabled=True, eegnet_norm_type="batch")
    cfg = _set(_config(dp_enabled=True), dp_clip_C=1.0, dp_sigma=0.1)

    def eegnet_bn():
        return create_eegnet(n_classes=3, n_channels=8, n_timepoints=64,
                             sampling_rate=128, norm_type="batch")

    with pytest.raises(ValueError):
        _train(DPFLClient, cfg, eegnet_bn, _raw_splits(), 1)


# --------------------------------------------------------------------------- #
#  setup_dp
# --------------------------------------------------------------------------- #
def _spd_clients(cfg, sizes=(56, 64, 48)):
    clients = []
    for cid, n in enumerate(sizes):
        clients.append(_client(DPFLClient, cfg, _spd_splits(n=n, seed=cid), cid=cid))
    return clients


def _spd_dp_config(agg="projavg", **kw):
    base = dict(fl_rounds=2, fl_local_epochs=2, fl_aggregation=agg, dp_enabled=True,
                dp_target_eps=5.0, dp_target_delta=1e-5, dp_accountant_h=0.05)
    base.update(kw)
    return _config("spdnet", **base)


def test_setup_dp_budget():
    set_seed(0, warn=False)
    cfg = _spd_dp_config()
    clients = _spd_clients(cfg)
    gm = _spdnet()
    for c in clients:
        c.set_model(gm)
    m_star, m_max = client_dataset_sizes(clients)
    budget = setup_dp(cfg, clients, gm, CRIT, CPU, verbose=False)

    assert (budget["m_star"], budget["m_max"]) == (m_star, m_max)
    assert budget["q"] == pytest.approx(cfg.batch_size / m_star)
    assert budget["tau"] == cfg.fl_local_epochs * math.ceil(m_max / cfg.batch_size)
    assert budget["n_steps"] == cfg.fl_rounds * budget["tau"]
    assert budget["mu0"] == pytest.approx(2 * budget["C"] / (cfg.batch_size * budget["sigma"]))
    assert 0 < budget["realized_eps"] <= 5.0 + 1e-9
    assert budget["certify"]["min_gap"] >= -1e-11
    assert budget["C"] == 1.0                       # fixed a priori, as in the paper
    assert budget["sigma_source"] == "calibrated" and "prirfed" not in budget
    assert (cfg.dp_clip_C, cfg.dp_sigma) == (budget["C"], budget["sigma"])
    # Every client runs at most the tau steps the accountant composes.
    for c in clients:
        steps = c.train_local(cfg.fl_local_epochs, CRIT, 1e-3)[1]["dp_local_steps"]
        assert steps == cfg.fl_local_epochs * (c.n_samples // cfg.batch_size)
        assert steps <= budget["tau"]


def test_setup_dp_prirfed_accounting_needs_more_noise():
    budgets = {}
    for accounting in ("ours", "prirfed"):
        set_seed(0, warn=False)
        cfg = _spd_dp_config(dp_accounting=accounting)
        clients = _spd_clients(cfg)
        gm = _spdnet()
        for c in clients:
            c.set_model(gm)
        budgets[accounting] = setup_dp(cfg, clients, gm, CRIT, CPU, verbose=False)
    prirfed = budgets["prirfed"]
    assert prirfed["sigma_source"] == "prirfed_thm7" and "prirfed" not in prirfed
    assert prirfed["sigma"] > budgets["ours"]["sigma"]
    # At PriRFed's sigma our accountant certifies a smaller epsilon than the target.
    assert prirfed["realized_eps_ours"] < 5.0


def test_clip_constant_must_be_finite_and_positive():
    for C in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError):
            _spd_dp_config(dp_clip_C=C)


def test_budget_does_not_depend_on_the_aggregation_rule():
    budgets = {}
    for agg in ("projavg", "rlavg", "retavg"):
        set_seed(0, warn=False)
        cfg = _spd_dp_config(agg)
        clients = _spd_clients(cfg)
        gm = _spdnet()
        for c in clients:
            c.set_model(gm)
        budgets[agg] = setup_dp(cfg, clients, gm, CRIT, CPU, verbose=False)
    ref = budgets["projavg"]
    for agg in ("rlavg", "retavg"):
        for key in ("C", "sigma", "mu0", "q", "realized_eps"):
            assert budgets[agg][key] == pytest.approx(ref[key], rel=1e-9), (agg, key)
        assert budgets[agg]["n_steps"] == ref["n_steps"]


# --------------------------------------------------------------------------- #
#  Private federated rounds
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("agg", ["projavg", "retavg"])
def test_private_rounds_keep_bimap_on_stiefel(agg):
    from rfl.federated.server import FLServer, _client_class

    set_seed(0, warn=False)
    cfg = _spd_dp_config(agg)
    assert _client_class(cfg) is DPFLClient
    clients = _spd_clients(cfg)
    gm = _spdnet()
    for c in clients:
        c.set_model(gm)
    setup_dp(cfg, clients, gm, CRIT, CPU, verbose=False)

    server = FLServer(global_model=gm, config=cfg, device=CPU)
    for r in (1, 2):
        m = server.run_round(round_num=r, clients=clients, criterion=CRIT, verbose=False)
        assert math.isfinite(m["global_val_acc"]) and math.isfinite(m["global_test_acc"])
        assert m["agg_seconds"] >= 0.0
    W = server.global_model.bimap.weight.detach()
    assert torch.norm(W.T @ W - torch.eye(W.shape[1])) < 1e-3
    assert len(server.round_history["agg_seconds"]) == 2
