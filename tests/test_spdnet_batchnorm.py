"""SPD batch norm in SPDNet: layers, options, FedBN scope and refusal under DP."""
import numpy as np
import pytest
import torch

pytest.importorskip("yetanotherspdnet")

from rfl.federated.aggregation import identify_batchnorm_keys          # noqa: E402
from rfl.federated.param_groups import (identify_bn_buffer_names,     # noqa: E402
                                    identify_bn_param_names)
from rfl.models.spdnet import create_spdnet, validate_bn_options      # noqa: E402


def _spd(n=12, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, d, 3 * d, generator=g)
    return A @ A.transpose(-1, -2) / (3 * d) + 1e-3 * torch.eye(d)


@pytest.mark.parametrize("kw", [dict(), dict(bn_type="mean_var_scalar"),
                                dict(bn_mean_type="affine_invariant"),
                                dict(bn_norm_and_bias_mode="parallel_transport"),
                                dict(bn_position="before_reeig")])
def test_forward_backward_and_float64_layers(kw):
    m = create_spdnet(n_classes=3, n_channels=8, hidden_dim=4, seed=0,
                      use_batch_norm=True, **kw)
    out = m(_spd())
    assert out.shape == (12, 3) and out.dtype == torch.float32
    out.sum().backward()
    bn = m.batch_norms()[0]
    assert all(p.dtype == torch.float64 for p in bn.parameters())
    assert any(p.grad is not None for p in bn.parameters())


def test_one_layer_per_block_with_its_own_keys():
    m = create_spdnet(n_classes=2, n_channels=8, hidden_dims=[6, 4],
                      epsilons=[1e-4, 1e-3], seed=0, use_batch_norm=True)
    assert len(m.batch_norms()) == 2
    keys = set(m.state_dict())
    assert {"batch_norm.running_mean", "batch_norm_2.running_mean"} <= keys
    assert m.batch_norm.running_mean.shape == (6, 6)
    assert m.batch_norm_2.running_mean.shape == (4, 4)


def test_without_batch_norm_the_state_dict_is_unchanged():
    m = create_spdnet(n_classes=2, n_channels=8, hidden_dim=4, seed=0)
    assert set(m.state_dict()) == {"bimap.weight", "classifier.weight",
                                   "classifier.bias"}
    assert m.batch_norms() == []


def test_running_mean_moves_in_train_mode_only():
    m = create_spdnet(n_classes=2, n_channels=8, hidden_dim=4, seed=0,
                      use_batch_norm=True)
    r0 = m.batch_norm.running_mean.clone()
    m.eval()
    m(_spd())
    assert torch.equal(m.batch_norm.running_mean, r0)
    m.train()
    m(_spd())
    assert not torch.equal(m.batch_norm.running_mean, r0)


def test_position_matters_when_reeig_is_active():
    a = create_spdnet(n_classes=2, n_channels=8, hidden_dim=4, seed=0, epsilon=0.5,
                      use_batch_norm=True, bn_position="after_reeig")
    b = create_spdnet(n_classes=2, n_channels=8, hidden_dim=4, seed=0, epsilon=0.5,
                      use_batch_norm=True, bn_position="before_reeig")
    a.train(), b.train()
    x = _spd()
    assert not torch.allclose(a(x), b(x))


@pytest.mark.parametrize("kw", [dict(bn_type="other"), dict(bn_mean_type="other"),
                                dict(bn_momentum=0.0), dict(bn_position="end"),
                                dict(bn_norm_and_bias_mode="parallel_transport",
                                     bn_type="mean_var_scalar"),
                                dict(bn_norm_and_bias_mode="parallel_transport",
                                     bn_mean_type="bures_wasserstein")])
def test_invalid_options_raise(kw):
    opts = dict(bn_type="mean_only", bn_mean_type="geometric_arithmetic_harmonic",
                bn_momentum=0.1, bn_norm_and_bias_mode="whitening",
                bn_position="after_reeig")
    opts.update(kw)
    with pytest.raises(ValueError):
        validate_bn_options(**opts)


def test_module_and_state_dict_rules_agree():
    """FedBN scope: the module tree and the state-dict keys select the same keys."""
    m = create_spdnet(n_classes=2, n_channels=8, hidden_dims=[6, 4],
                      epsilons=[1e-4, 1e-3], seed=0, use_batch_norm=True,
                      bn_type="mean_var_scalar")
    by_module = identify_bn_param_names(m) | identify_bn_buffer_names(m)
    assert by_module == identify_batchnorm_keys(m.state_dict())
    assert any("Covbias" in k for k in by_module)
    assert not any("bimap" in k or "classifier" in k for k in by_module)


def test_config_passes_the_options_and_dp_refuses_them():
    from rfl.federated.fl_config import FLConfig
    base = dict(experiment_name="t", run_name="t", dataset="synthetic", subjects=[],
                data_type="spd", model_type="spdnet", device="cpu")
    cfg = FLConfig(**base, spdnet_use_batch_norm=True, spdnet_bn_type="mean_var_scalar",
                   spdnet_bn_position="before_reeig")
    p = cfg.get_model_params()
    assert p["use_batch_norm"] and p["bn_type"] == "mean_var_scalar"
    assert p["bn_position"] == "before_reeig" and p["bn_momentum"] == 0.1
    assert "bn_type" not in FLConfig(**base).get_model_params()
    with pytest.raises(ValueError):
        FLConfig(**base, spdnet_use_batch_norm=True, spdnet_bn_momentum=2.0)
    with pytest.raises(ValueError):
        FLConfig(**base, spdnet_use_batch_norm=True, dp_enabled=True)


def test_dp_mechanism_refuses_the_layer():
    from rfl.federated.privacy.dp_grad import assert_dp_safe_norm_layers
    m = create_spdnet(n_classes=2, n_channels=8, hidden_dim=4, seed=0,
                      use_batch_norm=True)
    with pytest.raises(ValueError):
        assert_dp_safe_norm_layers(m)


def test_single_record_last_batch_is_dropped():
    from rfl.data.datasets import make_loader
    ds = torch.utils.data.TensorDataset(torch.zeros(33, 2), torch.zeros(33))
    assert len(make_loader(ds, 16, shuffle=True, avoid_single_batch=True)) == 2
    assert len(make_loader(ds, 16, shuffle=True)) == 3
    ds2 = torch.utils.data.TensorDataset(torch.zeros(34, 2), torch.zeros(34))
    assert len(make_loader(ds2, 16, shuffle=True, avoid_single_batch=True)) == 3


def test_federated_runner_with_batch_norm(tmp_path):
    import json

    from rfl.data.synthetic import generate_synthetic_spd, save_dataset
    from rfl.experiments.run_federated_experiment import main
    data = generate_synthetic_spd(n_clients=3, samples_per_client=60, n_channels=6,
                                  n_classes=2, mode="iid", signal_strength=1.0, seed=0)
    save_dataset(data, str(tmp_path / "s.npz"))
    main(["--synthetic-clients", str(tmp_path / "s.npz"), "--hidden-dim", "3",
          "--rounds", "2", "--batch-size", "8", "--device", "cpu",
          "--use-batch-norm", "--local-fc", "--save-dir", str(tmp_path / "r")])
    r = json.loads((tmp_path / "r" / "fl_results.json").read_text())
    assert r["config"]["spdnet_use_batch_norm"]
    assert np.isfinite(r["final_metrics"]["global_test_loss"])


# --------------------------------------------------------------------------- #
#  ReEig floors per block
# --------------------------------------------------------------------------- #
def test_each_block_has_its_own_reeig_floor():
    m = create_spdnet(n_classes=2, n_channels=8, hidden_dims="6,4,3",
                      epsilons="1e-4,1e-3,1e-2", seed=0)
    assert [m.reig.eps, m.reig_2.eps, m.reig_3.eps] == [1e-4, 1e-3, 1e-2]


def test_equal_floors_refused_without_batch_norm_allowed_with_it():
    from rfl.federated.fl_config import FLConfig
    with pytest.raises(ValueError):
        create_spdnet(n_classes=2, n_channels=8, hidden_dims=[6, 4],
                      epsilons=[1e-3, 1e-3], seed=0)
    m = create_spdnet(n_classes=2, n_channels=8, hidden_dims=[6, 4],
                      epsilons=[1e-3, 1e-3], seed=0, use_batch_norm=True)
    assert m.reig_2.eps == 1e-3
    base = dict(experiment_name="t", run_name="t", dataset="synthetic", subjects=[],
                data_type="spd", model_type="spdnet", device="cpu",
                spdnet_hidden_dims="6,4", spdnet_epsilons="1e-3,1e-3")
    with pytest.raises(ValueError):
        FLConfig(**base)
    FLConfig(**base, spdnet_use_batch_norm=True)


def test_reeig_activity_detects_an_identity_block():
    """Without batch norm, block 2 with eps_2 = eps_1 never clamps (Oswald & Massart)."""
    m = create_spdnet(n_classes=2, n_channels=8, hidden_dims=[6, 4],
                      epsilons=[0.05, 0.05], seed=0, strict_epsilons=False)
    m.track_clipping = True
    m.eval()
    with torch.no_grad():
        m(_spd(64))
    act = m.reeig_activity()
    assert act[0] > 0 and act[1] < 1e-6


# --------------------------------------------------------------------------- #
#  FedBN on and off
# --------------------------------------------------------------------------- #
def _federation(model, fedbn, n_clients=3):
    import torch.nn as nn

    from rfl.federated.fl_config import FLConfig
    from rfl.federated.server import FLServer, create_fl_clients_synthetic
    rng = np.random.RandomState(0)
    raw = model == "eegnet"
    splits = []
    for _ in range(n_clients):
        d = {}
        for name, n in (("train", 32), ("val", 8), ("test", 8)):
            if raw:
                d[f"X_{name}"] = rng.randn(n, 4, 64).astype(np.float32)
            else:
                A = rng.randn(n, 6, 18)
                d[f"X_{name}"] = (A @ A.transpose(0, 2, 1) / 18
                                  + 1e-3 * np.eye(6)).astype(np.float32)
            d[f"y_{name}"] = (np.arange(n) % 2).astype(np.int64)
        splits.append(d)
    kw = (dict(data_type="raw", eegnet_norm_type="batch", eegnet_dropout=0.0) if raw
          else dict(data_type="spd", spdnet_use_batch_norm=True, spdnet_hidden_dim=3,
                    fl_aggregation="projavg"))
    cfg = FLConfig(experiment_name="t", run_name="t", dataset="synthetic", subjects=[],
                   model_type=model, batch_size=8, learning_rate=1e-2, fl_rounds=2,
                   fl_local_epochs=1, device="cpu", seed=0, fl_fedbn=fedbn, **kw)
    clients = create_fl_clients_synthetic(splits, cfg, torch.device("cpu"), verbose=False)
    if raw:
        from rfl.models.eegnet import create_eegnet
        net = create_eegnet(n_classes=2, n_channels=4, n_timepoints=64,
                            sampling_rate=None, **cfg.get_model_params())
    else:
        net = create_spdnet(n_classes=2, n_channels=6, seed=0, **cfg.get_model_params())
    for c in clients:
        c.set_model(net)
    server = FLServer(net, cfg, torch.device("cpu"))
    for r in (1, 2):
        server.run_round(r, clients, nn.CrossEntropyLoss(), verbose=False)
    return clients, server


@pytest.mark.parametrize("model", ["eegnet", "spdnet"])
@pytest.mark.parametrize("fedbn", [True, False])
def test_fedbn_switch(model, fedbn):
    clients, server = _federation(model, fedbn)
    m = clients[0].local_model
    keys = sorted(identify_bn_param_names(m) | identify_bn_buffer_names(m))
    assert keys
    for k in keys:
        if not torch.is_floating_point(server.global_state_dict[k]):
            continue                                  # counters keep client 0's value
        same = [torch.equal(c.local_model.state_dict()[k], server.global_state_dict[k])
                for c in clients]
        if fedbn:
            assert not all(same), k                   # personal, never aggregated
        else:
            assert all(same), k                       # shared: every client = global
    from rfl.analysis.results import arch_name
    assert ("shared" in arch_name(server.config.to_dict())) != fedbn


# --------------------------------------------------------------------------- #
#  Shared running means (--no-fedbn)
# --------------------------------------------------------------------------- #
def test_layers_are_those_of_the_library():
    from yetanotherspdnet.nn import BatchNormSPDMean, BatchNormSPDMeanScalarVariance
    for bn_type, cls in (("mean_only", BatchNormSPDMean),
                         ("mean_var_scalar", BatchNormSPDMeanScalarVariance)):
        m = create_spdnet(n_classes=2, n_channels=8, hidden_dims=[6, 4],
                          epsilons=[1e-3, 1e-2], seed=0, use_batch_norm=True,
                          bn_type=bn_type)
        other = (BatchNormSPDMeanScalarVariance if cls is BatchNormSPDMean
                 else None)
        assert all(isinstance(bn, cls) for bn in m.batch_norms())
        if other is not None:
            assert not any(isinstance(bn, other) for bn in m.batch_norms())


@pytest.mark.parametrize("mean_type", ["arithmetic", "harmonic", "log_euclidean",
                                       "affine_invariant",
                                       "geometric_arithmetic_harmonic"])
def test_weighted_spd_mean(mean_type):
    from rfl.models.spd_batchnorm import _exp, _inv, _invsqrt, _log, weighted_spd_mean
    mats = [_spd(1, 5, seed=s)[0].double() for s in range(3)]
    w = [1.0, 2.0, 3.0]
    G = weighted_spd_mean(mats, w, mean_type)
    assert torch.all(torch.linalg.eigvalsh(G) > 0)
    same = weighted_spd_mean([mats[0]] * 3, w, mean_type)
    assert torch.allclose(same, mats[0], atol=1e-8)            # idempotent
    wn = torch.tensor(w, dtype=torch.float64) / 6
    S = torch.stack(mats)
    if mean_type == "arithmetic":
        assert torch.allclose(G, (wn.view(-1, 1, 1) * S).sum(0))
    if mean_type == "harmonic":
        assert torch.allclose(_inv(G), (wn.view(-1, 1, 1) * _inv(S)).sum(0), atol=1e-8)
    if mean_type == "log_euclidean":
        assert torch.allclose(_log(G), (wn.view(-1, 1, 1) * _log(S)).sum(0), atol=1e-8)
    if mean_type == "affine_invariant":                         # Karcher condition
        Gi = _invsqrt(G)
        T = (wn.view(-1, 1, 1) * _log(Gi @ S @ Gi)).sum(0)
        assert float(T.norm()) < 1e-8
    del _exp


def test_unsupported_shared_layers_are_refused():
    from rfl.federated.fl_config import FLConfig
    base = dict(experiment_name="t", run_name="t", dataset="synthetic", subjects=[],
                data_type="spd", model_type="spdnet", device="cpu",
                spdnet_use_batch_norm=True, fl_fedbn=False)
    FLConfig(**base)
    for kw in (dict(spdnet_bn_type="mean_var_scalar"),
               dict(spdnet_bn_mean_type="bures_wasserstein"),
               dict(spdnet_bn_mean_type="adaptive_geometric_arithmetic_harmonic")):
        with pytest.raises(ValueError):
            FLConfig(**base, **kw)
        FLConfig(**{**base, "fl_fedbn": True}, **kw)            # fine under FedBN


def test_shared_spd_running_means_are_combined_in_their_geometry():
    """With --no-fedbn, the global running mean is the GAH combination of the clients'
    running means, not their Euclidean average."""
    import copy

    import torch.nn as nn

    from rfl.federated.fl_config import FLConfig
    from rfl.federated.server import FLServer, create_fl_clients_synthetic
    from rfl.models.spd_batchnorm import weighted_spd_mean
    rng = np.random.RandomState(0)
    splits = []
    for c in range(3):
        d = {}
        for name, n in (("train", 32), ("val", 8), ("test", 8)):
            A = rng.randn(n, 6, 18) * (1 + c)                   # client-specific scale
            d[f"X_{name}"] = (A @ A.transpose(0, 2, 1) / 18
                              + 1e-3 * np.eye(6)).astype(np.float32)
            d[f"y_{name}"] = (np.arange(n) % 2).astype(np.int64)
        splits.append(d)
    cfg = FLConfig(experiment_name="t", run_name="t", dataset="synthetic", subjects=[],
                   model_type="spdnet", data_type="spd", batch_size=8,
                   learning_rate=1e-2, fl_rounds=1, fl_local_epochs=1, device="cpu",
                   seed=0, spdnet_hidden_dim=3, spdnet_use_batch_norm=True,
                   fl_fedbn=False,
                   fl_aggregation="projavg")
    clients = create_fl_clients_synthetic(splits, cfg, torch.device("cpu"), verbose=False)
    net = create_spdnet(n_classes=2, n_channels=6, seed=0, **cfg.get_model_params())
    for c in clients:
        c.set_model(net)
    server = FLServer(net, cfg, torch.device("cpu"))
    uploads = [c.train_local(1, nn.CrossEntropyLoss(), 1e-2)[0] for c in clients]
    agg = server.aggregate_updates(copy.deepcopy(uploads),
                                   [{"bimap.weight": c.local_model.bimap.weight.detach()}
                                    for c in clients], [1.0, 1.0, 1.0])
    R = [u["batch_norm.running_mean"] for u in uploads]
    euclid = sum(R) / 3
    expected = weighted_spd_mean(R, [1, 1, 1], "geometric_arithmetic_harmonic")
    assert torch.allclose(agg["batch_norm.running_mean"], expected, atol=1e-8)
    assert not torch.allclose(agg["batch_norm.running_mean"], euclid, atol=1e-6)


def test_karcher_mean_converges_on_widely_spread_matrices():
    from rfl.models.spd_batchnorm import _invsqrt, _log, weighted_spd_mean
    g = torch.Generator().manual_seed(0)
    mats = []
    for _ in range(4):
        Q, _ = torch.linalg.qr(torch.randn(5, 5, generator=g, dtype=torch.float64))
        lam = torch.exp(3.0 * torch.randn(5, generator=g, dtype=torch.float64))
        mats.append(Q @ torch.diag(lam) @ Q.T)
    w = [0.1, 0.2, 0.3, 0.4]
    G = weighted_spd_mean(mats, w, "affine_invariant")
    Gi = _invsqrt(G)
    T = sum(wi * _log(Gi @ M @ Gi) for wi, M in zip(w, mats))
    assert float(T.norm()) < 1e-8


def test_option_combinations_without_an_implementation_are_refused():
    from rfl.federated.fl_config import FLConfig
    base = dict(experiment_name="t", run_name="t", dataset="synthetic", subjects=[],
                data_type="spd", model_type="spdnet", device="cpu",
                spdnet_use_batch_norm=True)
    FLConfig(**base, spdnet_bn_type="mean_var_scalar")
    for kw in (dict(spdnet_bn_type="mean_var_scalar",
                    spdnet_bn_mean_type="adaptive_geometric_arithmetic_harmonic"),
               dict(batch_size=1)):
        with pytest.raises(ValueError):
            FLConfig(**base, **kw)
