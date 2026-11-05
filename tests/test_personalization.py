"""Personalisation, alone and combined with differential privacy.

The checks that matter for soundness under DP:

* every parameter that trains during the rounds is inside the privatised vector, so
  no raw-record gradient reaches the model (`federated_param_groups`);
* a personal head is never uploaded, whatever its mode;
* a frozen head or trunk does not move during the rounds;
* the head re-fit after training changes the head only, and sends nothing.
"""
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from rfl.federated.fl_config import FLConfig
from rfl.federated.server import FLServer, create_fl_clients_synthetic

CRIT = nn.CrossEntropyLoss()


# --------------------------------------------------------------------------- #
#  Synthetic federation
# --------------------------------------------------------------------------- #
def _spd(rng, n, d=6):
    A = rng.randn(n, d, 3 * d)
    return (A @ A.transpose(0, 2, 1) / (3 * d) + 1e-3 * np.eye(d)).astype(np.float32)


def _raw(rng, n, c=4, t=64):
    return rng.randn(n, c, t).astype(np.float32)


def _splits(n_clients, make_x, n_classes=2, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_clients):
        d = {}
        for name, n in (("train", 32), ("val", 8), ("test", 8)):
            d[f"X_{name}"] = make_x(rng, n)
            d[f"y_{name}"] = (np.arange(n) % n_classes).astype(np.int64)
        out.append(d)
    return out


def _cfg(model="spdnet", **kw):
    base = dict(experiment_name="t", run_name="t", dataset="synthetic", subjects=[],
                data_type="spd" if model == "spdnet" else "raw", model_type=model,
                batch_size=8, learning_rate=1e-2, fl_rounds=2, fl_local_epochs=1,
                device="cpu", seed=0, spdnet_hidden_dim=3, early_stop_patience=0,
                eegnet_dropout=0.0)
    base.update(kw)
    if base.get("dp_enabled"):
        base.setdefault("dp_sigma", 0.0)
    return FLConfig(**base)


def _federation(cfg, n_clients=3, sigma=0.5):
    torch.manual_seed(0)
    make_x = _spd if cfg.model_type == "spdnet" else _raw
    clients = create_fl_clients_synthetic(_splits(n_clients, make_x), cfg,
                                          torch.device("cpu"), verbose=False)
    info = clients[0].info
    if cfg.model_type == "spdnet":
        pytest.importorskip("yetanotherspdnet")
        from rfl.models.spdnet import create_spdnet
        model = create_spdnet(n_classes=2, n_channels=info["n_channels"], seed=0,
                              **cfg.get_model_params())
    else:
        from rfl.models.eegnet import create_eegnet
        model = create_eegnet(n_classes=2, n_channels=info["n_channels"],
                              n_timepoints=info["n_timepoints"], sampling_rate=None,
                              **cfg.get_model_params())
    for c in clients:
        c.set_model(model)
    if cfg.dp_enabled:
        cfg.dp_sigma = sigma           # set directly: calibration is tested elsewhere
    return clients, FLServer(model, cfg, torch.device("cpu"))


def _run(server, clients, rounds):
    for r in range(1, rounds + 1):
        server.run_round(r, clients, CRIT, verbose=False)


def _head(client):
    return client.local_model.classifier.weight.detach().clone()


# --------------------------------------------------------------------------- #
#  Configuration matrix
# --------------------------------------------------------------------------- #
VALID = [
    dict(),
    dict(fl_local_fc=True),
    dict(fl_local_fc=True, fl_local_head_mode="frozen", fl_head_refit_epochs="5"),
    dict(fl_head_refit_epochs="5,10"),
    dict(freeze_trunk=True),
    dict(freeze_trunk=True, fl_local_fc=True),
    dict(freeze_trunk=True, fl_local_fc=True, fl_local_head_mode="frozen",
         fl_head_refit_epochs="5"),
]


@pytest.mark.parametrize("kw", VALID)
def test_valid_personalisations_without_dp(kw):
    _cfg(**kw)


@pytest.mark.parametrize("kw", VALID + [dict(fl_local_only=True)])
def test_valid_personalisations_with_dp(kw):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _cfg(dp_enabled=True, **kw)
    assert cfg.dp_enabled


@pytest.mark.parametrize("kw", [
    dict(fl_local_head_mode="frozen"),                       # no personal head
    dict(fl_local_only=True, fl_local_fc=True),              # redundant
    dict(fl_local_only=True, fl_head_refit_epochs="5"),      # nothing federated to re-fit
    dict(freeze_trunk=True, fl_local_fc=True,                # trains nothing
         fl_local_head_mode="frozen"),
    dict(fl_head_refit_epochs="-1"),
    dict(fl_head_refit_weight_decays="-0.1", fl_head_refit_epochs="5"),
    dict(fl_head_refit_selection="best"),
    dict(report_rule="last", early_stop_patience=5),
])
def test_invalid_personalisations_raise(kw):
    with pytest.raises(ValueError):
        _cfg(**kw)


def test_frozen_head_without_refit_warns():
    with pytest.warns(UserWarning, match="random initialisation"):
        _cfg(fl_local_fc=True, fl_local_head_mode="frozen")


@pytest.mark.parametrize("kw,match", [
    (dict(fl_local_only=True), "private local training"),
    (dict(freeze_trunk=True, fl_local_fc=True), "only protects the personal heads"),
    (dict(freeze_trunk=True, fl_local_fc=True, fl_local_head_mode="frozen",
          fl_head_refit_epochs="5"), "no private step"),
])
def test_dp_edge_cases_warn(kw, match):
    with pytest.warns(UserWarning, match=match):
        _cfg(dp_enabled=True, **kw)


# --------------------------------------------------------------------------- #
#  DP scope
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", ["spdnet", "eegnet"])
@pytest.mark.parametrize("kw", VALID + [dict(fl_local_only=True)])
def test_dp_scope_covers_every_trainable_parameter(model, kw):
    """No parameter trains outside the privatised vector, in any accepted setting."""
    import warnings

    from rfl.federated.privacy.dp_grad import federated_param_groups
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _cfg(model, dp_enabled=True, **kw)
    clients, _ = _federation(cfg)
    m = clients[0].local_model
    in_scope = {n for n, _, _ in federated_param_groups(m)}
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert in_scope == trainable
    head = {"classifier.weight", "classifier.bias"}
    frozen_head = kw.get("fl_local_head_mode") == "frozen"
    assert (head <= in_scope) != frozen_head
    if kw.get("freeze_trunk"):
        assert in_scope == (set() if frozen_head else head)


@pytest.mark.parametrize("model", ["spdnet", "eegnet"])
def test_dp_local_head_is_trained_privately_and_never_uploaded(model):
    cfg = _cfg(model, dp_enabled=True, fl_local_fc=True,
               fl_aggregation="projavg" if model == "spdnet" else "fedavg")
    clients, server = _federation(cfg)
    h0 = [_head(c) for c in clients]
    state, metrics = clients[0].train_local(1, CRIT, 1e-2)
    assert metrics["dp_local_steps"] == 32 // 8
    assert not any(k.startswith("classifier.") for k in state)
    _run(server, clients, 2)
    for c, h in zip(clients, h0):
        assert not torch.equal(_head(c), h)                        # trained
    assert not torch.equal(_head(clients[0]), _head(clients[1]))   # personal
    # The server head is never updated in local-FC mode.
    assert torch.equal(server.global_state_dict["classifier.weight"], h0[0])
    if model == "spdnet":
        W = server.global_state_dict["bimap.weight"]
        assert float(torch.norm(W.T @ W - torch.eye(W.shape[1]))) < 1e-4


@pytest.mark.parametrize("model", ["spdnet", "eegnet"])
@pytest.mark.parametrize("dp", [False, True])
def test_frozen_head_stays_at_its_initialisation(model, dp):
    """Exactly constant, on every client, also with partial participation (the EEGNet
    max-norm constraint does not touch frozen parameters)."""
    kw = dict(dp_enabled=True) if dp else {}
    cfg = _cfg(model, fl_local_fc=True, fl_local_head_mode="frozen",
               fl_head_refit_epochs="3", fl_client_fraction=0.5, **kw)
    clients, server = _federation(cfg)
    h0 = _head(clients[0])
    trunk = [n for n in server.global_state_dict if not n.startswith("classifier.")]
    t0 = {n: server.global_state_dict[n].clone() for n in trunk}
    _run(server, clients, 2)
    for c in clients:
        assert torch.equal(_head(c), h0)
    assert any(not torch.equal(server.global_state_dict[n], t0[n]) for n in trunk)


@pytest.mark.parametrize("model", ["spdnet", "eegnet"])
def test_frozen_trunk_stays_at_its_initialisation(model):
    """No training step and no constraint moves it. The aggregation of identical client
    copies returns it up to rounding."""
    cfg = _cfg(model, freeze_trunk=True, fl_local_fc=True,
               fl_aggregation="projavg" if model == "spdnet" else "fedavg")
    clients, server = _federation(cfg)
    t0 = {n: v.clone() for n, v in clients[0].local_model.named_parameters()
          if not n.startswith("classifier.")}
    _run(server, clients, 2)
    for c in clients:
        for n, p in c.local_model.named_parameters():
            if n in t0:
                assert torch.allclose(p, t0[n], rtol=0, atol=1e-6), n


def test_dp_with_nothing_to_train_runs_no_step():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _cfg("spdnet", dp_enabled=True, freeze_trunk=True, fl_local_fc=True,
                   fl_local_head_mode="frozen", fl_head_refit_epochs="2")
    clients, server = _federation(cfg)
    state, metrics = clients[0].train_local(1, CRIT, 1e-2)
    assert metrics["dp_local_steps"] == 0
    _run(server, clients, 1)


def test_dp_local_only_trains_privately_without_aggregation():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = _cfg("spdnet", dp_enabled=True, fl_local_only=True)
    clients, server = _federation(cfg)
    init = {k: v.clone() for k, v in server.global_state_dict.items()}
    _run(server, clients, 2)
    for k, v in init.items():
        assert torch.equal(server.global_state_dict[k], v)
    assert not torch.equal(clients[0].local_model.bimap.weight,
                           clients[1].local_model.bimap.weight)


def test_dp_frozen_trunk_trains_the_shared_head_only():
    cfg = _cfg("spdnet", dp_enabled=True, freeze_trunk=True)
    clients, server = _federation(cfg)
    W0 = server.global_state_dict["bimap.weight"].clone()
    h0 = server.global_state_dict["classifier.weight"].clone()
    _run(server, clients, 1)
    assert torch.allclose(server.global_state_dict["bimap.weight"], W0, atol=1e-6)
    assert not torch.equal(server.global_state_dict["classifier.weight"], h0)


def test_dp_local_head_noise_reaches_the_head():
    """With C tiny and sigma = 1 the head update is pure noise: the head is inside the
    mechanism, not trained on the raw gradient."""
    import copy

    from rfl.federated.privacy.dp_grad import dp_local_step, federated_param_groups
    cfg = _cfg("eegnet", dp_enabled=True, fl_local_fc=True)
    clients, _ = _federation(cfg)
    x, y = clients[0]._sample_wor_batch(8)
    deltas = []
    for seed in (1, 2):
        m = copy.deepcopy(clients[0].local_model)
        opt = torch.optim.SGD(m.parameters(), lr=1.0)
        before = m.classifier.weight.detach().clone()
        dp_local_step(m, opt, x, y, CRIT, federated_param_groups(m), C=1e-8, sigma=1.0,
                      generator=torch.Generator().manual_seed(seed))
        deltas.append(m.classifier.weight.detach() - before)
    # The clipped signal is at most 1e-8, so the update is the noise of each seed.
    assert not torch.allclose(deltas[0], deltas[1])
    assert float(deltas[0].std()) > 0.1


# --------------------------------------------------------------------------- #
#  Head re-fit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw", [dict(), dict(fl_local_fc=True),
                                dict(fl_local_fc=True, fl_local_head_mode="frozen")])
@pytest.mark.parametrize("dp", [False, True])
def test_refit_changes_the_head_only(kw, dp):
    from rfl.federated.personalization import refit_and_evaluate
    extra = dict(dp_enabled=True) if dp else {}
    cfg = _cfg("spdnet", fl_head_refit_epochs="2,4", **kw, **extra)
    clients, server = _federation(cfg)
    _run(server, clients, 1)
    state = {k: v.clone() for k, v in server.global_state_dict.items()}
    heads = [_head(c) for c in clients]
    out = refit_and_evaluate(clients, server.global_state_dict, CRIT,
                             torch.device("cpu"), lr=1e-2, epoch_grid=(2, 4),
                             weight_decays=(0.0, 1e-3), verbose=False)
    for k, v in state.items():                                     # nothing sent back
        assert torch.equal(server.global_state_dict[k], v)
    for c, h in zip(clients, heads):
        assert not torch.equal(_head(c), h)
        assert torch.equal(c.local_model.bimap.weight, state["bimap.weight"])
        assert c.local_model.bimap.weight.requires_grad == (not cfg.freeze_trunk)
    assert len(out["personalized_fits"]) == len(clients)
    assert all(f["epochs"] in (2, 4) and len(f["trace"]) == 4
               for f in out["personalized_fits"])
    assert 0.0 <= out["personalized_test_acc"] <= 1.0


def test_refit_applies_the_eegnet_max_norm():
    from rfl.federated.personalization import refit_head
    cfg = _cfg("eegnet", fl_head_refit_epochs="5")
    clients, server = _federation(cfg)
    refit_head(clients[0], server.global_state_dict, CRIT, torch.device("cpu"),
               lr=0.5, epoch_grid=(5,), selection="fixed")
    m = clients[0].local_model
    assert float(m.classifier.weight.detach().norm(dim=1).max()) <= m.norm_rate + 1e-5


def test_refit_fixed_selection_takes_the_last_budget():
    from rfl.federated.personalization import refit_head
    cfg = _cfg("spdnet", fl_head_refit_epochs="1,3")
    clients, server = _federation(cfg)
    fit = refit_head(clients[0], server.global_state_dict, CRIT, torch.device("cpu"),
                     lr=1e-2, epoch_grid=(1, 3), weight_decays=(1e-3, 0.0),
                     selection="fixed")
    assert fit["epochs"] == 3 and fit["weight_decay"] == 1e-3 and len(fit["trace"]) == 1


def test_parse_grids():
    from rfl.federated.personalization import parse_epoch_grid, parse_weight_decays
    assert parse_epoch_grid("100,25;50") == (25, 50, 100)
    assert parse_epoch_grid("") == () and parse_epoch_grid("0") == ()
    assert parse_weight_decays("") == (0.0,)
    assert parse_weight_decays("0, 1e-3") == (0.0, 1e-3)


# --------------------------------------------------------------------------- #
#  End to end through the runner
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    from rfl.data.synthetic import generate_synthetic_spd, save_dataset
    path = tmp_path_factory.mktemp("synth") / "synth.npz"
    data = generate_synthetic_spd(n_clients=3, samples_per_client=60, n_channels=6,
                                  n_classes=2, mode="iid", signal_strength=1.0, seed=0)
    save_dataset(data, str(path))
    return str(path)


@pytest.mark.parametrize("flags,scope", [
    ([], "shared"),
    (["--local-fc"], "fedper"),
    (["--local-fc", "--local-head-mode", "frozen", "--head-refit-epochs", "2,4"],
     "fedper_frozen+refit"),
    (["--head-refit-epochs", "3", "--head-refit-selection", "fixed"], "shared+refit"),
    (["--freeze-trunk", "--local-fc"], "fedper_rnd"),
])
@pytest.mark.parametrize("dp", [False, True])
def test_runner_matrix(synth, tmp_path, flags, scope, dp):
    from rfl.analysis.results import load_runs
    from rfl.experiments.run_federated_experiment import main
    argv = ["--synthetic-clients", synth, "--model", "spdnet", "--hidden-dim", "3",
            "--rounds", "2", "--batch-size", "8", "--learning-rate", "1e-2",
            "--device", "cpu", "--seed", "1", "--save-root", str(tmp_path),
            "--experiment-name", "m"] + flags
    if dp:
        argv += ["--dp-enabled", "--dp-target-eps", "10"]
    main(argv)
    (run_dir,) = list((tmp_path / "m").iterdir())
    assert scope.replace("+", "-") in run_dir.name and ("eps10" in run_dir.name) == dp
    r = json.loads((run_dir / "fl_results.json").read_text())
    assert r["personalisation"] == scope
    assert (r["dp_budget"] is not None) == dp
    if dp:
        assert r["dp_budget"]["realized_eps"] <= 10.0 + 1e-9
    has_refit = "+refit" in scope
    assert (r["personalized_metrics"] is not None) == has_refit
    assert (run_dir / "client_states.pt").exists() == ("--local-fc" in flags or has_refit)
    if (run_dir / "client_states.pt").exists():
        # The saved files rebuild the reported model of each client.
        import torch as _t
        saved = _t.load(run_dir / "client_states.pt", weights_only=False)
        ckpt = _t.load(run_dir / "global_model_final.pt", weights_only=False)
        state = dict(ckpt["model_state_dict"])
        state.update(saved["local_states"][0])
        state.update(saved["heads"][0])
        assert set(saved["heads"][0]) == {"classifier.weight", "classifier.bias"}
        # The rebuilt state is a complete model of the run's architecture.
        from rfl.models.spdnet import create_spdnet
        create_spdnet(n_classes=2, n_channels=6, hidden_dim=3, seed=0).load_state_dict(
            state, strict=True)
        if "--local-fc" in flags and "--freeze-trunk" not in flags:
            frozen = "frozen" in flags
            # A trained personal head differs per client. A frozen one is the common
            # initial head until the re-fit, which then makes it personal.
            h = saved["refit_heads"] if frozen else saved["heads"]
            assert not _t.equal(h[0]["classifier.weight"], h[1]["classifier.weight"])
            if frozen:
                assert _t.equal(saved["heads"][0]["classifier.weight"],
                                saved["heads"][1]["classifier.weight"])
    df = load_runs(str(tmp_path))
    assert list(df["scope"]) == [scope]
    assert np.isfinite(df["refit_test_acc"].iloc[0]) == has_refit


def test_runner_rejects_inconsistent_settings_before_training(synth, tmp_path):
    from rfl.experiments.run_federated_experiment import main
    with pytest.raises(SystemExit):
        main(["--synthetic-clients", synth, "--rounds", "1", "--device", "cpu",
              "--local-only", "--local-fc", "--save-root", str(tmp_path)])
    assert not (tmp_path / "federated").exists()


def test_selected_round_restores_each_clients_local_keys():
    """report_rule='selected': the reported model is the selected global state with the
    personal heads of that round, not of the last round."""
    cfg = _cfg("spdnet", fl_local_fc=True, report_rule="selected", fl_rounds=4,
               fl_aggregation="projavg")
    clients, server = _federation(cfg)
    heads = {}
    for r in range(1, 5):
        server.run_round(r, clients, CRIT, verbose=False)
        heads[r] = [_head(c) for c in clients]
    best = server.best_round
    server.load_reported_model(clients, "selected")
    for c, h in zip(clients, heads[best]):
        assert torch.equal(_head(c), h)
        assert torch.equal(c.local_model.bimap.weight,
                           server.best_state_dict["bimap.weight"])
    evals = clients[0].evaluate_local_model(CRIT, splits=("test",))
    pos = server.round_history["round"].index(best)
    assert evals["test"]["acc"] == pytest.approx(
        server.round_history["per_client_test_acc"][pos][0])


def test_default_run_names_differ_with_any_option():
    from rfl.experiments.run_federated_experiment import (_build_arg_parser, apply_defaults,
                                                      config_from_args, default_run_name)
    def name(extra):
        a = _build_arg_parser().parse_args(["--synthetic-clients", "x.npz"] + extra)
        apply_defaults(a, "synthetic")
        return default_run_name(config_from_args(a))
    base = name([])
    assert base == name([])
    for extra in (["--learning-rate", "0.1"], ["--rounds", "7"], ["--local-epochs", "2"],
                  ["--batch-size", "8"], ["--hidden-dim", "3"]):
        assert name(extra) != base
    assert name(["--seed", "5"]).replace("seed5", "seed42") == base



@pytest.mark.parametrize("flags", [
    ["--local-fc", "--head-refit-epochs", "3,6"],
    ["--local-fc", "--local-head-mode", "frozen", "--head-refit-epochs", "3,6",
     "--report", "selected", "--lr-schedule", "plateau", "--early-stop-patience", "2"],
])
def test_saved_files_rebuild_the_reported_and_the_refitted_models(synth, tmp_path, flags,
                                                                  monkeypatch):
    """global_model_final.pt + client_states.pt give back the models behind
    final_metrics and behind personalized_metrics."""
    import copy
    import json

    import numpy as np

    import rfl.federated.server as srv
    from rfl.experiments.run_federated_experiment import main
    held = {}
    orig = srv.create_fl_clients_synthetic

    def capture(*a, **k):
        held["clients"] = orig(*a, **k)
        return held["clients"]

    monkeypatch.setattr(srv, "create_fl_clients_synthetic", capture)
    out = tmp_path / "run"
    main(["--synthetic-clients", synth, "--model", "spdnet", "--hidden-dim", "3",
          "--rounds", "4", "--batch-size", "8", "--learning-rate", "5e-2", "--seed", "1",
          "--device", "cpu", "--save-dir", str(out)] + flags)
    res = json.loads((out / "fl_results.json").read_text())
    g = torch.load(out / "global_model_final.pt", weights_only=False)
    cs = torch.load(out / "client_states.pt", weights_only=False)
    gsd = (g["best_state_dict"] if res["report_rule"] == "selected" else g["model_state_dict"])
    def pooled_val_acc(refit):
        preds, labels = [], []
        for c in held["clients"]:
            m = copy.deepcopy(c.local_model)
            st = dict(gsd)
            st.update(cs["local_states"][c.client_id])
            if refit:
                st.update(cs["refit_heads"][c.client_id])
            m.load_state_dict(st)
            m.eval()
            with torch.no_grad():
                for x, y in c.val_loader:
                    preds += m(x).argmax(1).tolist()
                    labels += y.tolist()
        return float(np.mean(np.array(preds) == np.array(labels)))

    assert pooled_val_acc(False) == pytest.approx(res["final_metrics"]["global_val_acc"])
    assert pooled_val_acc(True) == pytest.approx(
        res["personalized_metrics"]["personalized_val_acc"])
