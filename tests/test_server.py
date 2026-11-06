"""Federated server: learning-rate schedules, round selection and short runs.

The runs use small synthetic clients (SPD matrices for SPDNet, random raw signals for
EEGNet) on the CPU.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from rfl.federated.fl_config import FLConfig
from rfl.federated.server import FLServer, compute_round_lr, create_fl_clients_synthetic


# --------------------------------------------------------------------------- #
#  compute_round_lr
# --------------------------------------------------------------------------- #
def test_constant_and_cosine_schedules():
    assert compute_round_lr(1e-2, 5, 10, "none") == 1e-2
    assert compute_round_lr(1e-2, 1, 10, "cosine", lr_min=1e-4) == pytest.approx(1e-2)
    assert compute_round_lr(1e-2, 10, 10, "cosine", lr_min=1e-4) == pytest.approx(1e-4)
    mid = 1e-4 + (1e-2 - 1e-4) * 0.5 * (1 + math.cos(math.pi * 4 / 9))
    assert compute_round_lr(1e-2, 5, 10, "cosine", lr_min=1e-4) == pytest.approx(mid)


def test_step_schedule_switches_at_seventy_percent():
    lrs = [compute_round_lr(1.0, r, 10, "step", lr_min=1e-4) for r in range(1, 11)]
    assert lrs[:7] == [1.0] * 7 and lrs[7:] == pytest.approx([0.3] * 3)   # 70% at base
    assert compute_round_lr(1.0, 1, 2, "step", lr_min=1e-4) == 1.0      # never starts decayed


def test_plateau_schedule_decays_after_patience():
    hist = [1.0, 0.9, 0.9, 0.95, 0.9]          # last strict improvement at index 1
    lr = compute_round_lr(1.0, 6, 10, "plateau", lr_min=0.1, val_history=hist,
                          plateau_patience=3, plateau_factor=0.5, mode="min")
    assert lr == 0.5
    # Accuracy-like metric, maximised: still improving, no decay.
    assert compute_round_lr(1.0, 6, 10, "plateau", lr_min=0.1, val_history=[1, 2, 3],
                            plateau_patience=1, mode="max") == 1.0
    # Floored at lr_min.
    assert compute_round_lr(1.0, 9, 10, "plateau", lr_min=0.3, val_history=[1.0] * 8,
                            plateau_patience=1, plateau_factor=0.5, mode="min") == 0.3


def test_plateau_without_history_and_unknown_schedule_raise():
    with pytest.raises(ValueError):
        compute_round_lr(1.0, 2, 10, "plateau")
    with pytest.raises(ValueError):
        compute_round_lr(1.0, 2, 10, "exponential")


# --------------------------------------------------------------------------- #
#  Synthetic federations
# --------------------------------------------------------------------------- #
def _split(rng, n, make_x, n_classes):
    y = np.arange(n) % n_classes
    return make_x(n), y.astype(np.int64)


def _splits(n_clients, make_x, n_classes=2, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_clients):
        d = {}
        for name, n in (("train", 24), ("val", 8), ("test", 8)):
            d[f"X_{name}"], d[f"y_{name}"] = _split(rng, n, lambda m: make_x(rng, m),
                                                    n_classes)
        out.append(d)
    return out


def _spd(rng, n, d=6):
    A = rng.randn(n, d, 3 * d)
    return (A @ A.transpose(0, 2, 1) / (3 * d) + 1e-3 * np.eye(d)).astype(np.float32)


def _raw(rng, n, c=4, t=64):
    return rng.randn(n, c, t).astype(np.float32)


def _cfg(model, **kw):
    base = dict(experiment_name="t", run_name="t", dataset="synthetic", subjects=[],
                data_type="spd" if model == "spdnet" else "raw", model_type=model,
                batch_size=8, learning_rate=1e-2, fl_rounds=2, fl_local_epochs=1,
                device="cpu", seed=0, spdnet_hidden_dim=3, early_stop_patience=0)
    base.update(kw)
    return FLConfig(**base)


def _federation(cfg, n_clients=3):
    """Clients and server for `cfg`, as run_federated_experiment builds them."""
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
    return clients, FLServer(model, cfg, torch.device("cpu"))


def _run(server, clients, rounds):
    for r in range(1, rounds + 1):
        server.run_round(r, clients, nn.CrossEntropyLoss(), verbose=False)


def _ortho_err(W):
    return float(torch.norm(W.T @ W - torch.eye(W.shape[1])))


@pytest.mark.parametrize("agg", ["projavg", "rlavg", "retavg"])
def test_spdnet_rounds_record_history(agg):
    cfg = _cfg("spdnet", fl_aggregation=agg, learning_rate=1e-3, lr_schedule="none")
    clients, server = _federation(cfg)
    _run(server, clients, 2)
    h = server.round_history
    assert h["round"] == [1, 2]
    for key in ("global_test_acc", "global_val_loss", "agg_seconds", "proj_defect",
                "client_disp", "head_norm"):
        assert len(h[key]) == 2, key
    assert len(h["per_client_test_acc"][0]) == 3
    W = server.global_state_dict["bimap.weight"]
    assert _ortho_err(W) < 1e-4                # the rules keep W on the manifold
    assert server.best_round in (1, 2)
    rnd, metrics = server.get_selected_round()
    assert rnd == server.best_round and "global_test_acc" in metrics
    assert server.get_metrics_at_round(2)["round"] == 2
    assert server.get_final_metrics()["round"] == 2


def test_local_head_is_not_uploaded_or_overwritten():
    cfg = _cfg("spdnet", fl_local_fc=True)
    clients, server = _federation(cfg)
    heads = [c.local_model.classifier.weight.detach().clone() for c in clients]
    state, _ = clients[0].train_local(1, nn.CrossEntropyLoss(), 1e-2)
    assert not any(k.startswith("classifier.") for k in state)
    _run(server, clients, 1)
    for c, h0 in zip(clients, heads):
        assert not torch.equal(c.local_model.classifier.weight, h0)     # trained locally
    assert not torch.equal(clients[0].local_model.classifier.weight,
                           clients[1].local_model.classifier.weight)


def test_frozen_trunk_trains_only_the_head():
    cfg = _cfg("spdnet", freeze_trunk=True, fl_aggregation="projavg")
    clients, server = _federation(cfg)
    W0 = server.global_state_dict["bimap.weight"].clone()
    h0 = server.global_state_dict["classifier.weight"].clone()
    _run(server, clients, 1)
    assert torch.allclose(server.global_state_dict["bimap.weight"], W0, atol=1e-6)
    assert not torch.equal(server.global_state_dict["classifier.weight"], h0)


def test_local_only_runs_without_aggregation():
    cfg = _cfg("spdnet", fl_local_only=True, early_stop_patience=1, report_rule="selected")
    clients, server = _federation(cfg)
    init = {k: v.clone() for k, v in server.global_state_dict.items()}
    _run(server, clients, 2)
    for k, v in init.items():
        assert torch.equal(server.global_state_dict[k], v)
    assert len(server.round_history["n_active_clients"]) == 2
    assert server.best_state_dict is None


def test_partial_participation_samples_ceil_fraction():
    cfg = _cfg("spdnet", fl_client_fraction=0.5)
    clients, server = _federation(cfg)
    chosen = server._select_participants(clients, 1, verbose=False)
    assert len(chosen) == 2                    # ceil(3 * 0.5)


def test_eegnet_fedbn_keeps_batchnorm_local():
    # Optional BatchNorm (non-private runs only, not used in the paper).
    cfg = _cfg("eegnet", eegnet_dropout=0.0, eegnet_norm_type="batch")
    clients, server = _federation(cfg)
    _run(server, clients, 1)
    # The global BatchNorm buffers are never aggregated, so they keep their init.
    assert torch.all(server.global_state_dict["batchnorm_temporal.running_mean"] == 0)
    assert not torch.equal(clients[0].local_model.batchnorm_temporal.running_mean,
                           clients[1].local_model.batchnorm_temporal.running_mean)


def test_spdnet_batchnorm_runs_and_stays_local():
    """SPD batch norm (non-private): it trains, and its bias, held by a parametrization
    child module, stays on the client (FedBN)."""
    pytest.importorskip("yetanotherspdnet")
    from rfl.federated.param_groups import identify_bn_param_names
    cfg = _cfg("spdnet", spdnet_use_batch_norm=True, fl_aggregation="projavg")
    clients, server = _federation(cfg)
    names = identify_bn_param_names(clients[0].local_model)
    assert any("Covbias" in n for n in names)
    _run(server, clients, 2)
    key = next(n for n in names if "Covbias" in n)
    biases = [c.local_model.state_dict()[key] for c in clients]
    assert not torch.equal(biases[0], biases[1])            # personal, trained locally
    assert not torch.equal(biases[0], server.global_state_dict[key])


def test_local_only_selected_keeps_each_clients_best_model():
    """With --local-only --report selected, each client ends with the model of its own
    best validation round, and the pooled metrics are those of these models."""
    cfg = _cfg("spdnet", fl_local_only=True, report_rule="selected",
               early_stop_patience=2, fl_rounds=8, learning_rate=5e-2,
               lr_schedule="none")
    clients, server = _federation(cfg)
    crit = nn.CrossEntropyLoss()
    for r in range(1, 9):
        server.run_round(r, clients, crit, verbose=False)
        if server.all_clients_stopped:
            break
    hist = server._client_hist
    # Not vacuous: for some client the best round is not the last one.
    assert any(h.index(max(h)) < len(h) - 1 for h in hist.values())
    server.load_reported_model(clients, "selected")
    ev = server.evaluate_clients(clients, crit, local_only=True)
    for c, acc in zip(clients, ev["val_accs"]):
        assert acc == pytest.approx(max(server._client_hist[c.client_id]))


@pytest.mark.parametrize("agg", ["projavg", "rlavg", "retavg"])
def test_frozen_trunk_stays_exactly_at_its_initialisation(agg):
    cfg = _cfg("spdnet", fl_aggregation=agg, freeze_trunk=True, lr_schedule="none")
    clients, server = _federation(cfg)
    W0 = server.global_state_dict["bimap.weight"].clone()
    _run(server, clients, 3)
    assert torch.equal(server.global_state_dict["bimap.weight"], W0)


def test_weight_by_samples_weights_clients_by_their_training_size():
    for by_samples in (False, True):
        cfg = _cfg("eegnet", fl_weight_by_samples=by_samples, lr_schedule="none")
        clients, server = _federation(cfg)
        seen = {}
        orig = server.aggregate_updates

        def spy(states, stiefel, weights, _orig=orig):
            seen["w"] = list(weights) if weights is not None else None
            return _orig(states, stiefel, weights)

        server.aggregate_updates = spy
        _run(server, clients, 1)
        sizes = [c.n_samples for c in clients]
        if by_samples:
            assert seen["w"] == sizes or seen["w"] == [s / sum(sizes) for s in sizes]
        else:
            assert seen["w"] is None or len(set(seen["w"])) == 1


# ---- client partition of real EEG (MOABB loader replaced by a fake) -------------
def _partition_clients(monkeypatch, partition, n_subjects=4, per_subject=40):
    import pandas as pd

    import rfl.federated.server as sv
    n = n_subjects * per_subject
    subject = np.repeat(np.arange(1, n_subjects + 1), per_subject)
    y = np.tile(np.arange(4), n // 4)
    X = np.zeros((n, 3, 8))
    X[:, 0, 0] = np.arange(n)                    # trial id, read back below
    meta = pd.DataFrame({"subject": subject, "session": "0", "run": "0",
                         "event": y.astype(str)})
    monkeypatch.setattr(sv, "load_moabb_dataset", lambda **kw: (X, y, meta))
    monkeypatch.setattr(sv, "filter_binary_task", lambda X, y, meta, **kw: (X, y, meta))

    seen = []

    class Recorder:
        def __init__(self, client_id, subjects, **kw):
            self.client_id, self.subjects = client_id, subjects

        def prepare_data_from_splits(self, Xtr, ytr, Xva, yva, Xte, yte, verbose=False):
            ids = [a[:, 0, 0].astype(int).tolist() for a in (Xtr, Xva, Xte)]
            seen.append((self.subjects, ids, [ytr, yva, yte]))

    monkeypatch.setattr(sv, "_client_class", lambda config: Recorder)
    cfg = FLConfig(dataset="BNCI2014_001", subjects=list(range(1, n_subjects + 1)),
                   fl_client_partition=partition)
    sv.create_fl_clients_by_subject("BNCI2014_001", cfg.subjects, 1, cfg,
                                    torch.device("cpu"), verbose=False)
    return seen, subject


def test_iid_partition_deals_the_same_split_to_the_same_number_of_clients(monkeypatch):
    by_subject, subject = _partition_clients(monkeypatch, "subject")
    iid, _ = _partition_clients(monkeypatch, "iid")
    assert len(iid) == len(by_subject) == 4
    for k in range(3):                           # train, val, test: same trials overall
        a = sorted(i for _, ids, _ in by_subject for i in ids[k])
        b = sorted(i for _, ids, _ in iid for i in ids[k])
        assert a == b
    for subs, ids, _ in by_subject:              # one subject per client
        assert {int(subject[i]) for i in sum(ids, [])} == set(subs) and len(subs) == 1
    for k in range(3):                           # IID: balanced sizes and classes
        sizes = [len(ids[k]) for _, ids, _ in iid]
        assert max(sizes) - min(sizes) <= 1
        for c in range(4):
            per = [int((ys[k] == c).sum()) for _, _, ys in iid]
            assert max(per) - min(per) <= 1
    for subs, ids, _ in iid:                     # every client mixes the subjects
        assert set(subs) == {int(subject[i]) for i in sum(ids, [])}
        assert len(subs) == 4
    again, _ = _partition_clients(monkeypatch, "iid")
    assert [ids for _, ids, _ in again] == [ids for _, ids, _ in iid]   # reproducible


def test_client_partition_is_refused_on_synthetic_data(tmp_path):
    import contextlib
    import io

    from rfl.experiments import run_federated_experiment as F
    err = io.StringIO()
    with contextlib.redirect_stderr(err), pytest.raises(SystemExit):
        F.prepare_args(["--synthetic-clients", str(tmp_path / "s.npz"),
                        "--client-partition", "iid"])
    assert "--client-partition" in err.getvalue()


def test_a_registered_mean_runs_end_to_end(monkeypatch):
    import rfl.federated.aggregation as A
    monkeypatch.setattr(A, "AGGREGATION_RULES", dict(A.AGGREGATION_RULES))
    calls = []

    def mean(Ws, weights, base):
        calls.append(len(Ws))
        return A.polar_projection(sum(w * W for w, W in zip(weights, Ws)))

    A.register_stiefel_mean("mymean", mean)
    cfg = _cfg("spdnet", fl_aggregation="mymean", learning_rate=1e-3, lr_schedule="none")
    clients, server = _federation(cfg)
    _run(server, clients, 2)
    assert calls and set(calls) == {3}
    assert _ortho_err(server.global_state_dict["bimap.weight"]) < 1e-4
    with pytest.raises(ValueError, match="EEGNet"):
        _cfg("eegnet", fl_aggregation="mymean")
