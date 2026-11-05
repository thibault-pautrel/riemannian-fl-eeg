"""Tests for the centralised training loop, evaluation and checkpointing."""
import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rfl.training.checkpointing import CheckpointManager
from rfl.training.evaluation import Evaluator
from rfl.training.train_loop import epoch_lr, setup_training, train_model
from rfl.utils.config import ExperimentConfig


def _cfg(**kw):
    # Selection on the validation loss, as these tests check it.
    base = dict(run_name="t", epochs=6, learning_rate=1e-2, batch_size=16,
                early_stopping_patience=0, device="cpu", model_type="eegnet",
                selection_metric="val_loss", report_rule="selected")
    base.update(kw)
    return ExperimentConfig(**base)


def _loaders(seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(96, 5, generator=g)
    y = (X[:, 0] + 0.3 * torch.randn(96, generator=g) > 0).long()
    tr = DataLoader(TensorDataset(X[:64], y[:64]), batch_size=16, shuffle=False)
    va = DataLoader(TensorDataset(X[64:80], y[64:80]), batch_size=16)
    te = DataLoader(TensorDataset(X[80:], y[80:]), batch_size=16)
    return tr, va, te


def _run(tmp_path, cfg, seed=0):
    torch.manual_seed(seed)
    model = nn.Linear(5, 2)
    tr, va, te = _loaders()
    ckpt = CheckpointManager(str(tmp_path / "ckpt"), "run", cfg)
    ev = Evaluator(2, ["a", "b"])
    out = train_model(model, tr, va, nn.CrossEntropyLoss(), cfg, torch.device("cpu"),
                      ckpt, ev, test_loader=te)
    return model, ckpt, out


# ---- schedules ----------------------------------------------------------------
def test_epoch_lr_endpoints_and_monotonicity():
    lrs = [epoch_lr(1e-2, e, 10, 1e-4) for e in range(1, 11)]
    assert lrs[0] == pytest.approx(1e-2) and lrs[-1] == pytest.approx(1e-4)
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
    assert epoch_lr(1e-2, 1, 1, 1e-4) == 1e-2


@pytest.mark.parametrize("schedule", ["plateau", "step", "cosine", "none"])
def test_centralised_rate_follows_the_shared_schedule(tmp_path, schedule):
    """The centralised loop uses rfl/utils/schedules.py exactly as the federated server."""
    from rfl.utils.schedules import compute_round_lr
    cfg = _cfg(lr_schedule=schedule, lr_plateau_patience=1, lr_plateau_factor=0.5,
               lr_min=1e-5, epochs=12, learning_rate=1.0)
    _, _, out = _run(tmp_path, cfg)
    h = out["history"]
    expected = [compute_round_lr(1.0, e, 12, schedule, 1e-5,
                                 val_history=h["val_loss"][:e - 1], plateau_patience=1,
                                 plateau_factor=0.5, mode="min")
                for e in h["epoch"]]
    assert h["learning_rate"] == pytest.approx(expected)
    if schedule == "step":
        assert h["learning_rate"][-1] == pytest.approx(0.3)


def test_setup_training_returns_no_scheduler():
    for schedule in ("cosine", "none", "plateau", "step"):
        _, sched = setup_training(nn.Linear(2, 2), _cfg(lr_schedule=schedule),
                                  torch.device("cpu"))
        assert sched is None


# ---- train_model ----------------------------------------------------------------
def test_train_model_selects_best_validation_loss(tmp_path):
    _, ckpt, out = _run(tmp_path, _cfg(lr_schedule="none"))
    h = out["history"]
    assert len(h["epoch"]) == 6 and out["final_epoch"] == 6
    pos = min(range(6), key=lambda i: h["val_loss"][i])
    assert out["best_epoch"] == h["epoch"][pos]
    assert out["best_val_loss"] == h["val_loss"][pos]
    assert out["final_val_loss"] == h["val_loss"][-1]
    assert not out["stopped_early"]
    assert all(not math.isnan(v) for v in h["test_acc"])
    assert ckpt.load_best(nn.Linear(5, 2))["epoch"] == out["best_epoch"]


def test_train_model_cosine_sets_rate_per_epoch(tmp_path):
    _, _, out = _run(tmp_path, _cfg(lr_schedule="cosine", lr_min=1e-4, report_rule="last"))
    lrs = out["history"]["learning_rate"]
    assert lrs[0] == pytest.approx(1e-2) and lrs[-1] == pytest.approx(1e-4)
    assert out["report_rule"] == "last"


def test_train_model_early_stopping(tmp_path):
    cfg = _cfg(lr_schedule="none", learning_rate=1.0, epochs=40, early_stopping_patience=2)
    _, _, out = _run(tmp_path, cfg)
    if out["stopped_early"]:
        assert out["final_epoch"] - out["best_epoch"] == 2
    else:
        assert out["final_epoch"] == 40


def test_report_last_with_early_stopping_raises(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, _cfg(report_rule="last", early_stopping_patience=5))


def test_train_model_is_reproducible(tmp_path):
    cfg = _cfg(lr_schedule="none")
    m1, _, o1 = _run(tmp_path / "a", cfg, seed=3)
    m2, _, o2 = _run(tmp_path / "b", cfg, seed=3)
    assert o1["history"]["val_loss"] == o2["history"]["val_loss"]
    assert torch.equal(m1.weight, m2.weight)


# ---- checkpointing --------------------------------------------------------------
def test_checkpoint_keeps_last_n_and_resumes(tmp_path):
    cfg = _cfg()
    m = nn.Linear(2, 2)
    opt = torch.optim.Adam(m.parameters())
    ckpt = CheckpointManager(str(tmp_path), "r", cfg, keep_last_n=2)
    for e in range(1, 5):
        ckpt.save(m, opt, e, {"val_loss": 1.0}, {"epoch": list(range(1, e + 1))})
    files = sorted(p.name for p in tmp_path.glob("r_epoch_*.pt"))
    assert files == ["r_epoch_3.pt", "r_epoch_4.pt"]
    assert ckpt.load_latest(nn.Linear(2, 2)) == 5
    assert ckpt.load_history() == {"epoch": [1, 2, 3, 4]}


def test_checkpoint_of_another_configuration_is_not_resumed(tmp_path):
    m = nn.Linear(2, 2)
    opt = torch.optim.Adam(m.parameters())
    old = CheckpointManager(str(tmp_path), "r", _cfg(learning_rate=1e-2))
    for e in range(1, 4):
        old.save(m, opt, e, {"val_loss": 1.0}, {"epoch": list(range(1, e + 1))},
                 is_best=(e == 1))
    new = CheckpointManager(str(tmp_path), "r", _cfg(learning_rate=1e-4))
    assert new.load_history() is None
    assert new.load_latest(nn.Linear(2, 2)) == 1                  # starts from scratch
    assert not list(tmp_path.glob("r_*.pt"))                     # set aside
    assert len(list((tmp_path / "stale").glob("r_*.pt"))) == 4


def test_best_epoch_also_writes_a_periodic_checkpoint(tmp_path):
    m = nn.Linear(2, 2)
    ckpt = CheckpointManager(str(tmp_path), "r", _cfg())
    ckpt.save(m, torch.optim.Adam(m.parameters()), 1, {}, {"epoch": [1]}, is_best=True)
    assert (tmp_path / "r_epoch_1.pt").exists() and (tmp_path / "r_best.pt").exists()
    assert ckpt.load_latest(nn.Linear(2, 2)) == 2


def test_checkpoint_without_files(tmp_path):
    ckpt = CheckpointManager(str(tmp_path), "r", _cfg())
    assert ckpt.load_latest(nn.Linear(2, 2)) == 1
    assert ckpt.load_history() is None
    with pytest.raises(FileNotFoundError):
        ckpt.load_best(nn.Linear(2, 2))


# ---- evaluation -------------------------------------------------------------------
def test_evaluator_metrics_and_saving(tmp_path):
    model = nn.Linear(5, 2)
    _, va, _ = _loaders()
    ev = Evaluator(2, ["a", "b"], save_dir=str(tmp_path))
    res = ev.evaluate_and_save(model, va, nn.CrossEntropyLoss(), torch.device("cpu"),
                               "val", "run")
    assert 0.0 <= res["accuracy"] <= 1.0
    assert len(res["confusion_matrix"]) == 2
    assert sum(map(sum, res["confusion_matrix"])) == 16
    assert (tmp_path / "metrics" / "run_val_results.json").exists()


def test_selection_never_picks_nan():
    from rfl.utils.schedules import best_position
    assert best_position([float("nan"), 0.5, 0.4], lower_is_better=True) == 2
    assert best_position([float("nan"), 0.5, 0.7], lower_is_better=False) == 2
    assert best_position([float("nan")], lower_is_better=True) is None


def test_resumed_run_after_early_stop_does_not_train_again(tmp_path):
    cfg = _cfg(lr_schedule="none", learning_rate=1.0, epochs=40, early_stopping_patience=2)
    _, _, first = _run(tmp_path, cfg)
    assert first["stopped_early"]
    _, _, again = _run(tmp_path, cfg)                 # same folder: resumes
    assert again["final_epoch"] == first["final_epoch"]
    assert again["history"]["epoch"] == first["history"]["epoch"]


def test_resume_continues_the_same_random_stream(tmp_path):
    """Interrupted after epoch 4, then resumed: same weights as an uninterrupted run.

    The training loader shuffles and the model has dropout, so the random stream
    matters."""
    import shutil

    def run(path):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Dropout(0.3), nn.Linear(5, 2))
        tr, va, te = _loaders()
        tr = DataLoader(tr.dataset, batch_size=16, shuffle=True)
        ckpt = CheckpointManager(str(path / "ckpt"), "run", cfg)
        out = train_model(model, tr, va, nn.CrossEntropyLoss(), cfg, torch.device("cpu"),
                          ckpt, Evaluator(2, ["a", "b"]), test_loader=te)
        return model, out

    cfg = _cfg(lr_schedule="none", epochs=6)
    m_full, _ = run(tmp_path / "full")
    shutil.copytree(tmp_path / "full", tmp_path / "cut")
    for e in (5, 6):                                   # the run "died" after epoch 4
        (tmp_path / "cut" / "ckpt" / f"run_epoch_{e}.pt").unlink()
    assert (tmp_path / "cut" / "ckpt" / "run_epoch_4.pt").exists()
    torch.manual_seed(123)                             # a fresh process: another state
    m_res, out = run(tmp_path / "cut")
    assert out["history"]["epoch"] == [1, 2, 3, 4, 5, 6]
    assert out["final_epoch"] == 6
    for a, b in zip(m_full.parameters(), m_res.parameters()):
        assert torch.allclose(a, b, atol=1e-7)
