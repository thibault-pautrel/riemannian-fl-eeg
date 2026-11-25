"""Inter-subject protocol: folds, target split, adaptation, runners, plans and analysis."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from rfl.experiments import loso as L

N_SUBJECTS, PER_CLASS, N_CH, T = 4, 20, 6, 64


def _fake_eeg(n_subjects=N_SUBJECTS, per_class=PER_CLASS, seed=0):
    """Trials whose class sets the power of one channel, mixed per subject."""
    rng = np.random.default_rng(seed)
    X, y, subj = [], [], []
    for s in range(1, n_subjects + 1):
        A = np.eye(N_CH) + 0.3 * rng.normal(size=(N_CH, N_CH))
        for c in range(4):
            for _ in range(per_class):
                z = rng.normal(size=(N_CH, T))
                z[c] *= 3.0
                X.append(A @ z)
                y.append(c)
                subj.append(s)
    meta = pd.DataFrame({"subject": subj, "session": "0", "run": "0",
                         "event": np.asarray(y).astype(str)})
    return np.asarray(X, dtype=np.float32), np.asarray(y), meta


@pytest.fixture
def fake_moabb(monkeypatch):
    """The MOABB loader of every runner replaced by `_fake_eeg`."""
    import rfl.experiments.run_centralized_experiment as rc
    import rfl.federated.server as sv
    X, y, meta = _fake_eeg()

    def load(**kw):
        keep = meta.subject.isin(kw["subjects"]).to_numpy()
        return X[keep], y[keep], meta[keep].reset_index(drop=True)
    for mod in (sv, rc):
        monkeypatch.setattr(mod, "load_moabb_dataset", load)
        monkeypatch.setattr(mod, "filter_binary_task", lambda X, y, meta, **kw: (X, y, meta))
    return X, y, meta


COMMON = ["--dataset", "BNCI2014_001", "--subjects", "1", "2", "3", "4",
          "--batch-size", "16", "--learning-rate", "0.05", "--device", "cpu"]
SPD = ["--model", "spdnet", "--hidden-dim", "4"]
EEG = ["--model", "eegnet", "--resample", "0"]


def _run_fed(tmp_path, *extra, name="fed"):
    from rfl.experiments.run_federated_experiment import main
    out = tmp_path / name
    main([*COMMON, "--rounds", "2", *extra, "--save-dir", str(out)])
    return json.loads((out / "fl_results.json").read_text())


# ----------------------------------------------------------------------- folds
def test_folds_cover_every_subject_once():
    from rfl.data.registry import get_subjects
    for ds, n in (("BNCI2014_001", 9), ("Weibo2014", 10), ("PhysionetMI", 10)):
        f = L.folds(ds)
        assert len(f) == n
        flat = [s for g in f for s in g]
        assert sorted(flat) == sorted(get_subjects(ds)) and len(set(flat)) == len(flat)
    assert all(len(g) == 1 for g in L.folds("BNCI2014_001"))
    sizes = [len(g) for g in L.folds("PhysionetMI")]
    assert max(sizes) - min(sizes) <= 1
    with pytest.raises(SystemExit):
        L.folds("Cho2017")


# ------------------------------------------------------------------ target split
def _cfg(**kw):
    base = dict(seed=3, calibration_val_fraction=0.2, calibration_order="random",
                target_subjects=[2], target_ns=[0, 2, 5, 10])
    base.update(kw)
    return SimpleNamespace(**base)


def test_target_split_is_disjoint_nested_and_that_of_the_datasize_study():
    from rfl.data.data_utils import split_indices
    from rfl.data.reduction import reduce_pool, subject_rng
    X, y, meta = _fake_eeg()
    X = np.arange(len(y))[:, None, None] * np.ones((1, 1, 1))    # trial ids
    subj = meta.subject.to_numpy()
    tr, va, te = split_indices(y, 0.1, 0.15, 3)
    (t,) = L.split_targets(X, y, subj, tr, va, te, [1, 2, 3, 4], _cfg())
    pool, test = t.X_pool[:, 0, 0].astype(int), t.X_test[:, 0, 0].astype(int)
    assert set(subj[pool]) == set(subj[test]) == {2}
    assert not set(pool) & set(test)
    assert set(test) == {i for i in te if subj[i] == 2}           # the within-subject test
    prev = set()
    for n in (2, 5, 10):
        kept = set(pool[t.labeled[n]])
        assert all((y[list(kept)] == c).sum() == n for c in range(4))
        assert prev <= kept                                        # nested in N
        prev = kept
        own = [i for i in tr if subj[i] == 2], [i for i in va if subj[i] == 2]
        a, b, _ = reduce_pool(*own, y, n, 0.2, "random", subject_rng(3, 1))
        assert kept == set(a + b)                                  # the datasize trials


def test_validation_of_the_target_fields():
    from rfl.federated.fl_config import FLConfig
    from rfl.utils.config import ExperimentConfig
    with pytest.raises(ValueError, match="need target_subjects"):
        ExperimentConfig(target_ns=[5])
    with pytest.raises(ValueError, match="not among"):
        ExperimentConfig(subjects=[1, 2], target_subjects=[3])
    with pytest.raises(ValueError, match="no source"):
        ExperimentConfig(subjects=[1, 2], target_subjects=[1, 2])
    with pytest.raises(ValueError, match="local-only"):
        ExperimentConfig(subjects=[1, 2], target_subjects=[1], target_role="local",
                         train_trials_per_class=5)
    with pytest.raises(ValueError, match="client-partition"):
        FLConfig(subjects=[1, 2], target_subjects=[1], fl_client_partition="iid")
    with pytest.raises(ValueError, match="'adapt' adapts"):
        FLConfig(subjects=[1, 2], target_subjects=[1], fl_local_only=True)
    FLConfig(subjects=[1, 2], target_subjects=[1], target_role="local",
             fl_local_only=True, train_trials_per_class=5)
    with pytest.raises(ValueError, match="combines the FedBN layers"):
        FLConfig(subjects=[1, 2], target_subjects=[1], target_ns=[0],
                 spdnet_use_batch_norm=True, spdnet_bn_mean_type="bures_wasserstein")
    # The within-subject configuration and hash are those of the code before.
    cfg = ExperimentConfig()
    assert not {"target_subjects", "target_ns", "target_role"} & set(cfg.to_dict())
    assert "target_subjects" in ExperimentConfig(subjects=[1, 2],
                                                 target_subjects=[1]).to_dict()


def test_inert_target_options_are_refused():
    from rfl.experiments.run_federated_experiment import prepare_args
    with pytest.raises(SystemExit):
        prepare_args(["--dataset", "BNCI2014_001", "--target-ns", "5"])
    with pytest.raises(SystemExit):
        prepare_args(["--dataset", "BNCI2014_001", "--target-subjects", "1",
                      "--target-role", "local", "--target-ns", "5"])


# ------------------------------------------------------------------------- heads
def test_head_matches_scikit_learn_and_stays_at_its_start_when_penalised():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    F = rng.normal(size=(60, 5))
    y = (F[:, 0] + 0.5 * rng.normal(size=60) > 0).astype(int) + (F[:, 1] > 1)
    W0, b0 = np.zeros((3, 5)), np.zeros(3)
    W, b = L.fit_head(F, y, 3, W0, b0, C=1.0)
    sk = LogisticRegression(C=1.0, max_iter=10000, tol=1e-10).fit(F, y)
    assert np.allclose(L._logits(F, W, b).argmax(1), sk.predict(F))
    assert np.allclose(W - W.mean(0), sk.coef_ - sk.coef_.mean(0), atol=1e-3)
    start = rng.normal(size=(3, 5))
    W, _ = L.fit_head(F, y, 3, start, b0, C=1e-8)
    assert np.allclose(W, start, atol=1e-4)
    fit = lambda X, Y, c: (lambda Z, p=L.fit_head(X, Y, 3, W0, b0, c): L._logits(Z, *p))
    assert L.choose_C(F, y, fit, L.C_GRID, 0) in L.C_GRID
    assert L.choose_C(F[:4], np.array([0, 1, 2, 2]), fit, L.C_GRID, 0) == 1.0


def test_fedbn_layers_of_a_new_subject_are_combined():
    from rfl.models.spd_batchnorm import weighted_spd_mean
    from rfl.models.spdnet import create_spdnet
    models = [create_spdnet(n_classes=2, n_channels=4, hidden_dim=3, use_batch_norm=True,
                            seed=s) for s in range(3)]
    states = []
    for i, m in enumerate(models):
        st = m.state_dict()
        A = torch.randn(3, 3, dtype=st["batch_norm.running_mean"].dtype)
        st["batch_norm.running_mean"] = A @ A.T + (i + 1) * torch.eye(3, dtype=A.dtype)
        st["batch_norm.parametrizations.Covbias.original"] = A + A.T
        states.append(st)
    target = create_spdnet(n_classes=2, n_channels=4, hidden_dim=3, use_batch_norm=True,
                           seed=9)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    L.combine_local_norms(target, states, [1.0, 1.0, 2.0])
    after = target.state_dict()
    want = weighted_spd_mean([s["batch_norm.running_mean"] for s in states], [1, 1, 2],
                             target.batch_norm.mean_type)
    assert torch.allclose(after["batch_norm.running_mean"], want)
    k = "batch_norm.parametrizations.Covbias.original"   # the bias: GAH of the f(S_i)
    f = target.batch_norm.parametrizations.Covbias[0]
    with torch.no_grad():
        want = weighted_spd_mean([f(s[k]) for s in states], [1, 1, 2],
                                 target.batch_norm.mean_type)
        assert torch.allclose(f(after[k]), want, atol=1e-6)
    for k in ("classifier.weight", "bimap.weight"):  # the rest is left as it was
        if k in before:
            assert torch.equal(after[k], before[k])


# ----------------------------------------------------------------------- runners
@pytest.mark.parametrize("model", ["spdnet", "eegnet"])
def test_federated_run_sets_the_targets_apart_and_adapts(fake_moabb, tmp_path, model):
    net = (SPD + ["--use-batch-norm"] if model == "spdnet" else
           EEG + ["--eegnet-norm-type", "batch"])
    r = _run_fed(tmp_path, *net, "--target-subjects", "4",
                 "--target-ns", "0", "2", "5")
    info = r["fl_info"]
    assert info["client_subjects"] == [[1], [2], [3]]
    assert info["target_subjects"] == [4]
    tm = r["target_metrics"]
    assert tm["subjects"] == [4] and tm["ns"] == [0, 2, 5]
    got = {(x["n"], x["stats"], x["head"]) for x in tm["records"]}
    want = {(0, "none", "shared"), (0, "pool", "shared")}
    for n in (2, 5):
        want |= {(n, "labeled", "shared")}
        want |= {(n, s, h) for s in ("none", "pool", "labeled") for h in ("shared_ft", "local")}
    assert got == want
    assert all(0 <= x["acc"] <= 1 and x["n_test"] == tm["test_sizes"]["4"]
               for x in tm["records"])
    # Every trial of the target is in its pool or its test, none in the clients.
    assert tm["pool_sizes"]["4"] + tm["test_sizes"]["4"] == 4 * PER_CLASS
    assert sum(info["client_train_samples"]) + sum(info["client_val_samples"]) < \
        3 * 4 * PER_CLASS


def test_fedper_has_no_shared_head_for_a_new_subject(fake_moabb, tmp_path):
    r = _run_fed(tmp_path, *SPD, "--use-batch-norm", "--local-fc", "--target-subjects", "4",
                 "--target-ns", "0", "5")
    heads = {(x["n"], x["head"]) for x in r["target_metrics"]["records"]}
    assert heads == {(5, "local")}


def test_target_statistics_come_from_the_target_only(monkeypatch):
    from rfl.models.spdnet import create_spdnet
    from rfl.utils.config import ExperimentConfig
    X, y, meta = _fake_eeg()
    own = meta.subject.to_numpy() == 2
    Xs, ys = X[own], y[own]
    t = L.Target(2, Xs[:60], ys[:60], Xs[60:], ys[60:],
                 {5: np.flatnonzero(np.isin(np.arange(60), np.r_[0:60:3]))[:20]})
    cfg = ExperimentConfig(subjects=[1, 2], target_subjects=[2], target_ns=[0, 5],
                           spdnet_use_batch_norm=True, device="cpu")
    seen = []
    real = L.recalibrated
    monkeypatch.setattr(L, "recalibrated",
                        lambda m, ds, dev: seen.append(len(ds)) or real(m, ds, dev))
    model = create_spdnet(n_classes=4, n_channels=N_CH, hidden_dim=4, use_batch_norm=True)
    before = model.state_dict()["batch_norm.running_mean"].clone()
    out = L.evaluate_targets(model, [t], cfg, torch.device("cpu"), 4, shared_head=True,
                             verbose=False)
    assert seen == [60, 20]                       # the pool, then the 5 x 4 labeled trials
    assert torch.equal(model.state_dict()["batch_norm.running_mean"], before)  # a copy
    assert {r["stats"] for r in out["records"]} == {"none", "pool", "labeled"}


def test_local_role_trains_the_targets_alone(fake_moabb, tmp_path):
    r = _run_fed(tmp_path, *SPD, "--use-batch-norm", "--local-only", "--target-subjects",
                 "3", "4", "--target-role", "local", "--train-trials-per-class", "5",
                 "--bn-recalibrate")
    info = r["fl_info"]
    assert info["client_subjects"] == [[3], [4]] and info["reduced_clients"] == [0, 1]
    assert all(n == 4 * 4 for n in info["client_train_samples"])   # 4 of 5 per class
    assert r["target_metrics"] is None
    assert len(r["final_metrics"]["per_client_test_acc"]) == 2


def test_centralized_run_sets_the_targets_apart(fake_moabb, tmp_path):
    from rfl.experiments.run_centralized_experiment import main
    out = tmp_path / "central"
    main([*COMMON, *SPD, "--use-batch-norm", "--epochs", "2", "--target-subjects", "4",
          "--target-ns", "0", "5", "--save-dir", str(out)])
    r = json.loads((out / "centralized_results.json").read_text())
    d = r["dataset"]
    assert d["train_samples"] + d["val_samples"] + d["test_samples"] == 3 * 4 * PER_CLASS
    tm = r["target_metrics"]
    assert tm["subjects"] == [4]
    assert {(x["n"], x["head"]) for x in tm["records"]} == {
        (0, "shared"), (5, "shared"), (5, "shared_ft"), (5, "local")}


def test_tslr_loso(fake_moabb, tmp_path):
    from rfl.experiments.run_tslr import main
    out = tmp_path / "tslr"
    r = main(["--dataset", "BNCI2014_001", "--subjects", "1", "2", "3", "4", "--mode",
              "loso", "--target-subjects", "4", "--target-ns", "0", "5",
              "--save-dir", str(out)])
    assert r["fl_info"]["client_subjects"] == [[1], [2], [3]]
    got = {(x["n"], x["stats"], x["head"]) for x in r["target_metrics"]["records"]}
    assert got == {(0, "pool", "pooled"), (5, "pool", "pooled_plus"), (5, "pool", "local"),
                   (5, "labeled", "pooled"), (5, "labeled", "pooled_plus"),
                   (5, "labeled", "local")}
    with pytest.raises(SystemExit):
        main(["--dataset", "BNCI2014_001", "--mode", "loso", "--save-dir", str(out)])


# ---------------------------------------------------------------------- analysis
def test_analysis_tables_figure_and_tests(fake_moabb, tmp_path, monkeypatch):
    """Final runs of two folds and two seeds, laid out as the plans write them."""
    from rfl.analysis import loso as LA
    from rfl.analysis import paper
    from rfl.experiments.run_centralized_experiment import main as central
    from rfl.experiments.run_federated_experiment import main as fed
    from rfl.experiments.run_tslr import main as tslr
    root = tmp_path / "results"
    bn = SPD + ["--use-batch-norm"]
    for k, target in ((2, "3"), (3, "4")):
        tgt = ["--target-subjects", target]
        adapt = tgt + ["--target-ns", "0", "5"]
        for seed in ("101", "102"):
            def out(plan, arm):
                return str(root / f"mi_BNCI2014_001_{plan}" / "final" / f"{arm}_f{k}"
                           / "default" / f"seed{seed}")
            central([*COMMON, *bn, "--epochs", "2", *adapt, "--seed", seed,
                     "--save-dir", out("loso", "central_spdnet_bn")])
            for scope, flag in (("shared", []), ("fedper", ["--local-fc"])):
                fed([*COMMON, *bn, "--rounds", "2", *flag, *adapt, "--seed", seed,
                     "--save-dir", out("loso", f"fed_spdnet_projavg_{scope}_bn")])
            fed([*COMMON, *bn, "--rounds", "2", "--local-only", *tgt, "--target-role",
                 "local", "--train-trials-per-class", "5", "--seed", seed,
                 "--save-dir", out("loso", "fed_spdnet_local_bn_U5")])
            tslr(["--dataset", "BNCI2014_001", "--subjects", "1", "2", "3", "4",
                  "--mode", "loso", *adapt, "--seed", seed,
                  "--save-dir", out("loso_tslr", "tslr_loso")])
    runs = paper.final_runs(root, "BNCI2014_001", ("loso", "loso_eegnet", "loso_tslr"))
    rec = LA.records(runs)
    assert set(rec.subject) == {3, 4} and set(rec.seed) == {101, 102}
    assert set(rec.source) == {"central", "shared", "fedper", "local", "tslr"}
    assert set(rec.fold) == {2, 3}
    s = LA.summary(rec)
    assert (s["count"] == 2).all()                       # every curve on both seeds
    t = LA.paired_tests(rec)
    assert set(t.family) == {"recalibration", "new head", "federation"}
    assert set(zip(t.n, t.stats)) <= {(0, "pool"), (5, "labeled")}   # primary only
    # A seed without every target leaves the means, not only that target.
    partial = rec[~((rec.seed == 102) & (rec.subject == 4) & (rec.source == "shared"))]
    s2 = LA.summary(partial)
    assert set(s2[s2.source == "shared"]["count"]) == {1}
    assert set(s2[s2.source == "central"]["count"]) == {2}
    out = tmp_path / "paper"
    with pytest.raises(SystemExit):              # the inter-subject study is opt-in
        paper.main(["BNCI2014_001", "--results", str(root), "--out", str(out)])
    paper.main(["BNCI2014_001", "--results", str(root), "--out", str(out), "--loso"])
    for name in ("tab_loso.tex", "fig_loso.tex", "tab_loso_tests.tex", "loso_tests.csv",
                 "tab_loso_subjects_spdnet_N0.tex", "tab_loso_subjects_spdnet_N5.tex",
                 "results.tex", "clients.csv"):
        assert (out / name).exists(), name
    tab = (out / "tab_loso.tex").read_text()
    assert "Local from scratch" in tab and "TS+LR" in tab and "target only" in tab
    assert "\\input{tab_loso}" in (out / "results.tex").read_text()
    clients = pd.read_csv(out / "clients.csv")
    assert set(clients.study) == {"loso"} and len(clients) == len(rec)


def test_fedbn_combination_with_a_scalar_variance():
    from rfl.models.spdnet import create_spdnet
    models = [create_spdnet(n_classes=2, n_channels=4, hidden_dim=3, use_batch_norm=True,
                            bn_type="mean_var_scalar", seed=s) for s in range(2)]
    target = create_spdnet(n_classes=2, n_channels=4, hidden_dim=3, use_batch_norm=True,
                           bn_type="mean_var_scalar", seed=5)
    L.combine_local_norms(target, [m.state_dict() for m in models], [1.0, 1.0])
    for k, v in target.state_dict().items():
        if k.startswith("batch_norm.") and "Covbias" not in k and v.is_floating_point():
            avg = (models[0].state_dict()[k] + models[1].state_dict()[k]) / 2
            assert torch.allclose(v, avg.to(v.dtype))
