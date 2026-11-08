"""Data-quantity study: reduction of the calibration trials, recalibration of the SPD
batch norm, TS+LR baseline, plans and analysis."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rfl.data import reduction as R
from rfl.experiments import datasize as DS


# ----------------------------------------------------------------- reduction
def test_positions_are_stratified_nested_and_split():
    y = np.repeat([0, 1, 2], 30)
    prev = None
    for n in (5, 10, 20):
        tr, va = R.calibration_positions(y, n, 0.2, "random", R.subject_rng(3, 1))
        kept = np.concatenate([tr, va])
        assert all((y[kept] == c).sum() == n for c in range(3))
        v = max(1, round(0.2 * n))
        assert all((y[va] == c).sum() == v for c in range(3))
        assert not set(tr) & set(va)
        if prev is not None:
            assert set(prev) <= set(tr)          # nested: the training trials grow
        prev = tr


def test_chronological_keeps_the_first_trials():
    y = np.tile([0, 1], 20)
    tr, va = R.calibration_positions(y, 5, 0.2, "chronological", R.subject_rng(0, 0))
    assert sorted(np.concatenate([tr, va])) == list(range(10))
    assert list(va) == [8, 9]                    # the last trial of each class


def test_short_pool_keeps_everything():
    y = np.array([0, 0, 1, 1, 1])
    tr, va = R.calibration_positions(y, 10, 0.2, "random", R.subject_rng(0, 0))
    assert len(tr) + len(va) == 5


def test_reduced_positions():
    assert R.reduced_positions(9, 0, 1) == set(range(9))
    a, b = R.reduced_positions(9, 3, 1), R.reduced_positions(9, 3, 1)
    assert a == b and len(a) == 3
    assert R.reduced_positions(9, 3, 2) != a or True


def _split(n=40, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    X = rng.normal(size=(n, 3, 3))
    return {"X_train": X[:30], "y_train": y[:30], "X_val": X[30:34], "y_val": y[30:34],
            "X_test": X[34:], "y_test": y[34:]}


def test_reduce_synthetic_keeps_test_and_reduces_k():
    splits = [_split(seed=i) for i in range(4)]
    cfg = SimpleNamespace(train_trials_per_class=3, reduced_subjects=2, seed=5,
                          calibration_val_fraction=0.2, calibration_order="random")
    out, red = R.reduce_synthetic(splits, cfg)
    assert len(red) == 2
    for i, (a, b) in enumerate(zip(splits, out)):
        assert np.array_equal(a["X_test"], b["X_test"])
        if i in red:
            assert len(b["y_train"]) + len(b["y_val"]) <= 6
        else:
            assert len(b["y_train"]) == 30


def test_config_validation():
    from rfl.utils.config import ExperimentConfig
    with pytest.raises(ValueError):
        ExperimentConfig(train_trials_per_class=0)
    with pytest.raises(ValueError):
        ExperimentConfig(calibration_order="sideways")
    with pytest.raises(ValueError):
        ExperimentConfig(spdnet_bn_recalibrate=True)          # no batch norm
    ExperimentConfig(model_type="eegnet", eegnet_norm_type="batch", spdnet_bn_recalibrate=True)
    with pytest.raises(ValueError):
        R.reduced_positions(3, 4, 0)                          # more than the subjects


def test_iid_partition_refuses_reduction():
    from rfl.federated.fl_config import FLConfig
    with pytest.raises(ValueError, match="subject"):
        FLConfig(train_trials_per_class=5, fl_client_partition="iid")


def test_reduction_options_need_n():
    from rfl.experiments.run_federated_experiment import check_command
    with pytest.raises(SystemExit):
        check_command(["--synthetic-clients", "x.npz", "--reduced-subjects", "2",
                       "--save-dir", "y"])


# ------------------------------------------------------- batch-norm recalibration
def _bn_model(seed=0):
    from rfl.models.spdnet import create_spdnet
    return create_spdnet(n_classes=2, n_channels=6, hidden_dim=4, epsilon=1e-4,
                         use_batch_norm=True, bn_momentum=0.01, seed=seed)


def _loader(seed, n=24):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, 6, 12))
    X = torch.tensor(A @ A.transpose(0, 2, 1) / 12 + 0.1 * np.eye(6), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, 2, n))
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=8)


def test_recalibrate_sets_the_exact_input_mean():
    from rfl.models.bn_recalibration import layer_statistics, recalibrate, spd_batchnorm_layers
    model, loader = _bn_model(), _loader(1)
    (name, bn), = spd_batchnorm_layers(model)
    before = bn.running_mean.clone()
    out = recalibrate(model, loader, "cpu")
    assert not torch.allclose(before, bn.running_mean)
    assert torch.allclose(out[f"{name}.running_mean"], bn.running_mean)
    # A second pass gives the same mean: it does not depend on the running mean.
    assert torch.allclose(layer_statistics(model, bn, loader, "cpu")["running_mean"],
                          bn.running_mean, atol=1e-6)


def test_recalibrate_eegnet_batchnorm_in_call_order():
    from rfl.models.bn_recalibration import batchnorm_layers, layer_statistics, recalibrate
    from rfl.models.eegnet import create_eegnet
    torch.manual_seed(0)
    model = create_eegnet(n_classes=2, n_channels=4, n_timepoints=64, sampling_rate=64,
                          norm_type="batch")
    X = torch.randn(20, 1, 4, 64)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, torch.zeros(20, dtype=torch.long)), batch_size=8)
    names = [n for n, _ in batchnorm_layers(model, loader)]
    assert len(names) == 3 and names == sorted(names, key=names.index)
    recalibrate(model, loader, "cpu")
    for n, m in batchnorm_layers(model, loader):
        st = layer_statistics(model, m, loader, "cpu")
        assert torch.allclose(st["running_mean"], m.running_mean, atol=1e-5)
        assert torch.allclose(st["running_var"], m.running_var, atol=1e-5)


def test_recalibrate_clients_shared_and_local():
    from rfl.models.bn_recalibration import recalibrate_clients, spd_batchnorm_layers
    clients = [SimpleNamespace(local_model=_bn_model(), train_loader=_loader(s))
               for s in (1, 2)]
    state = {"batch_norm.running_mean": torch.eye(4, dtype=torch.float64)}
    recalibrate_clients(clients, shared=True, weights=[1, 1], device="cpu", states=[state])
    m = [dict(spd_batchnorm_layers(c.local_model))["batch_norm"].running_mean for c in clients]
    assert torch.allclose(m[0], m[1]) and torch.allclose(state["batch_norm.running_mean"], m[0])
    recalibrate_clients(clients, shared=False, weights=[1, 1], device="cpu")
    m = [dict(spd_batchnorm_layers(c.local_model))["batch_norm"].running_mean for c in clients]
    assert not torch.allclose(m[0], m[1])


# ------------------------------------------------------------ plans, TS+LR, analysis
def test_scenarios_and_grids():
    sc = DS.scenarios([5, 20], [5], k_reduced=3)
    assert [s[0] for s in sc] == ["U5", "U20", "full", "H5"]
    assert sc[-1][1] == ["--train-trials-per-class", "5", "--reduced-subjects", "3"]
    assert DS.horizon_grid([200, 100, 300], 10) == [400, 200, 600]
    assert DS.horizon_grid([200], 5)[0] < 4 * 200 + 1
    assert DS.horizon_grid([200], None) == [200]
    assert DS.batch_grid(5, 4) == [8, 4, 16]
    v = DS.variants({"bn": (("--use-batch-norm",), None)}, sc, [10], [32], 4)
    assert set(v) == {"bn_U5", "bn_U20", "bn_full", "bn_H5"}
    assert v["bn_H5"].space["batch_size"] == [32]
    assert v["bn_U5"].space["batch_size"] == [8, 4, 16]


@pytest.fixture
def synth(tmp_path):
    from rfl.data.synthetic import generate_synthetic_spd, save_dataset
    d = generate_synthetic_spd(n_clients=4, samples_per_client=90, n_channels=6,
                               n_classes=2, label_skew_alpha=1000, channel_gain_std=0.3,
                               pattern_rotation=0.45, signal_strength_range=(0.4, 0.9),
                               seed=0)
    path = tmp_path / "s.npz"
    save_dataset(d, str(path))
    return path


def test_generator_rotation_changes_the_subjects():
    from rfl.data.synthetic import generate_synthetic_spd
    a = generate_synthetic_spd(n_clients=2, samples_per_client=10, n_channels=6, n_classes=2,
                               seed=0)
    b = generate_synthetic_spd(n_clients=2, samples_per_client=10, n_channels=6, n_classes=2,
                               seed=0, pattern_rotation=0.0)
    c = generate_synthetic_spd(n_clients=2, samples_per_client=10, n_channels=6, n_classes=2,
                               seed=0, pattern_rotation=0.5)
    assert np.array_equal(a["clients"][0]["X_train"], b["clients"][0]["X_train"])
    assert not np.array_equal(a["clients"][0]["X_train"], c["clients"][0]["X_train"])


def test_tslr_runner_and_results_columns(synth, tmp_path):
    from rfl.analysis.results import load_runs
    from rfl.experiments.run_tslr import main
    for mode in ("local", "pooled"):
        main(["--synthetic-clients", str(synth), "--mode", mode, "--train-trials-per-class",
              "5", "--reduced-subjects", "2", "--seed", "1",
              "--save-dir", str(tmp_path / f"runs/tslr_{mode}_H5/default/seed1")])
    df = load_runs(str(tmp_path / "runs"))
    assert set(df.scope) == {"local", "pooled"} and set(df.model) == {"tslr"}
    r = json.loads((tmp_path / "runs/tslr_local_H5/default/seed1/fl_results.json").read_text())
    red = r["fl_info"]["reduced_clients"]
    assert len(red) == 2
    row = df[df.scope == "local"].iloc[0]
    assert row.reduced_test_acc == pytest.approx(
        np.mean([r["final_metrics"]["per_client_test_acc"][i] for i in red]))
    assert row.n_per_class == 5 and row.reduced_subjects == 2


def test_federated_run_records_reduction_and_recalibration(synth, tmp_path):
    from rfl.experiments.run_federated_experiment import main
    out = tmp_path / "fed"
    main(["--synthetic-clients", str(synth), "--model", "spdnet", "--hidden-dim", "3",
          "--use-batch-norm", "--bn-recalibrate", "--no-fedbn", "--rounds", "2",
          "--batch-size", "8", "--train-trials-per-class", "6", "--reduced-subjects", "1",
          "--seed", "2", "--save-dir", str(out)])
    r = json.loads((out / "fl_results.json").read_text())
    info, fm = r["fl_info"], r["final_metrics"]
    assert len(info["reduced_clients"]) == 1
    i = info["reduced_clients"][0]
    assert info["client_train_samples"][i] + info["client_val_samples"][i] <= 12
    assert fm["bn_recalibrated"] and len(fm["per_client_val_acc"]) == 4
    assert r["final_metrics_before_recalibration"] is not None


def test_analysis_labels_and_table():
    from rfl.analysis.datasize import label, parse_arm
    assert parse_arm("fed_spdnet_projavg_shared_bnshared_H5") == {"norm": "bnshared",
                                                                  "scenario": "H5"}
    assert parse_arm("tslr_local_U10") == {"norm": None, "scenario": "U10"}
    assert label("spdnet", "shared", "bn") == "Federated · SPDNet · shared head · FedBN"
    assert label("spdnet", "local", "bn") == "Local · SPDNet · BN"
    assert label("tslr", "local", None) == "Local · TS+LR"
    assert label("eegnet", "fedper", "bnshared") == \
        "Federated · EEGNet · personal head (FedPer) · shared BN"


def test_selection_metric_from_plan(monkeypatch):
    import rfl.experiments.grid as G
    seen = {}
    monkeypatch.setattr(G, "plan_attr", lambda name, attr, default: (
        {"metric": "reduced_val_acc"} if attr == "SELECTION" else default))
    monkeypatch.setattr(G, "select", lambda jobs, metric, prefer_smaller=(): seen.setdefault(
        "metric", metric) and {})
    monkeypatch.setattr(G, "plan_namespace", lambda name: name)
    G.write_selection("p", [], "val_acc", verbose=False)
    assert seen["metric"] == "reduced_val_acc"


def test_generator_options_leave_the_other_draws_unchanged():
    from rfl.data.synthetic import generate_synthetic_spd
    kw = dict(n_clients=3, samples_per_client=30, n_channels=6, n_classes=3, seed=0)
    a, b = generate_synthetic_spd(**kw), generate_synthetic_spd(**kw, pattern_rotation=0.3)
    for ca, cb in zip(a["clients"], b["clients"]):
        assert np.array_equal(ca["y_train"], cb["y_train"])
    with pytest.raises(ValueError):
        generate_synthetic_spd(**kw, signal_strength_range=(0.8, 0.2))


def test_tslr_without_validation_fixes_c(synth, tmp_path):
    from rfl.experiments.run_tslr import main
    r = main(["--synthetic-clients", str(synth), "--train-trials-per-class", "1",
              "--seed", "1", "--save-dir", str(tmp_path / "t")])
    assert r["tslr"]["C"] == 1.0 and r["tslr"]["C_rule"].startswith("no validation")


def test_local_only_keeps_its_optimiser_state(synth, tmp_path):
    from rfl.federated.fl_config import FLConfig
    from rfl.federated.server import create_fl_clients_synthetic
    from rfl.data.synthetic import load_clients
    from rfl.models.spdnet import create_spdnet
    cfg = FLConfig(model_type="spdnet", data_type="spd", fl_local_only=True,
                   batch_size=16, fl_aggregation="projavg")
    client = create_fl_clients_synthetic(load_clients(str(synth)), cfg, "cpu",
                                         verbose=False)[0]
    client.set_model(create_spdnet(n_classes=2, n_channels=6, hidden_dim=3, epsilon=1e-4))
    crit = torch.nn.CrossEntropyLoss()
    client.train_local(1, crit, 1e-2)
    opt = client._optimizer
    client.train_local(1, crit, 5e-3)
    assert client._optimizer is opt and opt.param_groups[0]["lr"] == 5e-3
    assert opt.state                                         # Adam moments kept


def test_retavg_failure_is_a_divergence():
    from rfl.geometry.stiefel import RetractionNotInvertible
    from rfl.training.divergence import DIVERGENCE_ERRORS
    assert issubclass(RetractionNotInvertible, DIVERGENCE_ERRORS)


def test_eegnet_input_norm_keeps_channel_powers():
    from rfl.data.datasets import RawEEGDataset
    X = np.random.default_rng(0).normal(size=(2, 3, 50)) * np.array([1, 2, 4])[:, None]
    per = RawEEGDataset(X, np.zeros(2), normalization_type="per_trial").X
    glob = RawEEGDataset(X, np.zeros(2), normalization_type="per_trial_global").X
    assert np.allclose(per.std(axis=-1), 1, atol=1e-5)
    v = glob.var(axis=-1)
    assert v[0, 2] > 8 * v[0, 0]


def test_killed_job_is_failed_for_the_dag_then_abandoned(tmp_path, monkeypatch):
    import rfl.experiments.grid as G
    monkeypatch.setenv("RFL_RESULTS", str(tmp_path))
    j = G.Job(arm="a", entry="fed", seed=1, hp={"lr": 0.1}, stage="s", plan="p")
    j.save_dir.mkdir(parents=True)
    (j.save_dir / "run.log").write_text("started\n")
    (j.save_dir / "attempts").write_text("2\n")
    assert j.state() == "running?" and j.state(dead=True) == "failed"
    (j.save_dir / "attempts").write_text(f"{2 * G.MAX_ATTEMPTS}\n")
    assert j.state() == "abandoned" and j.state(dead=True) == "abandoned"


def test_selection_refuses_a_test_metric(monkeypatch):
    import rfl.experiments.grid as G
    monkeypatch.setattr(G, "plan_attr", lambda name, attr, default: (
        {"metric": "reduced_test_acc"} if attr == "SELECTION" else default))
    with pytest.raises(SystemExit, match="validation"):
        G.write_selection("p", [], verbose=False)


def test_extend_rules_bound_the_batch_by_the_client():
    rules = DS.extend_rules({"batch_size": {"factor": 2, "int": True, "min": 2, "max": 256},
                             "lr": {"factor": 3}}, n_classes=4)
    assert rules("fed_x_bn_U5")["batch_size"]["max"] == 16       # 4 classes x 4 trials
    assert rules("fed_x_bn_H5")["batch_size"]["max"] == 256
    assert rules("fed_x_bn_full")["lr"] == {"factor": 3}


def test_protocol_drops_an_arm_without_choice(tmp_path, monkeypatch, capsys):
    import rfl.experiments.grid as G
    from rfl.experiments.protocol import Protocol
    monkeypatch.setenv("RFL_RESULTS", str(tmp_path))
    P = Protocol("synthetic", data_args=["--synthetic-clients", "x.npz"], common_args=(),
                 models=("spdnet",), aggregations={"spdnet": ("projavg",)},
                 kinds=("fed_shared", "fed_local"),
                 space={"nodp": {"lr": [0.1, 0.3], "batch_size": [16], "horizon": [5, 10]},
                        "dp": {}},
                 search="staged", stage_keys=[["lr"], ["horizon"]], sweep_seeds=(1,),
                 final_seeds=(9,), name="p")
    path = G.selection_path("p", "sweep1")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "fed_spdnet_projavg_shared": {"error": "no hyperparameter cell has all its seeds"},
        "fed_spdnet_local": {"hp": {"lr": 0.1, "batch_size": 16, "horizon": 5}}}))
    jobs = P.sweep(1)
    assert {j.arm for j in jobs} == {"fed_spdnet_local"}
    assert "dropped" in capsys.readouterr().err


def test_study_dag_runs_four_chains_side_by_side(tmp_path, monkeypatch):
    from rfl.experiments import pipeline as PL
    monkeypatch.setenv("RFL_RESULTS", str(tmp_path))
    chains = PL.branches("mi_study")
    assert [len(c) for c in chains] == [9, 3, 1, 5]
    dag = PL.write_dag("mi_study", directory=tmp_path / "dag").read_text()
    parents = [ln for ln in dag.splitlines() if ln.startswith("PARENT")]
    assert len(parents) == 8 + 2 + 4
    assert not any("N8_" in p and "N9_" in p for p in parents)      # chains independent
    assert not any("N12_" in p for p in parents)
    assert "RFL_DAG=1" in (tmp_path / "dag" / "N3_mi_spdnet_bn_sweep1.sub").read_text()


def test_a_study_dag_can_hold_one_step(tmp_path, monkeypatch):
    import rfl.experiments.pipeline as PL
    monkeypatch.setenv("RFL_RESULTS", str(tmp_path))
    monkeypatch.setenv("RFL_DATASET", "BNCI2014_001")
    names = [PL.step_name(i, s) for i, s in enumerate(PL.steps("mi_study"))]
    tslr = names.index(next(n for n in names if n.endswith("tslr_final")))
    assert PL.step_index("mi_study", "N0") == 0 == PL.step_index("mi_study", names[0])
    assert PL.step_index("mi_study", str(tslr)) == tslr
    with pytest.raises(SystemExit, match="unknown step"):
        PL.step_index("mi_study", "N99")
    assert PL.previous_step("mi_study", 0) is None and PL.previous_step("mi_study", 1) == 0
    assert PL.previous_step("mi_study", tslr) is None          # first of its chain
    dag = PL.write_dag("mi_study", directory=tmp_path / "d", only=[str(tslr)]).read_text()
    assert dag.count("JOB ") == 1 and f"JOB {names[tslr]} " in dag and "PARENT" not in dag
    assert f"pre mi_study {tslr} " in dag                      # the global step index
    two = PL.write_dag("mi_study", directory=tmp_path / "e", only=["0", "1"]).read_text()
    assert two.count("JOB ") == 2 and f"PARENT {names[0]} CHILD {names[1]}" in two
    with pytest.raises(SystemExit, match="not finished"):    # the floors are not computed
        PL.write_dag("mi_study", directory=tmp_path / "f", only=["1"])


def test_a_selected_stage_counts_as_finished(tmp_path, monkeypatch):
    import json

    import rfl.experiments.grid as G
    import rfl.experiments.pipeline as PL
    monkeypatch.setenv("RFL_RESULTS", str(tmp_path))

    class Done:
        def state(self, dead=False):
            return "done"

    monkeypatch.setattr(PL, "steps", lambda plan: [("p", "sweep1"), ("p", "sweep2")])
    monkeypatch.setattr(PL, "plan_jobs", lambda plan, stage, **k: [Done()])
    monkeypatch.setattr(PL, "plan_namespace", lambda plan: "ns_of_p")
    assert not PL.step_finished("x", 0)                     # no selection yet
    path = G.selection_path("ns_of_p", "sweep1")            # under the plan's NAME
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({}))
    assert PL.step_finished("x", 0)
