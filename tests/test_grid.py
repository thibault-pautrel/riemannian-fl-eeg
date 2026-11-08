"""Experiment plans (rfl/experiments/grid.py, rfl/experiments/plans/) and the HTCondor scripts."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from rfl.experiments import grid as G

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("RFL_RESULTS", str(tmp_path))
    return tmp_path


def _parse(job):
    """Parse a job's arguments with the parser of its runner, as run_one.sh would run it."""
    if job.entry == "fed":
        from rfl.experiments.run_federated_experiment import _build_arg_parser
    else:
        from rfl.experiments.run_centralized_experiment import _build_arg_parser
    return _build_arg_parser().parse_args(job.runner_args() + ["--save-dir", "x"])


def test_job_line_and_directory(root):
    j = G.Job("arm", "fed", 3, {"lr": 0.01, "batch_size": 64, "horizon": 100},
              ("--dataset", "BNCI2014_001"), stage="s", plan="p")
    assert j.save_dir == root / "p" / "s" / "arm" / "lr0.01_B64_T100" / "seed3"
    fields = j.line().split()
    assert fields[:2] == [str(j.save_dir), "fed"]
    ns = _parse(j)
    assert (ns.seed, ns.learning_rate, ns.batch_size, ns.rounds) == (3, 0.01, 64, 100)
    with pytest.raises(ValueError):
        G.Job("a", "fed", 1, {"lr": 0.1}, ("--x", "has space"), "s", "p").line()


def test_states(root):
    j = G.Job("a", "fed", 1, {"lr": 0.1, "batch_size": 8, "horizon": 2}, (), "s", "p")
    assert j.state() == "pending"
    j.save_dir.mkdir(parents=True)
    (j.save_dir / "queued").write_text("")
    assert j.state() == "queued"
    (j.save_dir / "queued").unlink()                   # run_one.sh at start
    (j.save_dir / "run.log").write_text("")
    assert j.state() == "running?"
    (j.save_dir / "exit_code").write_text("1\n")
    assert j.state() == "failed"
    (j.save_dir / "fl_results.json").write_text("{}")
    assert j.state() == "failed"                     # an old result, a failed rerun
    (j.save_dir / "exit_code").write_text("0\n")
    assert j.state() == "done"
    (j.save_dir / "job_args").write_text(j.command() + "\n")
    assert j.state() == "done"
    (j.save_dir / "job_args").write_text("fed --other\n")
    assert j.state() == "conflict"
    with pytest.raises(SystemExit, match="another command"):
        plan_with = type("P", (), {"STAGES": {"s": lambda: [G.Job(
            "a", "fed", 1, {"lr": 0.1, "batch_size": 8, "horizon": 2}, ())]}, "NAME": "p"})
        import sys
        sys.modules["rfl.experiments.plans._conflict_tmp"] = plan_with
        try:
            G.plan_jobs("_conflict_tmp", "s")
        finally:
            del sys.modules["rfl.experiments.plans._conflict_tmp"]


def test_values_keep_their_exact_text():
    assert G._fmt(0.00123456789) == "0.00123456789"
    assert G._fmt(1e-3) == "0.001" and G._fmt(2.0) == "2" and G._fmt(1e-5) == "1e-05"


def test_select_refuses_unfinished_stages_and_test_metrics(root, monkeypatch):
    jobs = G.grid("a", "fed", [], [1, 2], lrs=[0.1])
    for j in jobs:
        j.stage, j.plan = "sweep", "p"
    monkeypatch.setattr(G, "plan_jobs", lambda *a, **k: jobs)
    monkeypatch.setattr(G, "plan_namespace", lambda name: "p")
    monkeypatch.setattr(G, "require_plan", lambda name: None)
    _fake_result(jobs[0], 0.5)
    with pytest.raises(SystemExit, match="not finished"):
        G.main(["p", "--select"])
    assert G.main(["p", "--select", "--allow-incomplete"]) == 0
    with pytest.raises(SystemExit):
        G.main(["p", "--select", "--metric", "test_acc"])


@pytest.mark.parametrize("plan,stage", [("smoke", None), ("mi_protocol", "sweep"),
                                        ("mi_spdnet_bn_calib", "sweep1")])
def test_plan_jobs_parse_with_their_runner(root, plan, stage, monkeypatch):
    if plan == "smoke":
        monkeypatch.setattr("rfl.experiments.plans.smoke.prepare", lambda: None, raising=False)
    if plan == "mi_spdnet_bn_calib":
        _write_floors("BNCI2014_001", ["16"])
    jobs = G.plan_jobs(plan, stage)
    assert jobs and len({j.save_dir for j in jobs}) == len(jobs)
    for j in jobs:
        ns = _parse(j)
        assert ns.seed == j.seed
        if j.entry == "fed" and getattr(ns, "dp_enabled", False):
            assert ns.batch_size >= 64            # DP arms use larger batches


def test_any_runner_option_can_be_swept(root):
    jobs = G.grid("arm", "fed", ["--dataset", "BNCI2014_001", "--dp-enabled"], seeds=[1],
                  lrs=[0.1], space={"--dp-clip-C": [0.5, 2.0], "--hidden-dim": [8]})
    for j in jobs:
        j.stage, j.plan = "s", "p"
    assert len(jobs) == 2
    assert {j.hp_name for j in jobs} == {"lr0.1_dp-clip-C=0.5_hidden-dim=8",
                                          "lr0.1_dp-clip-C=2_hidden-dim=8"}  # 2.0 -> "2"
    ns = _parse(jobs[0])
    assert (ns.dp_clip_C, ns.hidden_dim) == (0.5, 8)
    with pytest.raises(ValueError):          # swept and fixed at the same time
        G.grid("arm", "fed", ["--hidden-dim", "4"], [1], space={"--hidden-dim": [8]})
    with pytest.raises(ValueError):          # not written as a command-line option
        G.grid("arm", "fed", [], [1], space={"hidden_dim": [8]})


def test_check_catches_invalid_jobs_before_submission(root):
    ok = G.grid("a", "fed", ["--dataset", "BNCI2014_001"], [1], lrs=[0.1])
    typo = G.grid("b", "fed", ["--dataset", "BNCI2014_001", "--locl-fc"], [1], lrs=[0.1])
    bad = G.grid("c", "fed", ["--dataset", "BNCI2014_001", "--report", "last",
                              "--early-stop-patience", "5"], [1], lrs=[0.1])
    for j in ok + typo + bad:
        j.stage, j.plan = "s", "p"
    G.check_jobs(ok)
    with pytest.raises(SystemExit, match="locl-fc"):
        G.check_jobs(typo)
    with pytest.raises(SystemExit, match="invalid job c"):
        G.check_jobs(bad)


def test_protocol_separates_datasets_and_reuses_the_selection(root):
    from rfl.experiments.protocol import Protocol
    a, b = Protocol("BNCI2014_001"), Protocol("Cho2017")
    assert a.name != b.name
    with pytest.raises(ValueError):
        Protocol("Cho2017", sweep_seeds=(1, 2), final_seeds=(2, 3))
    P = Protocol("Cho2017", models=("spdnet",), kinds=("fed_shared",),
                 aggregations={"spdnet": ("projavg",)},
                 space={"nodp": {"lr": [1e-3, 1e-2], "horizon": [50]}},
                 model_space={"spdnet": {"--hidden-dim": [8, 16]}},
                 sweep_seeds=(1, 2), final_seeds=(7,))
    jobs = P.sweep()
    assert len(jobs) == 2 * 2 * 2
    for j in jobs:
        j.stage, j.plan = "sweep", P.name
        score = 0.6 + (0.1 if j.hp["--hidden-dim"] == 16 else 0) + (0.05 if j.lr == 1e-2 else 0)
        _fake_result(j, score)
    sel = G.select(jobs)
    G.selection_path(P.name).parent.mkdir(parents=True, exist_ok=True)
    G.selection_path(P.name).write_text(json.dumps(sel))
    arm = "fed_spdnet_projavg_shared"
    assert sel[arm]["hp"] == {"lr": 1e-2, "horizon": 50, "--hidden-dim": 16}
    final = P.final()
    assert len(final) == 1 and final[0].hp == sel[arm]["hp"]
    assert "--head-refit-epochs" in final[0].args
    for j in final:
        j.stage, j.plan = "final", P.name
    G.check_jobs(final)


def test_mi_protocol_final_needs_a_selection(root):
    with pytest.raises(SystemExit):
        G.plan_jobs("mi_protocol", "final")


def _fake_result(job, val_acc):
    job.save_dir.mkdir(parents=True, exist_ok=True)
    r = {"config": {"dataset": "D", "model_type": "spdnet", "fl_aggregation": "projavg",
                    "learning_rate": job.lr, "batch_size": job.batch_size,
                    "fl_rounds": job.horizon, "seed": job.seed},
         "final_metrics": {"global_val_acc": val_acc, "global_test_acc": 0.5},
         "last_round_metrics": {}, "round_history": {}}
    (job.save_dir / "fl_results.json").write_text(json.dumps(r))


def test_select_picks_the_best_mean_and_flags_edges(root):
    jobs = G.grid("arm", "fed", (), seeds=[1, 2], lrs=[1e-3, 1e-2, 1e-1],
                  batch_sizes=[32], horizons=[25, 50, 100])
    for j in jobs:
        j.stage, j.plan = "sweep", "p"
        score = {1e-3: 0.5, 1e-2: 0.7, 1e-1: 0.6}[j.lr] + (0.05 if j.horizon == 100 else 0)
        _fake_result(j, score + 0.01 * j.seed)
    sel = G.select(jobs)["arm"]
    assert (sel["lr"], sel["horizon"]) == (1e-2, 100)
    assert sel["n_seeds"] == 2 and sel["on_grid_edge"] == ["horizon"]
    assert sel["mean"] == pytest.approx(0.765)


def test_select_ignores_incomplete_cells(root):
    jobs = G.grid("arm", "fed", (), seeds=[1, 2], lrs=[1e-3, 1e-2], batch_sizes=[32],
                  horizons=[50])
    for j in jobs:
        j.stage, j.plan = "sweep", "p"
    _fake_result(jobs[0], 0.5)                                  # lr 1e-3, seed 1
    _fake_result(jobs[1], 0.6)                                  # lr 1e-3, seed 2
    _fake_result(jobs[2], 0.9)                                  # lr 1e-2, seed 1 only
    sel = G.select(jobs)["arm"]
    assert sel["lr"] == 1e-3 and sel["n_cells"] == 1


def test_run_one_refuses_a_folder_of_another_command_and_force_clears(tmp_path):
    import sys
    env = {**os.environ, "RFL_RESULTS": str(tmp_path), "RFL_VENV": str(tmp_path / "none"),
           "MNE_DATA": str(tmp_path / "mne_data"),
           "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]}
    d = tmp_path / "job"
    d.mkdir()
    (d / "job_args").write_text("fed --dataset X\n")
    (d / "fl_results.json").write_text("{}")
    r = subprocess.run(["bash", str(REPO / "condor/run_one.sh"), str(d), "fed", "--bad"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 4 and "another command" in r.stderr
    r = subprocess.run(["bash", str(REPO / "condor/run_one.sh"), str(d), "fed", "--bad"],
                       capture_output=True, text=True, env={**env, "FORCE": "1"})
    assert r.returncode != 0 and not (d / "fl_results.json").exists()   # old result gone
    assert (d / "job_args").read_text().strip() == "fed --bad"


def test_run_one_skips_finished_runs_and_records_exit_codes(tmp_path):
    import sys
    env = {**os.environ, "RFL_RESULTS": str(tmp_path), "RFL_VENV": str(tmp_path / "none"),
           "MNE_DATA": str(tmp_path / "mne_data"),
           "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]}
    done = tmp_path / "done"
    done.mkdir()
    (done / "fl_results.json").write_text("{}")
    r = subprocess.run(["bash", str(REPO / "condor/run_one.sh"), str(done), "fed"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "SKIP" in r.stdout
    bad = tmp_path / "bad"
    r = subprocess.run(["bash", str(REPO / "condor/run_one.sh"), str(bad), "fed",
                        "--no-such-flag"], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and (bad / "exit_code").read_text().strip() != "0"
    r = subprocess.run(["bash", str(REPO / "condor/run_one.sh"), str(bad), "other"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2


def test_shell_scripts_parse():
    for f in sorted((REPO / "condor").glob("*.sh")):
        assert subprocess.run(["bash", "-n", str(f)]).returncode == 0, f


def test_uncalibrated_dataset_warns():
    from rfl.experiments.run_federated_experiment import check_command
    with pytest.warns(UserWarning, match="PhysionetMI: no calibrated"):
        check_command(["--dataset", "PhysionetMI", "--model", "spdnet", "--save-dir", "x"])


def test_protocol_arms_cover_the_three_spdnet_aggregations(root):
    from rfl.experiments.protocol import Protocol
    arms = {a for a, *_ in Protocol("BNCI2014_001").arms()}
    for agg in ("projavg", "rlavg", "retavg"):
        for kind in ("fed_spdnet_{}_shared", "fed_spdnet_{}_fedper", "dp10_spdnet_{}_shared",
                     "dp10_spdnet_{}_fedper"):
            assert kind.format(agg) in arms
    assert "fed_eegnet_fedavg_shared" in arms and "fed_spdnet_local" in arms
    assert not any("eegnet_projavg" in a for a in arms)
    jobs = Protocol("BNCI2014_001").sweep()
    for j in jobs:
        j.stage, j.plan = "sweep", "p"
    agg = {j.arm: _parse(j).aggregation for j in jobs if j.entry == "fed"}
    assert agg["fed_spdnet_rlavg_fedper"] == "rlavg" and agg["dp10_spdnet_retavg_shared"] == "retavg"
    assert agg["fed_spdnet_local"] is None


def test_random_search_draws_a_subset(root):
    full = G.grid("a", "fed", [], [1, 2], lrs=[1, 2, 3, 4], horizons=[10, 20])
    sub = G.grid("a", "fed", [], [1, 2], lrs=[1, 2, 3, 4], horizons=[10, 20], sample=3)
    assert len(full) == 16 and len(sub) == 6
    cells = {tuple(j.hp.items()) for j in sub}
    assert len(cells) == 3 and cells <= {tuple(j.hp.items()) for j in full}
    again = G.grid("a", "fed", [], [1, 2], lrs=[1, 2, 3, 4], horizons=[10, 20], sample=3)
    assert [j.hp for j in again] == [j.hp for j in sub]          # reproducible


def test_staged_search_fixes_earlier_choices(root):
    from rfl.experiments.protocol import Protocol
    P = Protocol("Cho2017", models=("spdnet",), kinds=("dp_shared",),
                 aggregations={"spdnet": ("projavg",)}, search="staged",
                 stage_keys=[["lr"], ["--dp-clip-C"]],
                 space={"nodp": {}, "dp": {"lr": [0.01, 0.1], "batch_size": [64],
                                           "horizon": [50], "--dp-clip-C": [1.0, 0.5, 2.0]}},
                 sweep_seeds=(1, 2), final_seeds=(9,))
    assert list(P.stages()) == ["sweep1", "sweep2", "final"]
    s1 = P.sweep(0)
    assert {j.hp["--dp-clip-C"] for j in s1} == {1.0}           # first value
    assert {j.lr for j in s1} == {0.01, 0.1} and len(s1) == 4
    for j in s1:
        j.stage, j.plan = "sweep1", P.name
        _fake_result(j, 0.7 if j.lr == 0.1 else 0.5)
    sel = G.select(s1)
    G.selection_path(P.name, "sweep1").parent.mkdir(parents=True, exist_ok=True)
    G.selection_path(P.name, "sweep1").write_text(json.dumps(sel))
    s2 = P.sweep(1)
    assert {j.lr for j in s2} == {0.1}                          # selected at stage 1
    assert {j.hp["--dp-clip-C"] for j in s2} == {0.5, 1.0, 2.0} and len(s2) == 6
    for j in s2:
        j.stage, j.plan = "sweep2", P.name
    G.check_jobs(s2)
    with pytest.raises(ValueError):          # a swept key in no group
        Protocol("Cho2017", search="staged", stage_keys=[["lr"]])
    with pytest.raises(ValueError, match="typo"):
        Protocol("Cho2017", search="staged",
                 stage_keys=[["lr", "horizon"], ["--dp-clip-c"]])
    with pytest.raises(ValueError):
        Protocol("Cho2017", search="random")


def test_submit_dry_run_writes_the_jobs_and_passes_the_environment(root, monkeypatch):
    import sys
    env = {**os.environ, "RFL_RESULTS": str(root), "RFL_VENV": str(root / "venv"),
           "MNE_DATA": str(root / "mne"),
           "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]}
    r = subprocess.run(["bash", str(REPO / "condor/submit.sh"), "smoke", "--dry-run",
                        "--cpus", "2", "--memory", "4GB", "--short"],
                       capture_output=True, text=True, env=env, cwd=str(REPO))
    assert r.returncode == 0, r.stderr
    jobs = list((root / "condor" / "smoke").glob("jobs_*.txt"))
    assert len(jobs) == 1 and len(jobs[0].read_text().splitlines()) == 17
    cmd = r.stdout.splitlines()[-1]
    for macro in ("cpus\\ =\\ 2", "memory\\ =\\ 4GB", "venv\\ =\\ " + str(root / "venv"),
                  "results\\ =\\ " + str(root), "+isShortJob"):
        assert macro in cmd, macro
    # A dry run submits nothing, so it marks nothing as queued.
    assert not list((root / "smoke").rglob("queued"))


def test_a_resubmitted_failed_job_shows_as_queued(root):
    j = G.Job("a", "fed", 1, {"lr": 0.1}, (), "s", "p")
    j.save_dir.mkdir(parents=True)
    (j.save_dir / "run.log").write_text("")
    (j.save_dir / "exit_code").write_text("1\n")
    assert j.state() == "failed"
    import os as _os
    past = j.save_dir.stat().st_mtime - 60
    for f in ("run.log", "exit_code"):                 # the failed run, a minute ago
        _os.utime(j.save_dir / f, (past, past))
    (j.save_dir / "queued").write_text("")             # submit.sh --retry-failed
    assert j.state() == "queued"


def test_select_merges_arms_and_refuses_unknown_metrics(root, monkeypatch):
    a = G.grid("a", "fed", [], [1], lrs=[0.1, 0.2])
    b = G.grid("b", "fed", [], [1], lrs=[0.1])
    for j in a + b:
        j.stage, j.plan = "sweep", "p"
        _fake_result(j, 0.5)
    monkeypatch.setattr(G, "plan_namespace", lambda name: "p")
    monkeypatch.setattr(G, "require_plan", lambda name: None)
    monkeypatch.setattr(G, "plan_jobs", lambda *x, **k: a)
    G.main(["p", "--select"])
    monkeypatch.setattr(G, "plan_jobs", lambda *x, **k: b)
    G.main(["p", "--select"])                          # another set of arms (RFL_EPS)
    sel = json.loads(G.selection_path("p").read_text())
    assert set(sel) == {"a", "b"}
    with pytest.raises(SystemExit, match="not in the results"):
        G.main(["p", "--select", "--metric", "val_accuracy"])


def test_arms_may_not_fix_the_seed(root):
    with pytest.raises(ValueError, match="plan machinery"):
        G.grid("a", "fed", ["--seed", "3"], [1], lrs=[0.1])


def test_protocol_gives_each_arm_only_the_options_its_runner_accepts(root):
    from rfl.experiments.protocol import Protocol
    P = Protocol("BNCI2014_001", models=("spdnet",),
                 space={"nodp": {"lr": [1e-2], "--local-epochs": [1, 2],
                                 "--client-fraction": [0.5, 1.0]},
                        "dp": {"lr": [1e-1], "batch_size": [64]}})
    jobs = P.sweep()
    for j in jobs:
        j.stage, j.plan = "sweep", P.name
    central = [j for j in jobs if j.arm == "central_spdnet"]
    local = [j for j in jobs if j.arm == "fed_spdnet_local"]
    assert central and all("--local-epochs" not in j.hp for j in central)
    assert local and all("--client-fraction" not in j.hp for j in local)
    G.check_jobs(jobs)


def test_staged_final_refuses_a_stage_selected_again(root):
    from rfl.experiments.protocol import Protocol
    P = Protocol("Cho2017", models=("spdnet",), kinds=("fed_shared",),
                 aggregations={"spdnet": ("projavg",)}, search="staged",
                 stage_keys=[["lr"], ["horizon"]],
                 space={"nodp": {"lr": [0.01, 0.1], "horizon": [50, 100]}, "dp": {}},
                 sweep_seeds=(1,), final_seeds=(9,))
    arm = "fed_spdnet_projavg_shared"

    def write_sel(st, hp):
        G.selection_path(P.name, st).parent.mkdir(parents=True, exist_ok=True)
        G.selection_path(P.name, st).write_text(json.dumps({arm: {"hp": hp}}))

    write_sel("sweep1", {"lr": 0.1, "horizon": 50})
    for j in P.sweep(1):                               # stage 2 ran at lr = 0.1
        j.stage, j.plan = "sweep2", P.name
        _fake_result(j, 0.5)
    write_sel("sweep2", {"lr": 0.1, "horizon": 100})
    assert len(P.final()) == 1
    G.selection_path(P.name, "sweep1").write_text(
        json.dumps({arm: {"hp": {"lr": 0.01, "horizon": 50}}}))  # sweep1 selected again
    with pytest.raises(SystemExit, match="selected again"):
        P.final()


def test_run_one_reruns_a_folder_whose_last_run_failed(tmp_path):
    import sys
    env = {**os.environ, "RFL_RESULTS": str(tmp_path), "RFL_VENV": str(tmp_path / "none"),
           "MNE_DATA": str(tmp_path / "mne_data"),
           "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]}
    d = tmp_path / "job"
    d.mkdir()
    (d / "fl_results.json").write_text("{}")
    (d / "exit_code").write_text("137\n")              # result of an earlier run, then killed
    r = subprocess.run(["bash", str(REPO / "condor/run_one.sh"), str(d), "fed", "--bad"],
                       capture_output=True, text=True, env=env)
    assert "SKIP" not in r.stdout and r.returncode != 0


def test_final_refuses_a_selection_older_than_the_sweep(root):
    """The grid was extended after the selection: final waits for a new selection."""
    from rfl.experiments.protocol import Protocol
    P = Protocol("Cho2017", models=("spdnet",), kinds=("fed_shared",),
                 aggregations={"spdnet": ("projavg",)},
                 space={"nodp": {"lr": [0.01, 0.1]}, "dp": {}}, sweep_seeds=(1,),
                 final_seeds=(9,))
    jobs = P.sweep()
    for j in jobs:
        j.stage, j.plan = "sweep", P.name
        _fake_result(j, 0.5)
    sel = G.select(jobs)
    G.selection_path(P.name).parent.mkdir(parents=True, exist_ok=True)
    G.selection_path(P.name).write_text(json.dumps(sel))
    assert len(P.final()) == 1
    P.space["nodp"]["lr"] = [0.01, 0.1, 1.0]           # extended grid, not run yet
    with pytest.raises(SystemExit, match="not finished"):
        P.final()


def test_run_one_resolves_relative_paths_from_the_submit_folder(tmp_path):
    """The plan's relative paths name files of the folder submit.sh ran in (RFL_WORKDIR)."""
    import sys
    wd = tmp_path / "wd"
    wd.mkdir()
    (wd / "s.npz").write_bytes(b"")
    env = {**os.environ, "RFL_RESULTS": str(tmp_path), "RFL_VENV": str(tmp_path / "none"),
           "MNE_DATA": str(tmp_path / "mne_data"), "RFL_WORKDIR": str(wd),
           "PYTHONPATH": str(REPO),
           "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]}
    d = tmp_path / "job"
    r = subprocess.run(["bash", str(REPO / "condor/run_one.sh"), str(d), "fed",
                        "--synthetic-clients", "s.npz", "--print-config"],
                       capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    cfg = json.loads((d / "run.log").read_text())
    assert cfg["data_path"] == str(wd / "s.npz")


def test_final_warns_about_folders_of_an_earlier_selection(root, capsys):
    from rfl.experiments.protocol import Protocol
    P = Protocol("Cho2017", models=("spdnet",), kinds=("fed_shared",),
                 aggregations={"spdnet": ("projavg",)},
                 space={"nodp": {"lr": [0.01, 0.1]}, "dp": {}}, sweep_seeds=(1,),
                 final_seeds=(9,))
    jobs = P.sweep()
    for j in jobs:
        j.stage, j.plan = "sweep", P.name
        _fake_result(j, 0.5)
    G.selection_path(P.name).parent.mkdir(parents=True, exist_ok=True)
    G.selection_path(P.name).write_text(json.dumps(G.select(jobs)))
    final = P.final()
    capsys.readouterr()
    old = G.results_root() / P.name / "final" / final[0].arm / "lr0.5"
    old.mkdir(parents=True)
    P.final()
    assert "earlier selection" in capsys.readouterr().err


def _fake_any(job, val_acc):
    """A result file of either runner, with the validation accuracy `val_acc`."""
    job.save_dir.mkdir(parents=True, exist_ok=True)
    if job.entry == "fed":
        return _fake_result(job, val_acc)
    r = {"config": {"dataset": "D", "model_type": "spdnet", "learning_rate": job.lr,
                    "batch_size": job.batch_size, "epochs": job.horizon, "seed": job.seed},
         "training": {"report_rule": "last", "final_val_acc": val_acc,
                      "final_val_loss": 1.0, "final_epoch": job.horizon},
         "test_metrics": {"accuracy": 0.5, "f1_macro": 0.5}}
    (job.save_dir / "centralized_results.json").write_text(json.dumps(r))


@pytest.fixture
def bn_plans(monkeypatch):
    """The two plans of the batch-norm study, imported for one dataset."""
    import importlib

    import rfl.experiments.plans.mi_spdnet_bn as main
    import rfl.experiments.plans.mi_spdnet_bn_calib as calib

    def load(dataset):
        monkeypatch.setenv("RFL_DATASET", dataset)
        importlib.reload(calib)
        importlib.reload(main)
    yield load
    monkeypatch.delenv("RFL_DATASET", raising=False)
    importlib.reload(calib)
    importlib.reload(main)


def _score(j):   # best: lr 0.1, batch 32, horizon 300, momentum 0.3, q 5, blocks 32,16
    hp = j.hp
    return (0.5 + 0.1 * (j.lr == 0.1) + 0.05 * (j.batch_size == 32)
            + 0.02 * (j.horizon == 300) + 0.01 * (hp.get("--bn-momentum") == 0.3)
            + 0.01 * (hp.get("--reeig-percentile") == 5) + 0.03 * ("a32_16" in j.arm))


def _run_stage(plan, stage, n):
    jobs = G.plan_jobs(plan, stage)
    assert len(jobs) == n, (plan, stage, len(jobs))
    for j in jobs:
        _fake_any(j, _score(j))
    assert G.main([plan, "--stage", stage, "--select"]) == 0
    return jobs


def _write_floors(dataset, archs):
    """A floors file as analyze_covariances --architectures writes it."""
    from rfl.experiments.analyze_covariances import PERCENTILES
    out = {}
    for a in archs:
        n = len(a.split(","))
        out[a] = {"nobn": {f"{q:g}": {"floors": [1e-4 * q * 10 ** k for k in range(n)]}
                           for q in PERCENTILES},
                  "bn": {f"{q:g}": {"floors": [1e-4 * q] + [0.5] * (n - 1)}
                         for q in PERCENTILES}}
    path = G.results_root() / "reeig_floors" / f"{dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": {"percentiles": list(PERCENTILES)},
                                "architectures": out}))


@pytest.mark.parametrize("dataset,n_calib,batches,deep", [
    ("BNCI2014_001", (72, 36), {32, 64, 128}, False),
    ("Schirrmeister2017", (288, 144), {32, 64, 128}, True),
    ("Cho2017", (72, 36), {16, 32, 64}, False)])
def test_batch_norm_study_calibrates_then_tunes_every_arm_alike(root, bn_plans, dataset,
                                                                 n_calib, batches, deep):
    """One protocol: floors, calibration on the centralised arms, then the same tuning
    (lr x batch, then horizon) for the 18 arms."""
    from rfl.experiments.plans.mi_spdnet_bn_calib import STUDY
    bn_plans(dataset)
    with pytest.raises(SystemExit, match="no ReEig floors"):
        G.plan_jobs("mi_spdnet_bn_calib", "sweep1")
    _write_floors(dataset, STUDY[dataset]["architectures"])
    with pytest.raises(SystemExit, match="not found"):
        G.plan_jobs("mi_spdnet_bn", "sweep1")             # calibrate first
    _run_stage("mi_spdnet_bn_calib", "sweep1", n_calib[0])
    _run_stage("mi_spdnet_bn_calib", "sweep2", n_calib[1])
    s1 = _run_stage("mi_spdnet_bn", "sweep1", 648)
    assert {j.batch_size for j in s1} == batches and {j.horizon for j in s1} == {200}
    s2 = _run_stage("mi_spdnet_bn", "sweep2", 162)
    assert {j.horizon for j in s2} == {100, 200, 300}
    final = G.plan_jobs("mi_spdnet_bn", "final")
    arms = {j.arm for j in final}
    assert len(final) == 180 and len(arms) == 18
    assert "fed_spdnet_local_bnshared" not in arms and "central_spdnet_iid_bn" not in arms
    for j in final:
        ns = _parse(j)
        bn = j.arm.endswith("_bn") or j.arm.endswith("bnshared")
        assert ns.use_batch_norm == bn and ns.learning_rate == 0.1 and ns.batch_size == 32
        assert ns.rounds == 300 if j.entry == "fed" else ns.epochs == 300
        assert ns.optimizer == "sgd" and ns.stiefel == "trivialization"
        assert ns.bn_momentum == (0.3 if bn else 0.1) and ns.reeig_percentile == 5
        assert ns.hidden_dims == ("32,16" if deep else STUDY[dataset]["architectures"][0])
        if j.entry == "fed":
            assert (ns.client_partition == "iid") == ("_iid_" in j.arm)
            assert ns.fedbn == (not j.arm.endswith("bnshared"))
            assert bool(ns.head_refit_epochs) == ("_shared_" in j.arm)


@pytest.fixture
def eegnet_bn_plans(monkeypatch):
    """The two plans of the EEGNet batch-norm study, imported for one dataset."""
    import importlib

    import rfl.experiments.plans.mi_eegnet_bn as main
    import rfl.experiments.plans.mi_eegnet_bn_calib as calib
    import rfl.experiments.plans.mi_spdnet_bn as spd
    import rfl.experiments.plans.mi_spdnet_bn_calib as spd_calib

    def load(dataset):
        monkeypatch.setenv("RFL_DATASET", dataset)
        for m in (spd_calib, spd, calib, main):
            importlib.reload(m)
    yield load
    monkeypatch.delenv("RFL_DATASET", raising=False)
    for m in (spd_calib, spd, calib, main):
        importlib.reload(m)


@pytest.mark.parametrize("dataset,batches", [("BNCI2014_001", {32, 64, 128}),
                                             ("Cho2017", {16, 32, 64})])
def test_eegnet_batch_norm_study_mirrors_the_spdnet_one(root, eegnet_bn_plans, dataset,
                                                        batches):
    """The 18 arms of mi_spdnet_bn with EEGNet: momentum calibrated centrally, then the
    same tuning, seeds and selection, FedAvg, no recalibration of the statistics."""
    import rfl.experiments.plans.mi_eegnet_bn as main
    import rfl.experiments.plans.mi_spdnet_bn as spd
    eegnet_bn_plans(dataset)
    with pytest.raises(SystemExit, match="not found"):
        G.plan_jobs("mi_eegnet_bn", "sweep1")             # calibrate first
    c1 = _run_stage("mi_eegnet_bn_calib", "sweep1", 36)
    assert {j.horizon for j in c1} == {200} and all(_parse(j).bn_momentum == 0.1 for j in c1)
    c2 = _run_stage("mi_eegnet_bn_calib", "sweep2", 9)
    assert {j.hp["--bn-momentum"] for j in c2} == {0.1, 0.01, 0.3}
    s1 = _run_stage("mi_eegnet_bn", "sweep1", 648)
    assert {j.batch_size for j in s1} == batches and {j.horizon for j in s1} == {200}
    s2 = _run_stage("mi_eegnet_bn", "sweep2", 162)
    assert {j.horizon for j in s2} == {100, 200, 300}
    final = G.plan_jobs("mi_eegnet_bn", "final")
    arms = {j.arm for j in final}
    assert len(final) == 180 and len(arms) == 18
    assert "fed_eegnet_local_bnshared" not in arms and "central_eegnet_iid_bn" not in arms
    assert {j.seed for j in final} == set(range(101, 111))
    assert main.SELECTION == spd.SELECTION and main.EXTEND == spd.EXTEND
    for j in final:
        ns = _parse(j)
        bn = j.arm.endswith("_bn") or j.arm.endswith("bnshared")
        assert ns.model == "eegnet" and ns.optimizer == "sgd"
        assert ns.eegnet_norm_type == ("batch" if bn else "none")
        assert ns.learning_rate == 0.1 and ns.batch_size == 32
        assert ns.rounds == 300 if j.entry == "fed" else ns.epochs == 300
        assert ns.bn_momentum == (0.3 if bn else 0.1) and not ns.bn_recalibrate
        if j.entry == "fed":
            assert ns.aggregation in (None, "fedavg")
            assert (ns.client_partition == "iid") == ("_iid_" in j.arm)
            assert ns.fedbn == (not j.arm.endswith("bnshared"))
            assert bool(ns.head_refit_epochs) == ("_shared_" in j.arm)


def test_mi_study_keeps_its_step_numbers():
    from rfl.experiments.pipeline import steps as plan_steps
    steps = [s if isinstance(s, tuple) else ("floors",) for s in plan_steps("mi_study")]
    assert steps[9] == ("mi_datasize_eegnet", "sweep1")
    assert steps[12] == ("mi_datasize_tslr", "final")
    assert steps[13:] == [("mi_eegnet_bn_calib", "sweep1"), ("mi_eegnet_bn_calib", "sweep2"),
                          ("mi_eegnet_bn", "sweep1"), ("mi_eegnet_bn", "sweep2"),
                          ("mi_eegnet_bn", "final")]


def test_eegnet_batch_norm_momentum():
    import torch

    from rfl.models.eegnet import EEGNet
    from rfl.utils.config import ExperimentConfig
    m = EEGNet(4, 22, 256, norm_type="batch", bn_momentum=0.3)
    assert {b.momentum for b in m.modules() if isinstance(b, torch.nn.BatchNorm2d)} == {0.3}
    assert EEGNet(4, 22, 256, norm_type="batch").batchnorm_temporal.momentum == 0.1
    with pytest.raises(ValueError, match="bn_momentum"):
        EEGNet(4, 22, 256, norm_type="batch", bn_momentum=0.0)
    cfg = ExperimentConfig(model_type="eegnet", eegnet_norm_type="batch",
                           spdnet_bn_momentum=0.01)
    assert cfg.get_model_params()["bn_momentum"] == 0.01
    assert "bn_momentum" not in ExperimentConfig(model_type="eegnet").get_model_params()


def test_variants_are_checked(root):
    from rfl.experiments.protocol import Protocol, Variant
    with pytest.raises(ValueError, match="applies to no arm"):
        Protocol("Cho2017", kinds=("central",), variants={"x": Variant(kinds=("fed_local",))})
    with pytest.raises(ValueError, match="folder name"):
        Protocol("Cho2017", variants={"a b": Variant()})
    with pytest.raises(ValueError, match="unknown kinds"):
        Protocol("Cho2017", variants={"x": Variant(kinds=("fed",))})
    P = Protocol("Cho2017", models=("spdnet",), kinds=("fed_local", "fed_shared"),
                 aggregations={"spdnet": ("projavg",)},
                 variants={"": Variant(), "bn": Variant(args=("--use-batch-norm",),
                                                       space={"--bn-momentum": [0.1, 0.3]})})
    names = [a for a, *_ in P.arms()]
    assert names == ["fed_spdnet_projavg_shared", "fed_spdnet_projavg_shared_bn",
                     "fed_spdnet_local", "fed_spdnet_local_bn"]
    assert P.space_of("fed_spdnet_local_bn", "spdnet", "nodp")["--bn-momentum"] == [0.1, 0.3]
    assert "--bn-momentum" not in P.space_of("fed_spdnet_local", "spdnet", "nodp")


def test_log_filter_drops_printed_tensors_only():
    import io

    import torch

    from rfl.utils.log_filter import TensorPrintFilter, main
    buf = io.StringIO()
    print("[Server] Round 1 (lr 0.1)", file=buf)
    print(torch.randn(64, 16, 16, dtype=torch.float64), file=buf)
    print(torch.randn(16, 16, dtype=torch.float64), file=buf)
    print("  val_acc=0.50 (best)", file=buf)
    print(torch.tensor(0.5), file=buf)
    print("END", file=buf)
    out = io.StringIO()
    main(io.StringIO(buf.getvalue()), out)
    lines = out.getvalue().splitlines()
    assert lines[:3] == ["[Server] Round 1 (lr 0.1)", "  val_acc=0.50 (best)", "END"]
    assert "removed 3 printed tensors" in lines[3] and len(lines) == 4
    assert list(TensorPrintFilter().filter(["a\n", "b\n"])) == ["a\n", "b\n"]   # no summary


def test_run_one_keeps_the_batch_norm_prints_out_of_run_log(tmp_path):
    import sys

    from rfl.data.synthetic import generate_synthetic_spd, save_dataset
    data = tmp_path / "s.npz"
    save_dataset(generate_synthetic_spd(n_clients=2, samples_per_client=40, n_channels=6,
                                        n_classes=2, mode="iid", seed=0), str(data))
    env = {**os.environ, "RFL_RESULTS": str(tmp_path), "RFL_VENV": str(tmp_path / "none"),
           "MNE_DATA": str(tmp_path / "mne_data"), "PYTHONPATH": str(REPO),
           "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]}
    d = tmp_path / "job"
    r = subprocess.run(["bash", str(REPO / "condor/run_one.sh"), str(d), "fed",
                        "--synthetic-clients", str(data), "--rounds", "1",
                        "--hidden-dim", "4", "--use-batch-norm", "--batch-size", "16"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    log = (d / "run.log").read_text()
    assert "tensor(" not in log and (d / "fl_results.json").exists()
    assert "[log_filter] removed" in log
    assert (d / "exit_code").read_text().strip() == "0"


def test_staged_search_skips_an_arm_with_nothing_to_sweep(root):
    """The arm without a stage-2 key keeps its stage-1 choice: no rerun."""
    from rfl.experiments.protocol import Protocol, Variant
    P = Protocol("Cho2017", models=("spdnet",), kinds=("central",),
                 aggregations={"spdnet": ("projavg",)},
                 variants={"nobn": Variant(),
                           "bn": Variant(args=("--use-batch-norm",),
                                         space={"--bn-momentum": [0.1, 0.3]})},
                 space={"nodp": {"lr": [0.01, 0.1]}, "dp": {}}, search="staged",
                 stage_keys=[["lr"], ["--bn-momentum"]], sweep_seeds=(1,),
                 final_seeds=(9,))
    for k, stage in enumerate(("sweep1", "sweep2")):
        jobs = P.sweep(k)
        for j in jobs:
            j.stage, j.plan = stage, P.name
            _fake_any(j, 0.5 + 0.1 * (j.lr == 0.1) + 0.01 * (j.hp.get("--bn-momentum") == 0.3))
        if stage == "sweep2":
            assert {j.arm for j in jobs} == {"central_spdnet_bn"}
        sel = G.select(jobs)
        G.selection_path(P.name, stage).parent.mkdir(parents=True, exist_ok=True)
        G.selection_path(P.name, stage).write_text(json.dumps(sel))
    final = {j.arm: j.hp for j in P.final()}
    assert final["central_spdnet_nobn"] == {"lr": 0.1}
    assert final["central_spdnet_bn"] == {"lr": 0.1, "--bn-momentum": 0.3}
    assert P.selection("central_spdnet_nobn")["mean"] == pytest.approx(0.6)


@pytest.fixture
def datasize_plans(bn_plans):
    """The data-quantity plans, imported for one dataset (after the batch-norm plans)."""
    import importlib

    import rfl.experiments.plans.mi_datasize as spd
    import rfl.experiments.plans.mi_datasize_eegnet as eeg
    import rfl.experiments.plans.mi_datasize_tslr as ts

    def load(dataset):
        bn_plans(dataset)
        for m in (spd, eeg, ts):
            importlib.reload(m)
    yield load
    for m in (spd, eeg, ts):
        importlib.reload(m)


def test_data_quantity_plans_tune_every_arm_on_the_reduced_clients(root, datasize_plans):
    """Every (head, normalisation) pair per scenario, its own tuning, the batch and
    horizon grids following N, selection on the reduced clients."""
    from rfl.experiments.plans.mi_spdnet_bn_calib import STUDY
    datasize_plans("BNCI2014_001")
    _write_floors("BNCI2014_001", STUDY["BNCI2014_001"]["architectures"])
    _run_stage("mi_spdnet_bn_calib", "sweep1", 72)
    _run_stage("mi_spdnet_bn_calib", "sweep2", 36)

    def fake(j, v):                       # per-client accuracies: reduced_val_acc = v
        j.save_dir.mkdir(parents=True, exist_ok=True)
        r = {"config": {"dataset": "D", "model_type": "spdnet", "seed": j.seed,
                        "learning_rate": j.lr, "batch_size": j.batch_size,
                        "fl_rounds": j.horizon},
             "fl_info": {"reduced_clients": [0]},
             "final_metrics": {"global_val_acc": 0.0, "global_test_acc": 0.5,
                               "per_client_val_acc": [v, 0.0], "per_client_test_acc":
                               [0.5, 0.5]}, "last_round_metrics": {}, "round_history": {}}
        (j.save_dir / "fl_results.json").write_text(json.dumps(r))

    s1 = G.plan_jobs("mi_datasize", "sweep1", check=True)
    arms = {j.arm for j in s1}
    assert len(arms) == 64 and len(s1) == 64 * 12 * 3
    assert "fed_spdnet_projavg_fedper_bnshared_H5" in arms
    assert "fed_spdnet_local_bnshared_U5" not in arms
    assert {j.batch_size for j in s1 if j.arm.endswith("_U5")} == {4, 8, 16}
    assert {j.batch_size for j in s1 if j.arm.endswith("_H5")} == {32, 64, 128}
    for j in s1:
        fake(j, 0.5 + 0.1 * (j.lr == 0.1) + 0.05 * (j.batch_size in (8, 64)))
    assert G.main(["mi_datasize", "--stage", "sweep1", "--select"]) == 0
    s2 = G.plan_jobs("mi_datasize", "sweep2")
    assert {j.horizon for j in s2 if j.arm.endswith("_U5")} == {566, 283, 849}
    assert {j.horizon for j in s2 if j.arm.endswith("_U20")} == {283, 141, 424}
    assert {j.horizon for j in s2 if j.arm.endswith("_full")} == {200, 100, 300}
    assert {(j.lr, j.batch_size) for j in s2 if j.arm.endswith("_U5")} == {(0.1, 8)}
    for j in s2:
        fake(j, 0.6)
    assert G.main(["mi_datasize", "--stage", "sweep2", "--select"]) == 0
    final = G.plan_jobs("mi_datasize", "final")
    assert len(final) == 640
    for j in final:
        ns = _parse(j)
        bn = "_bn_" in j.arm or "_bnshared_" in j.arm
        assert ns.use_batch_norm == bn and ns.bn_recalibrate == bn
        assert ns.optimizer == "sgd" and ns.reeig_percentile == 5
        assert (ns.train_trials_per_class is None) == j.arm.endswith("_full")
        assert ns.reduced_subjects == (3 if "_H" in j.arm else 0)
        assert bool(ns.head_refit_epochs) == ("_shared_" in j.arm)
    eeg = G.plan_jobs("mi_datasize_eegnet", "sweep1", check=True)
    assert len({j.arm for j in eeg}) == 64 and {j.lr for j in eeg} == {1e-2, 3e-2, 1e-1, 3e-1}
    assert all("--optimizer" in j.args and "sgd" in j.args for j in eeg)
    ts = G.plan_jobs("mi_datasize_tslr", "final", check=True)
    assert len(ts) == 8 * 2 * 10 and {j.entry for j in ts} == {"tslr"}


def test_log_filter_drops_a_tensor_printed_after_a_progress_bar():
    from rfl.utils.log_filter import TensorPrintFilter
    lines = ["\rEpoch 1 [Train]:   0%|  | 0/12 [00:00<?, ?it/s]tensor([[[ 3.0e-02,\n",
             "          1.0e-02]]], dtype=torch.float64)\n",
             "tensor([[-0.15,  0.10]])\n",
             "loss=tensor(0.5) kept\n",
             "Val - Loss: 1.2\n"]
    out = list(TensorPrintFilter().filter(lines))
    assert out[:3] == ["\rEpoch 1 [Train]:   0%|  | 0/12 [00:00<?, ?it/s]\n",
                       "loss= kept\n", "Val - Loss: 1.2\n"]
    assert "removed 3 printed tensors" in out[3] and len(out) == 4
