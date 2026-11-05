"""The automatic pipeline (rfl/experiments/pipeline.py): selection rules, grid
extension, and the HTCondor DAG, executed by a small DAGMan simulator."""
import json
import os
import shlex
import subprocess
import sys
import types
from pathlib import Path

import pytest

from rfl.experiments import grid as G
from rfl.experiments import pipeline as PL

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from test_grid import _fake_any  # noqa: E402


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("RFL_RESULTS", str(tmp_path / "results"))
    return tmp_path


def test_next_value_follows_the_rules():
    lr = {"factor": 3, "min": 1e-4, "max": 1.0}
    assert G.next_value(0.3, [0.01, 0.1, 0.3], lr) == 0.9
    assert G.next_value(0.01, [0.01, 0.1, 0.3], lr) == pytest.approx(0.00333)
    assert G.next_value(0.9, [0.1, 0.3, 0.9], lr) is None          # 2.7 > max
    assert G.next_value(0.1, [0.01, 0.1, 0.3], lr) is None         # not an end
    b = {"factor": 2, "int": True, "min": 8, "max": 64}
    assert G.next_value(64, [16, 32, 64], b) is None and G.next_value(16, [16, 32, 64], b) == 8
    h = {"factor": 1.5, "int": True, "max": 1000, "down": False}
    assert G.next_value(300, [100, 200, 300], h) == 450
    assert G.next_value(100, [100, 200, 300], h) is None
    q = {"values": [1, 2, 5, 10, 15, 20, 30]}
    assert G.next_value(30, [5, 15, 30], q) is None and G.next_value(5, [5, 15, 30], q) == 2


def _jobs_with(root, scores):
    """Jobs of one arm over horizons x seeds, with the given validation accuracies."""
    jobs = []
    for (T, seed), v in scores.items():
        j = G.Job("arm", "fed", seed, {"lr": 0.1, "horizon": T})
        j.stage, j.plan = "s", "p"
        _fake_any(j, v)
        jobs.append(j)
    return jobs


def test_one_standard_error_rule_takes_the_shortest_converged_horizon(root):
    # 200 and 300 are within noise of each other, 100 is clearly worse.
    sc = {(100, 1): .60, (100, 2): .62, (100, 3): .61,
          (200, 1): .71, (200, 2): .75, (200, 3): .73,
          (300, 1): .71, (300, 2): .75, (300, 3): .74}
    jobs = _jobs_with(root, sc)
    assert G.select(jobs)["arm"]["hp"]["horizon"] == 300                 # best mean
    s = G.select(jobs, prefer_smaller=["horizon"])["arm"]
    # 200 is chosen, and 300 (the best mean, at the top) asks for one longer horizon.
    assert s["hp"]["horizon"] == 200 and s["on_grid_edge"] == ["horizon"]
    sc_conv = {**sc, (400, 1): .70, (400, 2): .74, (400, 3): .72}     # no longer helps
    s = G.select(_jobs_with(root, sc_conv), prefer_smaller=["horizon"])["arm"]
    assert s["hp"]["horizon"] == 200 and s["on_grid_edge"] == []
    # Still improving at the longest horizon: that is an edge, to extend.
    sc2 = {**sc, (300, 1): .80, (300, 2): .81, (300, 3): .82}
    s2 = G.select(_jobs_with(root, sc2), prefer_smaller=["horizon"])["arm"]
    assert s2["hp"]["horizon"] == 300 and s2["on_grid_edge"] == ["horizon"]


@pytest.fixture
def auto_plan(monkeypatch):
    """A one-arm plan whose best learning rate (0.9) lies beyond its grid."""
    from rfl.experiments.protocol import Protocol
    P = Protocol("Cho2017", models=("spdnet",), kinds=("central",),
                 aggregations={"spdnet": ("projavg",)},
                 space={"nodp": {"lr": [0.01, 0.1, 0.3]}, "dp": {}},
                 sweep_seeds=(1, 2), final_seeds=(9,), name="auto")
    mod = types.ModuleType("rfl.experiments.plans.auto_test")
    mod.NAME, mod.STAGES = P.name, P.stages()
    mod.EXTEND = {"lr": {"factor": 3, "min": 1e-3, "max": 3.0}}
    monkeypatch.setitem(sys.modules, "rfl.experiments.plans.auto_test", mod)
    return "auto_test"


def test_advance_extends_an_edge_until_the_choice_is_inside(root, auto_plan):
    score = {0.01: .5, 0.1: .6, 0.3: .7, 0.9: .8, 2.7: .75}
    passes = 0
    while True:
        todo = PL.advance(auto_plan, "sweep", verbose=False)
        if not todo:
            break
        passes += 1
        for j in todo:
            _fake_any(j, score[j.lr])
    assert passes == 3                              # grid, + 0.9, + 2.7
    sel = json.loads(G.selection_path("auto", "sweep").read_text())
    assert sel["central_spdnet"]["hp"]["lr"] == 0.9
    ext = json.loads(G.extensions_path("auto", "sweep").read_text())
    assert ext == {"central_spdnet": {"lr": [0.9, 2.7]}}
    assert {j.lr for j in G.plan_jobs(auto_plan, "final")} == {0.9}
    assert PL.advance(auto_plan, "final", verbose=False)              # not run yet


# --------------------------------------------------------------------------- #
#  The DAG, run by a minimal DAGMan: PRE, job, POST, PRE_SKIP, RETRY, in order
# --------------------------------------------------------------------------- #
def _env_of(dag_dir):
    env = dict(os.environ)
    for line in (dag_dir / "dag_env.sh").read_text().splitlines():
        k, v = line[len("export "):].split("=", 1)
        env[k] = v
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]
    env["PYTHONPATH"] = str(REPO)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def _run_node_job(sub: Path, env):
    text = sub.read_text()
    macros = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith(("#", "+", "queue")):
            k, v = line.split("=", 1)
            macros[k.strip()] = v.strip()
    exe = macros["executable"].replace("$(repo)", macros["repo"])
    for line in text.splitlines():
        if line.startswith("queue job from "):
            for job in Path(line[len("queue job from "):]).read_text().splitlines():
                subprocess.run(["bash", exe, *job.split()], env=env, cwd=macros["workdir"],
                               capture_output=True)
        elif line == "queue 1":
            args = shlex.split(macros["arguments"].strip('"'))
            subprocess.run(["bash", exe, *args], env=env, cwd=macros["workdir"],
                           capture_output=True)


def simulate_dag(dag: Path):
    nodes, order = {}, []
    for line in dag.read_text().splitlines():
        t = line.split()
        if not t or t[0].startswith("#"):
            continue
        if t[0] == "JOB":
            nodes[t[1]] = {"sub": Path(t[2])}
            order.append(t[1])
        elif t[0] == "SCRIPT":
            nodes[t[2]][t[1].lower()] = t[3:]
        elif t[0] == "PRE_SKIP":
            nodes[t[1]]["skip"] = int(t[2])
        elif t[0] == "RETRY":
            nodes[t[1]]["retry"] = int(t[2])
    env = _env_of(dag.parent)
    log = []
    for name in order:                               # a chain: PARENT a CHILD b, in order
        n = nodes[name]
        for _ in range(n["retry"] + 1):
            rc = subprocess.run(["bash", *n["pre"]], env=env, capture_output=True,
                                text=True)
            if rc.returncode == n["skip"]:
                log.append((name, "skipped"))
                break
            assert rc.returncode == 0, rc.stdout + rc.stderr
            _run_node_job(n["sub"], env)
            post = subprocess.run(["bash", *n["post"]], env=env, capture_output=True,
                                  text=True)
            log.append((name, "post", post.returncode))
            if post.returncode == 0:
                break
        else:
            raise AssertionError(f"node {name} failed after its retries")
    return log


@pytest.mark.slow
def test_the_dag_runs_a_whole_study(root, monkeypatch):
    """smoke_auto as HTCondor would run it: data, two sweep stages, final runs. A second
    submission skips everything."""
    monkeypatch.chdir(root)
    monkeypatch.setenv("RFL_VENV", str(root / "none"))
    monkeypatch.setenv("MNE_DATA", str(root / "mne"))
    import importlib

    import rfl.experiments.plans.smoke_auto as sa
    importlib.reload(sa)                                # DATA under this RFL_RESULTS
    dag = PL.write_dag("smoke_auto", cpus=1, extra=["+isShortJob = True"],
                       directory=root / "dag")
    text = dag.read_text()
    assert text.count("JOB ") == 4 and "PRE_SKIP N1_smoke_auto_sweep1 99" in text
    assert "PARENT N2_smoke_auto_sweep2 CHILD N3_smoke_auto_final" in text
    assert "+isShortJob = True" in (root / "dag" / "N1_smoke_auto_sweep1.sub").read_text()
    log = simulate_dag(dag)
    assert ("N0_data", "post", 0) in log
    assert all(j.state() == "done" for j in G.plan_jobs("smoke_auto", "final"))
    for st in ("sweep1", "sweep2"):
        assert G.selection_path("smoke_auto", st).exists()
    assert {tuple(e) for _, *e in simulate_dag(dag)} == {("skipped",)}   # all done


@pytest.mark.parametrize("runner", ["fed", "central"])
def test_a_diverged_run_writes_its_result(tmp_path, monkeypatch, runner):
    import torch

    from rfl.analysis.results import read_run
    from rfl.experiments import run_centralized_experiment as C
    from rfl.experiments import run_federated_experiment as F
    mod, fn, name = ((F, "run_federated_experiment", "fl_results.json") if runner == "fed"
                     else (C, "run_centralized_experiment", "centralized_results.json"))

    def boom(*a, **k):
        raise torch.linalg.LinAlgError("linalg.eigh: failed to converge")
    monkeypatch.setattr(mod, fn, boom)
    mod.main(["--dataset", "BNCI2014_001", "--save-dir", str(tmp_path)])
    row = read_run(str(tmp_path / name))
    assert row["diverged"] and row["val_acc"] != row["val_acc"]          # NaN


def test_check_finite_stops_a_nan_loss():
    from rfl.training.divergence import TrainingDiverged, check_finite
    check_finite(0.5, "round 1")
    with pytest.raises(TrainingDiverged):
        check_finite(float("nan"), "round 2")


def test_failed_runs_are_abandoned_after_three_attempts(root, auto_plan):
    jobs = G.plan_jobs(auto_plan, "sweep")
    for j in jobs:
        if j.lr == 0.3 and j.seed == 1:                     # fails every time
            j.save_dir.mkdir(parents=True, exist_ok=True)
            (j.save_dir / "exit_code").write_text("1\n")
            (j.save_dir / "attempts").write_text("2\n")
        else:
            _fake_any(j, {0.01: .5, 0.1: .7, 0.3: .9}[j.lr])
    bad = [j for j in jobs if j.lr == 0.3 and j.seed == 1][0]
    assert bad.state() == "failed" and PL.advance(auto_plan, "sweep", verbose=False) == [bad]
    (bad.save_dir / "attempts").write_text("3\n")
    assert bad.state() == "abandoned"
    assert PL.advance(auto_plan, "sweep", verbose=False) == []       # selected without it
    sel = json.loads(G.selection_path("auto", "sweep").read_text())
    assert sel["central_spdnet"]["hp"]["lr"] == 0.1                   # the 0.3 cell is out


def test_a_diverged_cell_is_left_out_of_the_selection(root, auto_plan):
    for j in G.plan_jobs(auto_plan, "sweep"):
        _fake_any(j, float("nan") if j.lr == 0.3 else {0.01: .5, 0.1: .7}[j.lr])
    assert PL.advance(auto_plan, "sweep", verbose=False) == []
    s = json.loads(G.selection_path("auto", "sweep").read_text())["central_spdnet"]
    assert s["hp"]["lr"] == 0.1 and s["on_grid_edge"] == []


def test_the_horizon_extends_while_its_best_mean_is_the_longest(root):
    """Noisy seeds: the 1-SE rule picks 200, but 300 is still the best mean, so the
    grid grows from 300."""
    sc = {(100, 1): .50, (100, 2): .52,
          (200, 1): .66, (200, 2): .70,
          (300, 1): .66, (300, 2): .78}
    jobs = _jobs_with(root, sc)
    s = G.select(jobs, prefer_smaller=["horizon"])["arm"]
    assert s["hp"]["horizon"] == 200 and s["on_grid_edge"] == ["horizon"]
    added = G.extend_grids("p", "s", jobs, {"arm": s},
                           {"horizon": {"factor": 1.5, "int": True, "down": False}})
    assert added == {"arm": {"horizon": [450]}}
