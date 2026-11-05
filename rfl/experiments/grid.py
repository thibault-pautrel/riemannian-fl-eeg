"""Experiment plans: list, check, run, monitor and select a grid of runs.

A plan is a Python module in `rfl/experiments/plans/` that defines the runs of one study,
organised in stages (for example `sweep`, then `final`). Each run is a `Job`: an arm
(a named setting, such as `fed_spdnet_fedper`), a seed, its swept hyperparameters and
the command-line arguments of a runner. Any runner option can be swept: learning rate,
batch size and horizon have short names (`lr`, `batch_size`, `horizon`), the others are
written as on the command line (`--dp-clip-C`, `--hidden-dim`, `--weight-decay`). A job
writes into its own directory, so a finished job is recognised by its result file and
never run twice.

The same plan serves every way of running it:

    python -m rfl.experiments.grid smoke --check             # build every configuration
    python -m rfl.experiments.grid smoke --list              # pending jobs, one per line
    python -m rfl.experiments.grid smoke --run --jobs 2      # run them here
    python -m rfl.experiments.grid smoke --status            # done / failed / pending
    python -m rfl.experiments.grid mi_protocol --stage sweep --select
                                                          # best hyperparameters per arm
    bash condor/submit.sh mi_protocol sweep              # one HTCondor job per run

`--list` prints `<save_dir> <entry> <runner arguments>`, the format condor/run_one.sh
reads. `--list`, `--run` and `--check` first build the configuration of every distinct
command line, so an invalid option fails before any job starts. Results go under
`$RFL_RESULTS/<NAME>/<stage>/`, where NAME is the plan's `NAME` attribute (default: the
module name) and RFL_RESULTS defaults to `./results`.

Selection (`--select`) implements the protocol of docs/cluster.md: for each arm, the swept
values with the best mean over the seeds of the stage of the validation metric at the
reported (last) round. It refuses to run while jobs of the stage are unfinished. It
writes `$RFL_RESULTS/<NAME>/<stage>/selection.json`, which a later stage reads with
`selected` or `load_selection`. A numeric choice on the edge of its grid is flagged:
extend the grid then.

Job states (`--status`), from the files that condor/run_one.sh writes in the job
folder: pending, queued (submitted by condor/submit.sh, not started), running? (started,
no exit code: running, or killed), failed, done, conflict (the folder holds a run of
another command: the plan changed, give it a new NAME).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import itertools
import json
import math
import random
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]          # the repository root
ENTRIES = {"fed": "rfl.experiments.run_federated_experiment",
           "central": "rfl.experiments.run_centralized_experiment",
           "tslr": "rfl.experiments.run_tslr"}
RESULT_FILES = {"fed": "fl_results.json", "central": "centralized_results.json",
                "tslr": "fl_results.json"}

# Hyperparameter names with a short form. Any other name is a runner option, written as
# on the command line ("--dp-clip-C", "--hidden-dim", "--weight-decay", ...).
SHORT = {"lr": "--learning-rate", "batch_size": "--batch-size", "horizon": None}
LABELS = {"lr": "lr", "batch_size": "B", "horizon": "T"}
FIXED = ("--seed", "--save-dir")            # set by the plan machinery, never swept


# Failed attempts after which a run is left out ('abandoned', Job.state).
MAX_ATTEMPTS = int(os.environ.get("RFL_MAX_ATTEMPTS", "3"))


def results_root() -> Path:
    """Root of all plan results: $RFL_RESULTS, or ./results."""
    return Path(os.environ.get("RFL_RESULTS", "results")).resolve()


def _fmt(v: Any) -> str:
    """Exact text of a value: the shortest repr of a float, without a trailing '.0'."""
    if isinstance(v, float):
        s = repr(v)
        return s[:-2] if s.endswith(".0") else s
    return str(v)


def _flag(name: str, entry: str) -> str:
    """Command-line option of hyperparameter `name` for runner `entry`."""
    if name == "horizon":
        return "--rounds" if entry == "fed" else "--epochs"
    return SHORT.get(name) or name


def _order(hp: Dict[str, Any]) -> List[str]:
    """lr, batch_size, horizon first, then the other names in alphabetical order."""
    return [k for k in SHORT if k in hp] + sorted(k for k in hp if k not in SHORT)


@dataclass
class Job:
    """One run of a plan.

    Args:
        arm: name of the setting (one selection is made per arm).
        entry: 'fed' or 'central'.
        seed: the run seed (model initialisation, training randomness, data split).
        hp: the swept hyperparameters, name -> value. Names are 'lr', 'batch_size',
            'horizon' (rounds or epochs) or any runner option ('--dp-clip-C': 1.0). A
            value True adds a flag without value, False leaves it out.
        args: the other runner arguments, without the swept ones, --seed and
            --save-dir.
        stage, plan: set by `plan_jobs`.
    """
    arm: str
    entry: str
    seed: int
    hp: Dict[str, Any]
    args: Sequence[str] = ()
    stage: str = ""
    plan: str = ""

    # Read access in the old form, for plans written before `hp`.
    @property
    def lr(self):
        return self.hp.get("lr")

    @property
    def batch_size(self):
        return self.hp.get("batch_size")

    @property
    def horizon(self):
        return self.hp.get("horizon")

    @property
    def hp_name(self) -> str:
        """Directory name of the hyperparameters, e.g. lr0.01_B64_T100_dp-clip-C=1."""
        parts = []
        for k in _order(self.hp):
            v = self.hp[k]
            if k in LABELS:
                parts.append(f"{LABELS[k]}{_fmt(v)}")
            else:
                parts.append(f"{k.lstrip('-')}={_fmt(v)}")
        return "_".join(parts) or "default"

    @property
    def save_dir(self) -> Path:
        return results_root() / self.plan / self.stage / self.arm / self.hp_name / \
            f"seed{self.seed}"

    def runner_args(self) -> List[str]:
        out = [*map(str, self.args), "--seed", str(self.seed)]
        for k in _order(self.hp):
            v, flag = self.hp[k], _flag(k, self.entry)
            if v is True:
                out.append(flag)
            elif v is not False:
                out += [flag, _fmt(v)]
        return out

    def line(self) -> str:
        """`<save_dir> <entry> <args...>`, the input line of condor/run_one.sh.

        HTCondor splits the line on whitespace, so no field may contain one.
        """
        fields = [str(self.save_dir), self.entry, *self.runner_args()]
        bad = [f for f in fields if not f or any(c.isspace() for c in f)]
        if bad:
            raise ValueError(f"job fields may not be empty or contain spaces: {bad}")
        return " ".join(fields)

    def command(self) -> str:
        """`<entry> <runner arguments>`, as recorded in <save_dir>/job_args."""
        return " ".join([self.entry, *self.runner_args()])

    def state(self, dead: bool = False) -> str:
        """State of the job, from the files of its folder (condor/run_one.sh).

        `dead`: no job of the plan is running now (the PRE and POST scripts of a DAG
        node, `pipeline run`). A started job without exit code is then a killed one,
        'failed' instead of 'running?'. After 2 x RFL_MAX_ATTEMPTS starts without an
        exit code a job is 'abandoned' in both cases.

        'conflict'  the folder holds a run of another command (job_args differs)
        'done'      result file present and exit code 0 (or no exit code yet written)
        'failed'    exit code written and not 0, or no result file
        'running?'  started (run.log), no exit code: running, or killed (evicted,
                    over its memory, removed)
        'queued'    submitted to HTCondor (marker written by submit.sh), not started
        'abandoned' failed RFL_MAX_ATTEMPTS times (default 3): not run again, and its
                    cell is left out of the selection. A diverged run is not a failure:
                    it writes its result (rfl/training/divergence.py).
        'pending'   nothing yet
        """
        d = self.save_dir
        rec = d / "job_args"
        if rec.exists() and rec.read_text().strip() != self.command():
            return "conflict"
        code = d / "exit_code"
        rc = code.read_text().strip() if code.exists() else None
        if (d / RESULT_FILES[self.entry]).exists() and rc in (None, "0"):
            return "done"
        # A job submitted again after a failure keeps its old exit_code and run.log
        # until it starts: the newer 'queued' marker takes precedence.
        queued = d / "queued"
        if queued.exists():
            older = [f for f in (code, d / "run.log") if f.exists()]
            if all(f.stat().st_mtime < queued.stat().st_mtime for f in older):
                return "queued"
        att = d / "attempts"
        n = int(att.read_text().strip() or 0) if att.exists() else 0
        if rc is not None:
            return "abandoned" if n >= MAX_ATTEMPTS else "failed"
        if (d / "run.log").exists():
            # Started and never finished. After 2 x RFL_MAX_ATTEMPTS starts the job is
            # given up (a kill may hit a sound job, hence twice the failure count).
            if n >= 2 * MAX_ATTEMPTS:
                return "abandoned"
            # When nothing runs now, it was killed (memory rule, eviction, removal).
            return "failed" if dead else "running?"
        return "pending"


_DEST_MAPS: Dict[str, Dict[str, str]] = {}


def _dest_map(entry: str) -> Dict[str, str]:
    """Option string -> argparse destination, for the runner of `entry` (cached)."""
    if entry not in _DEST_MAPS:
        parser = importlib.import_module(ENTRIES[entry])._build_arg_parser()
        _DEST_MAPS[entry] = {o: a.dest for a in parser._actions for o in a.option_strings}
    return _DEST_MAPS[entry]


def grid(arm: str, entry: str, args: Sequence[str], seeds: Iterable[int],
         lrs: Optional[Iterable[float]] = None,
         batch_sizes: Optional[Iterable[int]] = None,
         horizons: Optional[Iterable[int]] = None,
         space: Optional[Dict[str, Iterable[Any]]] = None,
         sample: Optional[int] = None, sample_seed: int = 0) -> List[Job]:
    """Every combination of the swept values, for each seed, for one arm.

    With `sample`, a random subset of `sample` combinations (random search, drawn with
    `sample_seed`, the same for every seed of the arm) replaces the full grid.

    `lrs`, `batch_sizes` and `horizons` sweep the learning rate, batch size and number
    of rounds (or epochs). `space` sweeps any other runner option, for example
    {"--dp-clip-C": [0.5, 1, 2], "--hidden-dim": [8, 16]}. A value left out (None) is
    not swept: the runner default, or the value in `args`, is used.

    Raises:
        ValueError: if a swept option also appears in `args`, or is --seed/--save-dir.
    """
    axes: Dict[str, List[Any]] = {}
    for name, values in (("lr", lrs), ("batch_size", batch_sizes), ("horizon", horizons)):
        if values is not None:
            cast = float if name == "lr" else int
            axes[name] = [cast(v) for v in values]
    for name, values in (space or {}).items():
        if not name.startswith("--") and name not in SHORT:
            raise ValueError(f"swept option {name!r}: write it as on the command line, "
                             "for example '--dp-clip-C'")
        axes[name] = list(values)
    # Compare argparse destinations, not strings: an option may be written '--x=v' or
    # under an alias, and 'lr' is '--learning-rate'.
    dest = _dest_map(entry)

    def dest_of(token: str):
        flag = token.split("=", 1)[0]
        return dest.get(flag, flag)

    fixed_dests = {dest_of(t) for t in map(str, args) if t.startswith("--")}
    seen: Dict[str, str] = {}
    for name in axes:
        d = dest_of(_flag(name, entry))
        if d in seen:
            raise ValueError(f"arm {arm!r}: {name} and {seen[d]} set the same option")
        seen[d] = name
        if d in fixed_dests:
            raise ValueError(f"arm {arm!r}: {_flag(name, entry)} is swept, so it may not "
                             "also be fixed in the arm arguments (the plan's args, or "
                             "common_args/model_args of a Protocol)")
    fixed = [f for f in FIXED if dest_of(f) in fixed_dests | set(seen)]
    if fixed:
        raise ValueError(f"arm {arm!r}: {', '.join(fixed)} is set by the plan machinery "
                         "(seeds and folders), not in the arm arguments")
    combos = list(itertools.product(*axes.values())) if axes else [()]
    if sample is not None:
        if sample < 1:
            raise ValueError(f"sample must be at least 1, got {sample}")
        if sample < len(combos):
            keep = sorted(random.Random(sample_seed).sample(range(len(combos)), sample))
            combos = [combos[i] for i in keep]
    return [Job(arm, entry, int(s), dict(zip(axes, c)), tuple(map(str, args)))
            for c in combos for s in seeds]


def selected(arm: str, entry: str, args: Sequence[str], seeds: Iterable[int],
             plan: str, from_arm: Optional[str] = None,
             stage: str = "sweep") -> List[Job]:
    """The jobs of `arm` with the values selected for it (or for `from_arm`) in `stage`."""
    sel = load_selection(plan, from_arm or arm, stage)
    return [Job(arm, entry, int(s), dict(sel["hp"]), tuple(map(str, args)))
            for s in seeds]


# --------------------------------------------------------------------------- #
#  Plans
# --------------------------------------------------------------------------- #
def load_plan(name: str):
    """Import rfl/experiments/plans/<name>.py."""
    return importlib.import_module(f"rfl.experiments.plans.{name}")


def require_plan(name: str) -> None:
    """Stop with the list of plans when rfl/experiments/plans/<name>.py does not exist
    (the command-line entry points call it first)."""
    here = Path(__file__).resolve().parent / "plans"
    if f"rfl.experiments.plans.{name}" not in sys.modules and not (here / f"{name}.py").exists():
        known = sorted(p.stem for p in here.glob("*.py") if not p.stem.startswith("_"))
        raise SystemExit(f"no plan {name!r}. Plans: {', '.join(known)}")


def plan_attr(name: str, attr: str, default: Any) -> Any:
    """An optional setting of a plan module (SELECTION, EXTEND, PIPELINE)."""
    try:
        return getattr(load_plan(name), attr, default)
    except ModuleNotFoundError:
        return default


def plan_namespace(name: str) -> str:
    """Results folder of a plan: its NAME attribute, or the module name."""
    return getattr(load_plan(name), "NAME", name)


def plan_jobs(name: str, stage: Optional[str] = None, check: bool = False,
              allow_conflicts: bool = False) -> List[Job]:
    """The jobs of one stage of a plan (its first stage by default).

    With `check`, every distinct command line is built by its runner without running
    it (`check_command`), so a typo or an invalid combination of options fails here
    and not on the cluster.
    """
    plan = load_plan(name)
    stages = plan.STAGES
    stage = stage or next(iter(stages))
    if stage not in stages:
        raise SystemExit(f"plan {name!r} has no stage {stage!r}; stages: {list(stages)}")
    if hasattr(plan, "prepare"):
        plan.prepare()
    jobs = stages[stage]()
    space = getattr(plan, "NAME", name)
    for j in jobs:
        j.stage, j.plan = stage, space
    dirs = [j.save_dir for j in jobs]
    if len(set(dirs)) != len(dirs):
        raise SystemExit(f"plan {name!r}, stage {stage!r}: two jobs share a directory")
    conflicts = [j for j in jobs if j.state() == "conflict"]
    if conflicts and not allow_conflicts:
        raise SystemExit(
            f"{len(conflicts)} job folder(s) hold runs of another command, for example\n"
            f"  {conflicts[0].save_dir}\n  recorded: "
            f"{(conflicts[0].save_dir / 'job_args').read_text().strip()}\n  now:      "
            f"{conflicts[0].command()}\nThe plan changed after these runs. Give it a "
            "new NAME (results folder), or delete these folders.")
    if check:
        check_jobs(jobs)
    return jobs


def check_jobs(jobs: List[Job]) -> None:
    """Build the configuration of each distinct command line (one seed per arm).

    Raises:
        SystemExit: naming the first invalid job and the runner's message.
    """
    import warnings
    runners = {}
    seen = set()
    notes: Dict[str, None] = {}
    for j in jobs:
        key = (j.entry, tuple(j.args), tuple(sorted(j.hp.items(), key=str)))
        if key in seen:
            continue
        seen.add(key)
        if j.entry not in runners:
            runners[j.entry] = importlib.import_module(ENTRIES[j.entry])
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()), \
                    warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                runners[j.entry].check_command(j.runner_args() + ["--save-dir", "x"])
            for w in caught:
                notes.setdefault(f"{j.arm}: {w.message}", None)
        except SystemExit:
            text = err.getvalue()
            msg = text.split("error:", 1)[1].strip() if "error:" in text else text.strip()
            raise SystemExit(f"invalid job {j.arm} ({j.hp_name}): {msg}") from None
        except ValueError as e:
            raise SystemExit(f"invalid job {j.arm} ({j.hp_name}): {e}") from None
    if notes:
        print(f"{len(notes)} warning(s):\n  " + "\n  ".join(notes), file=sys.stderr)


# --------------------------------------------------------------------------- #
#  Running
# --------------------------------------------------------------------------- #
def run_job(job: Job, python: str = sys.executable, threads: int = 1) -> int:
    """Run one job here, as condor/run_one.sh does on a worker, with `threads` threads."""
    d = job.save_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "job_args").write_text(job.command() + "\n")
    for stale in (RESULT_FILES[job.entry], "exit_code", "queued"):
        (d / stale).unlink(missing_ok=True)
    att = d / "attempts"                        # as condor/run_one.sh
    att.write_text(f"{(int(att.read_text().strip() or 0) if att.exists() else 0) + 1}\n")
    cmd = [python, "-m", ENTRIES[job.entry], *job.runner_args(), "--save-dir", str(d)]
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
           "PYTHONUNBUFFERED": "1"}
    from rfl.utils.log_filter import TensorPrintFilter
    with open(d / "run.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=os.getcwd(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env, text=True)
        for line in TensorPrintFilter().filter(proc.stdout):   # as condor/run_one.sh
            log.write(line)
        rc = proc.wait()
    (d / "exit_code").write_text(f"{rc}\n")
    return rc


# --------------------------------------------------------------------------- #
#  Selection
# --------------------------------------------------------------------------- #
def _val_metric(job: Job, metric: str) -> Optional[float]:
    from rfl.analysis.results import read_run
    path = job.save_dir / RESULT_FILES[job.entry]
    if not path.exists():
        return None
    row = read_run(str(path))
    v = None if row is None else row.get(metric)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def select(jobs: List[Job], metric: str = "val_acc",
           prefer_smaller: Sequence[str] = ()) -> Dict[str, Dict[str, Any]]:
    """Best hyperparameters per arm: highest mean of `metric` over the seeds.

    Only hyperparameter cells where every seed finished are compared. Lower is better
    for metrics whose name contains 'loss'. A selected value at the end of its grid
    (numbers only, grids of 3 values or more) is listed in `on_grid_edge`.

    `prefer_smaller` (for example ['horizon']): among the cells within one standard
    error of the best mean, and equal to the best on the other keys, take the one with
    the smallest values of these keys (the one-standard-error rule, on the per-seed
    differences: a seed fixes the split and the initialisation). For the horizon,
    this is the shortest training after which longer training no longer helps: the run
    has converged. Such a key is on the edge when its best mean (not the chosen value)
    lies at the top of its grid: training longer still helps, so the grid extends from
    that top value (`edge_values`).
    """
    minimize = "loss" in metric
    by_arm: Dict[str, Dict[tuple, Dict[int, Optional[float]]]] = {}
    for j in jobs:
        cell = by_arm.setdefault(j.arm, {}).setdefault(tuple(j.hp.items()), {})
        cell[j.seed] = _val_metric(j, metric)
    out: Dict[str, Dict[str, Any]] = {}
    for arm, cells in by_arm.items():
        complete = {hp: list(s.values()) for hp, s in cells.items()
                    if all(v is not None for v in s.values())}
        if not complete:
            out[arm] = {"error": "no hyperparameter cell has all its seeds finished"}
            continue
        mean = {hp: sum(v) / len(v) for hp, v in complete.items()}

        def sd(hp):
            v = complete[hp]
            return (math.sqrt(sum((x - mean[hp]) ** 2 for x in v) / (len(v) - 1))
                    if len(v) > 1 else 0.0)
        best = (min if minimize else max)(mean, key=mean.get)
        top = dict(best)                      # the best mean, before the 1-SE rule
        rule = "best mean"
        pref = [k for k in prefer_smaller if k in dict(best)]
        if pref:
            others = {k: v for k, v in best if k not in pref}

            def close(hp) -> bool:
                """Within one standard error of the best. The seeds pair the cells (same
                split, same initialisation), so the error is that of the per-seed
                difference when two seeds or more are shared."""
                common = sorted(set(cells[hp]) & set(cells[best]))
                if len(common) >= 2:
                    d = [(cells[best][s] - cells[hp][s]) * (-1 if minimize else 1)
                         for s in common]
                    md = sum(d) / len(d)
                    sdd = math.sqrt(sum((x - md) ** 2 for x in d) / (len(d) - 1))
                    return md <= sdd / math.sqrt(len(d))
                se = sd(best) / math.sqrt(len(complete[best]))
                return (mean[hp] <= mean[best] + se if minimize
                        else mean[hp] >= mean[best] - se)
            near = [hp for hp in complete
                    if all(dict(hp).get(k) == v for k, v in others.items()) and close(hp)]
            best = min(near, key=lambda hp: (tuple(dict(hp)[k] for k in pref),
                                             mean[hp] if minimize else -mean[hp]))
            rule = (f"smallest {', '.join(pref)} within one standard error of the best "
                    "(paired over the seeds)")
        vals = complete[best]
        hp = dict(best)
        edges = []
        for k in hp:
            values = {dict(c)[k] for c in cells}
            # With two values every choice is an edge: flag only grids of 3 or more.
            if len(values) > 2 and all(isinstance(v, (int, float)) and
                                       not isinstance(v, bool) for v in values):
                values = sorted(values)
                # The smallest value of a `prefer_smaller` key is the intended choice
                # (for the horizon: converged early), not an edge.
                if k in prefer_smaller:
                    if top[k] == values[-1]:
                        edges.append(k)
                elif hp[k] in (values[0], values[-1]):
                    edges.append(k)
        out[arm] = {"hp": hp, **{k: hp[k] for k in SHORT if k in hp},
                    "edge_values": {k: top[k] for k in prefer_smaller if k in edges},
                    "metric": metric, "mean": mean[best], "std": sd(best),
                    "n_seeds": len(vals), "n_cells": len(complete),
                    "n_cells_total": len(cells), "on_grid_edge": edges, "rule": rule}
    return out


# --------------------------------------------------------------------------- #
#  Automatic extension of the grids
# --------------------------------------------------------------------------- #
def next_value(value: float, grid: Sequence[float], rule: Dict[str, Any]) -> Optional[float]:
    """The value that extends `grid` beyond `value`, an end of it, or None.

    `rule` is {"factor": f, "min": lo, "max": hi, "int": bool, "up": bool, "down": bool}
    (multiply or divide by f, within [lo, hi]; up and down default to True) or
    {"values": [...]} (the next listed value beyond the end).
    """
    lo_end, hi_end = min(grid), max(grid)
    up = value == hi_end and rule.get("up", True)
    down = value == lo_end and rule.get("down", True)
    if not (up or down):
        return None
    if "values" in rule:
        listed = sorted(rule["values"])
        beyond = ([v for v in listed if v > hi_end] if up else
                  [v for v in listed if v < lo_end][::-1])
        return beyond[0] if beyond else None
    f = float(rule["factor"])
    new = value * f if up else value / f
    if rule.get("int"):
        new = int(round(new))
    else:
        new = float(f"{new:.3g}")
    if new < rule.get("min", -math.inf) or new > rule.get("max", math.inf) or new in grid:
        return None
    return new


def extensions_path(plan: str, stage: str) -> Path:
    """Values added to the grid of each arm at one stage: <plan>/<stage>/extensions.json."""
    return results_root() / plan / stage / "extensions.json"


def load_extensions(plan: str, stage: str) -> Dict[str, Dict[str, List[Any]]]:
    path = extensions_path(plan, stage)
    return json.loads(path.read_text()) if path.exists() else {}


def extend_grids(plan: str, stage: str, jobs: List[Job], sel: Dict[str, Dict[str, Any]],
                 rules: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, List[Any]]]:
    """Add one value beyond each selected value on the edge of its grid, for the keys
    with a rule. `rules` is a dict, or a function of the arm giving one (rules that
    depend on the arm, such as a batch size bounded by the client size). Writes
    extensions.json and returns what it added ({arm: {key: [v]}})."""
    grids: Dict[str, Dict[str, set]] = {}
    for j in jobs:
        for k, v in j.hp.items():
            grids.setdefault(j.arm, {}).setdefault(k, set()).add(v)
    ext = load_extensions(plan, stage)
    added: Dict[str, Dict[str, List[Any]]] = {}
    at_bound: Dict[str, List[str]] = {}
    for arm, s in sel.items():
        if arm not in grids or "error" in s:
            continue
        arm_rules = rules(arm) if callable(rules) else rules
        for k in s.get("on_grid_edge", []):
            if k not in arm_rules:
                continue
            value = s.get("edge_values", {}).get(k, s["hp"][k])
            new = next_value(value, sorted(grids[arm][k]), arm_rules[k])
            if new is None:
                at_bound.setdefault(arm, []).append(k)
            if new is not None:
                ext.setdefault(arm, {}).setdefault(k, [])
                if new not in ext[arm][k]:
                    ext[arm][k].append(new)
                    added.setdefault(arm, {})[k] = [new]
    if added:
        path = extensions_path(plan, stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ext, indent=2))
    if at_bound:            # edges that no rule can extend: the final stage says nothing
        path = selection_path(plan, stage)
        if path.exists():
            full = json.loads(path.read_text())
            for arm, keys in at_bound.items():
                if arm in full:
                    full[arm]["at_bound"] = keys
            path.write_text(json.dumps(full, indent=2))
    return added


def write_selection(plan_name: str, jobs: List[Job], metric: str = "val_acc",
                    verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """Select (with the plan's SELECTION settings) and write selection.json.

    SELECTION may hold 'prefer_smaller' (see `select`) and 'metric', a validation column
    of rfl/analysis/results.py that replaces `metric` (for example 'reduced_val_acc',
    the clients with little data in a data-quantity study).

    The selection is merged with the arms selected before in this stage: a plan whose
    arms depend on an environment variable (RFL_EPS, for example) keeps every one.
    """
    settings = plan_attr(plan_name, "SELECTION", {})
    metric = settings.get("metric", metric)
    if not (metric.startswith("val_") or "_val_" in metric):
        raise SystemExit(f"selection metric {metric!r} (SELECTION of {plan_name}): the "
                         "selection uses validation metrics only (val_acc, val_loss, "
                         "reduced_val_acc, ...)")
    sel = select(jobs, metric, prefer_smaller=settings.get("prefer_smaller", ()))
    path = selection_path(plan_namespace(plan_name), jobs[0].stage if jobs else "sweep")
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = {**json.loads(path.read_text()), **sel} if path.exists() else sel
    path.write_text(json.dumps(merged, indent=2))
    if verbose:
        for arm, s in sel.items():
            if "error" in s:
                print(f"{arm:28s} {s['error']}")
                continue
            edge = ("  EDGE: " + ",".join(LABELS.get(k, k.lstrip('-'))
                                          for k in s["on_grid_edge"])
                    if s["on_grid_edge"] else "")
            hp = " ".join(f"{LABELS.get(k, k.lstrip('-'))}={_fmt(v)}"
                          for k, v in s["hp"].items())
            print(f"{arm:28s} {hp}  "
                  f"{s['metric']}={s['mean']:.4f} +/- {s['std']:.4f} "
                  f"({s['n_seeds']} seeds, {s['n_cells']}/{s['n_cells_total']} cells)"
                  f"{edge}")
        print(f"written: {path}")
    return sel


def selection_path(plan: str, stage: str = "sweep") -> Path:
    """Selection file of one stage: $RFL_RESULTS/<plan>/<stage>/selection.json."""
    return results_root() / plan / stage / "selection.json"


def load_selection(plan: str, arm: str, stage: str = "sweep") -> Dict[str, Any]:
    """Selection of one arm in one stage, written by `--select` ("hp": the values).

    Raises:
        SystemExit: if the selection file or the arm is missing.
    """
    path = selection_path(plan, stage)
    if not path.exists():
        raise SystemExit(f"{path} not found: finish the stage {stage!r} of {plan}, then "
                         "select it (`python -m rfl.experiments.grid <plan module> --stage "
                         f"{stage} --select`, or let `pipeline run` / the DAG do it).")
    sel = json.loads(path.read_text())
    if arm not in sel or "hp" not in sel[arm]:
        raise SystemExit(f"{path}: no selection for arm {arm!r}")
    return sel[arm]


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan", help="module name in rfl/experiments/plans/")
    p.add_argument("--stage", default=None, help="stage of the plan (default: first)")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true",
                      help="print the jobs that are not done, one per line")
    mode.add_argument("--list-all", action="store_true", help="print every job")
    mode.add_argument("--status", action="store_true", help="count jobs by state")
    mode.add_argument("--check", action="store_true",
                      help="build the configuration of every job without running it")
    mode.add_argument("--run", action="store_true", help="run the pending jobs here")
    mode.add_argument("--select", action="store_true",
                      help="select the best hyperparameters per arm (see module doc)")
    p.add_argument("--jobs", type=int, default=1, help="parallel jobs with --run")
    p.add_argument("--retry-failed", action="store_true",
                   help="with --list or --run, also take the failed, interrupted "
                        "('running?') and queued jobs. Use it only when no job of the "
                        "plan is left in condor_q.")
    p.add_argument("--metric", default="val_acc",
                   help="selection metric, a validation metric (val_acc, val_loss, ...)")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="with --select, select although some jobs are not finished")
    a = p.parse_args(argv)
    require_plan(a.plan)
    if a.select and not a.metric.startswith("val_") and "_val_" not in a.metric:
        p.error(f"--metric {a.metric}: the selection uses validation metrics only "
                "(selecting on the test set biases the reported results)")

    jobs = plan_jobs(a.plan, a.stage, check=a.list or a.run or a.check,
                     allow_conflicts=a.status)
    todo_states = ({"pending", "failed", "running?", "queued"} if a.retry_failed
                   else {"pending"})

    if a.check:
        print(f"{len(jobs)} jobs: every command line is valid")
    elif a.list_all:
        print("\n".join(j.line() for j in jobs))
    elif a.list:
        lines = [j.line() for j in jobs if j.state() in todo_states]
        if lines:
            try:
                print("\n".join(lines))
            except BrokenPipeError:          # `--list | head`
                sys.stdout = None
                return 0
    elif a.status:
        counts: Dict[str, int] = {}
        for j in jobs:
            counts[j.state()] = counts.get(j.state(), 0) + 1
        print(f"{a.plan}/{jobs[0].stage if jobs else '-'}: {len(jobs)} jobs  "
              + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        for j in jobs:
            if j.state() == "failed":
                print(f"  failed: {j.save_dir}/run.log")
    elif a.run:
        todo = [j for j in jobs if j.state() in todo_states]
        print(f"running {len(todo)} jobs with {a.jobs} in parallel")
        with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            threads = max(1, (os.cpu_count() or 1) // max(1, a.jobs))
            rcs = list(ex.map(lambda j: run_job(j, threads=threads), todo))
        bad = sum(rc != 0 for rc in rcs)
        print(f"done: {len(rcs) - bad} ok, {bad} failed")
        return 1 if bad else 0
    elif a.select:
        open_jobs = [j for j in jobs if j.state() in ("pending", "queued", "running?")]
        if open_jobs and not a.allow_incomplete:
            raise SystemExit(f"{len(open_jobs)} job(s) of this stage are not finished "
                             f"(--status). Wait for them, or pass --allow-incomplete.")
        failed = [j for j in jobs if j.state() == "failed"]
        if failed:
            print(f"warning: {len(failed)} failed job(s); their cells are left out "
                  f"(--status lists them)", file=sys.stderr)
        done = [j for j in jobs if j.state() == "done"]
        if done:
            from rfl.analysis.results import read_run
            row = read_run(str(done[0].save_dir / RESULT_FILES[done[0].entry])) or {}
            if a.metric not in row:
                known = sorted(k for k in row if k.startswith("val_") or "_val_" in k)
                raise SystemExit(f"--metric {a.metric}: not in the results. Validation "
                                 f"metrics available: {', '.join(known)}")
        write_selection(a.plan, jobs, a.metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
