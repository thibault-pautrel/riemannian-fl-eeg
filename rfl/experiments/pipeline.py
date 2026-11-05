"""Run a study end to end: its stages in order, each selection, and the grid extensions.

A plan runs stage after stage: submit the jobs of a stage, wait, select, submit the next
stage. This module does it without manual steps, on the cluster (an HTCondor DAG) or
here. One step of a stage (`advance`):

1. the jobs of the stage that are not done (pending, failed, interrupted) run again;
2. when every job is done, the stage is selected (`grid.write_selection`, with the
   plan's SELECTION settings);
3. a selected value on the edge of its grid, for a key with a rule in the plan's EXTEND,
   gets one more value beyond the edge (`grid.extend_grids`), and the new cells run;
4. the stage is finished when nothing is left to run: no failed job, no edge left with
   a rule (an edge at the bound of its rule stays, and the selection marks it).

The steps of a study come from the plan's PIPELINE (default: its STAGES in order). A
step is (plan, stage), or a command (for example the ReEig floors) with a test that
tells when it is done:

    PIPELINE = [Command("floors", [...], done=lambda: ...),
                ("mi_spdnet_bn_calib", "sweep1"), ..., ("mi_spdnet_bn", "final")]

A PIPELINE can also be a list of chains (lists of steps): the chains then run side by
side in the DAG, and `run` does them one after the other (plans/mi_study.py).

Usage (from the folder where the plan's relative paths make sense):

    python -m rfl.experiments.pipeline run mi_spdnet_bn --jobs 4   # here, until the end
    bash condor/submit_dag.sh mi_spdnet_bn --memory 12GB           # HTCondor DAG
    python -m rfl.experiments.pipeline status mi_spdnet_bn

The DAG (written by `write_dag`, submitted by condor/submit_dag.sh) has one node per
step. A node's PRE script writes the jobs still to run (it skips the node when there
are none), its job runs them, its POST script calls `advance` again and asks for a
retry while something is left to run. RETRY bounds the passes of a node.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

from rfl.experiments.grid import (Job, extend_grids, plan_attr, plan_jobs, plan_namespace,
                                  require_plan, results_root, run_job, write_selection)

SKIP = 99          # PRE script exit code that skips a DAG node (PRE_SKIP)


@dataclass
class Command:
    """A step that runs one command (no selection), done when `done()` is True."""
    name: str
    argv: Sequence[str]
    done: Callable[[], bool] = field(default=lambda: False)


Step = Union[Tuple[str, str], Command]


def _is_command(step) -> bool:
    # Duck typing: run as `python -m rfl.experiments.pipeline`, this module is __main__,
    # and the plans' Command is the class of rfl.experiments.pipeline.
    return hasattr(step, "argv") and hasattr(step, "done")


def branches(plan: str) -> List[List[Step]]:
    """The chains of steps of a study. A PIPELINE is one chain, or a list of chains
    (lists) that run side by side in the DAG, each step after the previous one of its
    chain. Without PIPELINE: the plan's stages in order."""
    pipe = plan_attr(plan, "PIPELINE", None)
    if pipe is None:
        from rfl.experiments.grid import load_plan
        return [[(plan, stage) for stage in load_plan(plan).STAGES]]
    chains = pipe if pipe and isinstance(pipe[0], list) else [pipe]
    return [[s if _is_command(s) else (s[0], s[1]) for s in c] for c in chains]


def steps(plan: str) -> List[Step]:
    """Every step of a study, chain after chain (`branches`)."""
    return [s for c in branches(plan) for s in c]


def step_name(i: int, step: Step) -> str:
    return f"N{i}_" + (step.name if _is_command(step) else f"{step[0]}_{step[1]}")


def advance(plan: str, stage: str, metric: str = "val_acc", check: bool = False,
            verbose: bool = True) -> List[Job]:
    """One step of a stage (module docstring). Returns the jobs left to run (empty when
    the stage is finished)."""
    jobs = plan_jobs(plan, stage, check=check)
    if not jobs:                    # a stage without its own key to sweep: nothing to do
        return []
    # Nothing of the plan runs while a step advances (DAG PRE/POST, `run`): a started
    # job without exit code was killed (Job.state, dead=True).
    todo = [j for j in jobs if j.state(dead=True) not in ("done", "abandoned")]
    dropped = [j for j in jobs if j.state(dead=True) == "abandoned"]
    if dropped and verbose and not todo:
        what = ("these seeds are missing from the final results" if stage == "final"
                else "their cells are left out")
        print(f"warning: {len(dropped)} job(s) abandoned after repeated failures: {what} "
              "(grid --status lists them)")
    if todo or stage == "final":
        return todo
    sel = write_selection(plan, jobs, metric, verbose=verbose)
    rules = plan_attr(plan, "EXTEND", {})
    added = extend_grids(plan_namespace(plan), stage, jobs, sel, rules) if rules else {}
    if not added:
        return []
    if verbose:
        for arm, keys in sorted(added.items()):
            print(f"extend {arm}: " + ", ".join(f"{k} + {v}" for k, v in keys.items()))
    jobs = plan_jobs(plan, stage, check=check)
    return [j for j in jobs if j.state(dead=True) not in ("done", "abandoned")]


def run_here(plan: str, n_jobs: int = 1, max_passes: int = 20,
             metric: str = "val_acc") -> None:
    """Run every step of a study on this machine, as the DAG does on the cluster."""
    threads = max(1, (os.cpu_count() or 1) // max(1, n_jobs))
    for i, st in enumerate(steps(plan)):
        name = step_name(i, st)
        if _is_command(st):
            if st.done():
                print(f"[{name}] done")
                continue
            print(f"[{name}] {' '.join(st.argv)}")
            if subprocess.call(list(st.argv)) != 0 or not st.done():
                raise SystemExit(f"[{name}] failed")
            continue
        for p in range(1, max_passes + 1):
            todo = advance(st[0], st[1], metric, check=(p == 1))
            if not todo:
                print(f"[{name}] finished")
                break
            print(f"[{name}] pass {p}: {len(todo)} jobs")
            with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as ex:
                list(ex.map(lambda j: run_job(j, threads=threads), todo))
        else:
            raise SystemExit(f"[{name}] still unfinished after {max_passes} passes "
                             "(failed jobs: python -m rfl.experiments.grid "
                             f"{st[0]} --stage {st[1]} --status)")


# --------------------------------------------------------------------------- #
#  HTCondor DAG
# --------------------------------------------------------------------------- #
def dag_pre(plan: str, index: int, jobs_file: str) -> int:
    """PRE script of node `index`: write the jobs left to run, or skip the node."""
    st = steps(plan)[index]
    if _is_command(st):
        return SKIP if st.done() else 0
    todo = advance(st[0], st[1], check=True)
    if not todo:
        return SKIP
    Path(jobs_file).write_text("".join(j.line() + "\n" for j in todo))
    for j in todo:                      # as condor/submit.sh: submitted, not started
        j.save_dir.mkdir(parents=True, exist_ok=True)
        (j.save_dir / "queued").touch()
    print(f"{len(todo)} jobs -> {jobs_file}")
    return 0


def dag_post(plan: str, index: int) -> int:
    """POST script of node `index`: 0 when the step is finished, else 1 (retry)."""
    st = steps(plan)[index]
    if _is_command(st):
        return 0 if st.done() else 1
    todo = advance(st[0], st[1])
    if todo:
        states = {}
        for j in todo:
            states[j.state(dead=True)] = states.get(j.state(dead=True), 0) + 1
        print(f"{len(todo)} jobs left ({states}): retry")
        return 1
    return 0


_RFL_SUB = Path(__file__).resolve().parents[2] / "condor" / "rfl.sub"


def write_dag(plan: str, cpus: int = 4, memory: str = "8GB", disk: str = "1GB",
              extra: Sequence[str] = (), retries: int = 20,
              directory: Optional[Path] = None) -> Path:
    """Write the DAG of a study and its node files. Returns the .dag path.

    `extra`: submit lines added to every node (GPU, duration class, accounting group).
    The environment of this process (RFL_*, MNE_DATA) is recorded in dag_env.sh, so the
    PRE and POST scripts and the jobs see the same folders and dataset.
    """
    repo = Path(os.environ.get("RFL_REPO", Path(__file__).resolve().parents[2]))
    ns = plan_namespace(plan)
    d = directory or (results_root() / "condor" /
                      f"dag_{ns}_{time.strftime('%Y%m%d_%H%M%S')}")
    (d / "logs").mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if k.startswith("RFL_") or k == "MNE_DATA"}
    env.setdefault("RFL_REPO", str(repo))
    env.setdefault("RFL_RESULTS", str(results_root()))
    env.setdefault("RFL_WORKDIR", os.getcwd())
    for k, v in env.items():
        if any(c.isspace() for c in v):
            raise SystemExit(f"{k}={v!r}: paths may not contain spaces")
    (d / "dag_env.sh").write_text("".join(f"export {k}={v}\n" for k, v in sorted(env.items())))
    macros = {"repo": str(repo), "logdir": str(d / "logs"), "cpus": str(cpus),
              "memory": memory, "disk": disk, "venv": env.get("RFL_VENV", str(repo / ".venv")),
              "results": env["RFL_RESULTS"], "mnedata": env.get("MNE_DATA", ""),
              "tmpdir": env.get("RFL_TMPDIR", env["RFL_RESULTS"] + "/.tmp"),
              "workdir": env["RFL_WORKDIR"]}
    head = "".join(f"{k} = {v}\n" for k, v in macros.items())
    body = _RFL_SUB.read_text()
    # In a DAG a job always exits 0 (its status is in exit_code, read by the POST
    # script): DAGMan would otherwise remove the other jobs of the node at the first
    # failure. A job held for more than an hour is removed, so that its node ends and
    # its POST script counts it (Job.state, dead=True).
    dag_body = (body.replace("RFL_WORKDIR=$(workdir)\"", "RFL_WORKDIR=$(workdir) RFL_DAG=1\"")
                + "periodic_remove = (JobStatus == 5) && "
                  "((time() - EnteredCurrentStatus) > 3600)\n")
    if "RFL_DAG=1" not in dag_body:
        raise SystemExit(f"{_RFL_SUB}: environment line not found")
    step_sh = repo / "condor" / "dag_step.sh"
    lines = [f"# DAG of {plan} ({ns}), written by rfl.experiments.pipeline",
             f"# Submit: condor_submit_dag {d / 'study.dag'}"]
    names = []
    for i, st in enumerate(steps(plan)):
        name = step_name(i, st)
        names.append(name)
        sub = d / f"{name}.sub"
        if _is_command(st):
            args = " ".join(st.argv)
            sub.write_text(head + body.replace("executable            = $(repo)/condor/run_one.sh",
                                               "executable            = $(repo)/condor/run_cmd.sh")
                           .replace("arguments             = $(job)",
                                    f'arguments             = "{args}"')
                           + "".join(f"{e}\n" for e in extra) + "queue 1\n")
        else:
            sub.write_text(head + dag_body + "".join(f"{e}\n" for e in extra)
                           + f"queue job from {d / (name + '.jobs')}\n")
        lines += [f"JOB {name} {sub}",
                  f"SCRIPT PRE {name} {step_sh} {d / 'dag_env.sh'} pre {plan} {i} "
                  f"{d / (name + '.jobs')}",
                  f"SCRIPT POST {name} {step_sh} {d / 'dag_env.sh'} post {plan} {i}",
                  f"PRE_SKIP {name} {SKIP}",
                  f"RETRY {name} {retries}"]
    start = 0
    for chain in branches(plan):           # each chain in order, the chains side by side
        own = names[start:start + len(chain)]
        lines += [f"PARENT {a} CHILD {b}" for a, b in zip(own, own[1:])]
        start += len(chain)
    dag = d / "study.dag"
    dag.write_text("\n".join(lines) + "\n")
    return dag


def status(plan: str) -> None:
    """One line per step: finished, or what is left."""
    print(f"{plan} -> {plan_namespace(plan)}  (RFL_DATASET="
          f"{os.environ.get('RFL_DATASET', 'unset')}, RFL_RESULTS={results_root()})")
    for i, st in enumerate(steps(plan)):
        name = step_name(i, st)
        if _is_command(st):
            print(f"{name:45s} {'done' if st.done() else 'to run'}")
            continue
        try:
            jobs = plan_jobs(st[0], st[1])
        except SystemExit as e:                      # an earlier step is not finished
            print(f"{name:45s} waiting: {str(e).splitlines()[0]}")
            continue
        counts = {}
        for j in jobs:
            counts[j.state()] = counts.get(j.state(), 0) + 1
        print(f"{name:45s} {len(jobs)} jobs  "
              + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run every step here")
    r.add_argument("plan")
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--max-passes", type=int, default=20)
    s = sub.add_parser("status", help="state of every step")
    s.add_argument("plan")
    w = sub.add_parser("dag", help="write the HTCondor DAG (condor/submit_dag.sh)")
    w.add_argument("plan")
    w.add_argument("--cpus", type=int, default=4)
    w.add_argument("--memory", default="8GB")
    w.add_argument("--disk", default="1GB")
    w.add_argument("--extra", action="append", default=[],
                   help="a submit line added to every node (repeatable)")
    w.add_argument("--retries", type=int, default=20)
    for name in ("pre", "post"):
        q = sub.add_parser(name, help=f"DAG {name.upper()} script of one node")
        q.add_argument("plan")
        q.add_argument("index", type=int)
        if name == "pre":
            q.add_argument("jobs_file")
    a = p.parse_args(argv)
    require_plan(a.plan)
    if a.cmd == "run":
        run_here(a.plan, a.jobs, a.max_passes)
    elif a.cmd == "status":
        status(a.plan)
    elif a.cmd == "dag":
        print(write_dag(a.plan, a.cpus, a.memory, a.disk, a.extra, a.retries))
    elif a.cmd == "pre":
        return dag_pre(a.plan, a.index, a.jobs_file)
    elif a.cmd == "post":
        return dag_post(a.plan, a.index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
