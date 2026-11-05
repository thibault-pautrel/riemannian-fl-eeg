"""Smoke test of the automatic pipeline: a whole study as one HTCondor DAG, in minutes.

A tiny synthetic federation, two arms (centralised and federated), a staged search
(learning rate, then horizon) with automatic grid extension, then the final runs. It
checks on the cluster that DAGMan, the PRE and POST scripts and the jobs work together:

    bash condor/submit_dag.sh smoke_auto --short
    python -m rfl.experiments.pipeline status smoke_auto

or here: python -m rfl.experiments.pipeline run smoke_auto --jobs 4. The accuracies
mean nothing.
"""

from rfl.experiments.grid import results_root
from rfl.experiments.pipeline import Command
from rfl.experiments.protocol import Protocol

DATA = results_root() / "smoke_auto" / "synth.npz"

P = Protocol(
    dataset="synthetic",
    data_args=["--synthetic-clients", str(DATA)],
    common_args=(),
    models=("spdnet",),
    aggregations={"spdnet": ("projavg",)},
    kinds=("central", "fed_shared"),
    model_args={"spdnet": ("--hidden-dim", "4", "--optimizer", "sgd")},
    space={"nodp": {"lr": [0.1, 0.03, 0.3], "batch_size": [16], "horizon": [2, 1, 3]},
           "dp": {}},
    search="staged",
    stage_keys=[["lr"], ["horizon"]],
    sweep_seeds=(1,),
    final_seeds=(9,),
    refit_args=(),
    name="smoke_auto",
)
NAME = P.name
STAGES = P.stages()
SELECTION = {"prefer_smaller": ["horizon"]}
EXTEND = {"lr": {"factor": 3, "min": 1e-3, "max": 1.0},
          "horizon": {"factor": 2, "int": True, "max": 6, "down": False}}
PIPELINE = [
    Command("data", ["python", "-m", "rfl.data.synthetic",
                     "--n-clients", "3", "--samples-per-client", "60", "--n-channels", "8",
                     "--n-classes", "2", "--seed", "0", "--output", str(DATA)],
            done=DATA.exists),
    ("smoke_auto", "sweep1"), ("smoke_auto", "sweep2"), ("smoke_auto", "final"),
]
