"""Step 2 of the aggregation study: how do the Stiefel aggregation rules interact with
the federated setting?

Model: the two-block SPDNet of mi_agg_calib, with a personal head (FedPer) and the SPD
batch norms kept on each client (FedBN). The server then averages the two BiMap
weights W1, W2 and nothing else. Factors, on one dataset (RFL_DATASET):

    rule           projavg | rlavg | retavg | pseavg        (rfl/federated/aggregation.py)
    clients        one subject per client (subj) | IID clients (iid: the same trials
                   pooled and dealt at random to the same number of clients)
    participation  10 | 25 | 50 | 80 | 100 % of the clients per round
    local epochs   E = 1 | 10 | 100

Full factorial: 4 x 2 x 5 x 3 = 120 arms, named fed_spdnet_<rule>_fedper_<clients>_p<%>_e<E>,
for example fed_spdnet_retavg_fedper_subj_p25_e10. No centralised, EEGNet or TS+LR arm.

Tuning. Every arm gets its own learning rate x batch size (12 cells x 3 seeds), the
grids extended at their edges (EXTEND), then 10 final seeds on the same splits for every
arm. Runs: 120 x 36 = 4320 sweep runs and 1200 final runs per dataset, plus the
extensions.

Convergence. Every run trains until its pooled validation loss stops improving: the
learning rate halves after P rounds without a lower validation loss, training stops after
S rounds, within a budget of R rounds, and the round of the lowest validation loss is
reported. The reference rule (E = 1, full participation) is P = 20, S = 150, R = 3000, the
rule of the batch-norm study. A round of the arm (participation p, E local epochs)
processes p E passes over the training data, against 1 for the reference, and each
client takes part in a share p of the rounds. So, with m = ceil(p K) of the K clients
sampled per round and pe = m / K:

    P = max(ceil(20 / (pe E)),   ceil(5 / pe))
    S = max(ceil(150 / (pe E)),  ceil(30 / pe))
    R = max(ceil(3000 / (pe E)), ceil(300 / pe))

The first term keeps the amount of local training of the reference rule, the second
asks every client to take part 5 (30, 300) times on average, so that a large E still
leaves enough aggregations to see a plateau. `selection.hit_budget` in fl_results.json
flags a run that ran out of budget before it stopped.

A RetAvg or PseAvg client too far from W_t (outside the domain of the inverse map, more
likely with E = 100) ends its run as diverged. Its cell is then left out of the
selection, and the divergence counts are part of the results
(rfl/analysis/aggregation.py).

A subset of the factors runs with environment variables (comma-separated), for
example E = 1 first. The arm names do not depend on the subset, so the results of several
subsets add up in the same folder:

    export RFL_DATASET=Cho2017 RFL_AGG_EPOCHS=1
    bash condor/submit_dag.sh mi_agg --memory 8GB
    python -m rfl.experiments.pipeline status mi_agg
    python -m rfl.analysis.aggregation $RFL_RESULTS/mi_Cho2017_agg/final --out agg_Cho2017

Variables: RFL_AGG_RULES, RFL_AGG_CLIENTS (subj,iid), RFL_AGG_PARTICIPATION (percent),
RFL_AGG_EPOCHS. Set them before submitting: the DAG records them.
"""

import math
import os

from rfl.data.registry import get_subjects
from rfl.experiments.pipeline import Command
from rfl.experiments.plans.mi_agg_calib import (
    DATASET, EXTEND, FINAL_SEEDS, LR, PLATEAU, SETTINGS, SWEEP_SEEDS, TRAINING, calibrated,
    floors_command, floors_ready)
from rfl.experiments.protocol import Protocol, Variant


def _levels(var: str, default: tuple, cast=str) -> tuple:
    raw = os.environ.get(var, "").strip()
    return tuple(cast(v) for v in raw.split(",") if v.strip()) if raw else default


RULES = _levels("RFL_AGG_RULES", ("projavg", "rlavg", "retavg", "pseavg"))
CLIENTS = _levels("RFL_AGG_CLIENTS", ("subj", "iid"))
PARTICIPATION = _levels("RFL_AGG_PARTICIPATION", (10, 25, 50, 80, 100), int)
EPOCHS = _levels("RFL_AGG_EPOCHS", (1, 10, 100), int)

# The reference rule (E = 1, every client): patience of the plateau, of the early stop,
# and round budget. The floors are participations per client (module docstring).
REFERENCE = {"plateau": 20, "stop": 150, "budget": 3000}
MIN_PARTICIPATIONS = {"plateau": 5, "stop": 30, "budget": 300}

NAME = f"mi_{DATASET}_agg"


def schedule(percent: int, epochs: int, n_clients: int) -> dict:
    """Plateau patience, early-stop patience and round budget of an arm (docstring)."""
    m = max(1, math.ceil(n_clients * (percent / 100)))     # as FLServer._select_participants
    pe = min(m, n_clients) / n_clients
    return {k: max(math.ceil(REFERENCE[k] / (pe * epochs)),
                   math.ceil(MIN_PARTICIPATIONS[k] / pe)) for k in REFERENCE}


def variant_name(clients: str, percent: int, epochs: int) -> str:
    return f"{clients}_p{percent}_e{epochs}"


def variants(n_clients: int) -> dict:
    """One Variant per (clients, participation, local epochs) of the current subset."""
    out = {}
    for c in CLIENTS:
        if c not in ("subj", "iid"):
            raise SystemExit(f"RFL_AGG_CLIENTS: {c!r} is not subj or iid")
        for pct in PARTICIPATION:
            if not 0 < pct <= 100:
                raise SystemExit(f"RFL_AGG_PARTICIPATION: {pct} is not in (0, 100]")
            for e in EPOCHS:
                s = schedule(pct, e, n_clients)
                args = (("--client-partition", "iid") if c == "iid" else ()) + \
                       (("--client-fraction", f"{pct / 100:g}") if pct < 100 else ()) + \
                       (("--local-epochs", str(e)) if e > 1 else ()) + \
                       ("--lr-plateau-patience", str(s["plateau"]),
                        "--early-stop-patience", str(s["stop"]))
                out[variant_name(c, pct, e)] = Variant(args=args,
                                                       space={"horizon": [s["budget"]]})
    return out


def study() -> Protocol:
    """The Protocol of the study, with the calibrated architecture (mi_agg_calib)."""
    return Protocol(
        dataset=DATASET,
        models=("spdnet",),
        aggregations={"spdnet": RULES},
        kinds=("fed_fedper",),
        model_args={"spdnet": TRAINING + PLATEAU + calibrated()},
        variants=variants(len(get_subjects(DATASET))),
        space={"nodp": {"lr": LR, "batch_size": SETTINGS["batches"]}, "dp": {}},
        search="staged",
        stage_keys=[["lr", "batch_size"]],
        sweep_seeds=SWEEP_SEEDS,
        final_seeds=FINAL_SEEDS,
        name=NAME,
    )


STAGES = {"sweep1": lambda: study().sweep(0), "final": lambda: study().final()}

# Every step of the study, for rfl/experiments/pipeline.py and condor/submit_dag.sh.
PIPELINE = [
    Command("floors", floors_command(), done=floors_ready),
    ("mi_agg_calib", "sweep1"), ("mi_agg_calib", "sweep2"), ("mi_agg_calib", "sweep3"),
    ("mi_agg", "sweep1"), ("mi_agg", "final"),
]
