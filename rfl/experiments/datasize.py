"""Building blocks of the data-quantity plans: when does federated training beat local?

Each subject keeps N calibration trials per class (rfl/data/reduction.py). Two scenarios:

    U<N>   every subject reduced to N (uniform)
    H<N>   K subjects reduced to N, the others keep all their trials (heterogeneous:
           new users with a short calibration join a federation of data-rich users)
    full   no reduction

For each scenario, every combination of the two levels of personalisation is an arm:

    head            shared (fed_shared, and its head re-fit in the final stage),
                    personal (fed_fedper), or nothing shared (fed_local)
    normalisation   none (nobn), batch norm kept on each client (bn, FedBN), or shared
                    batch norm (bnshared, federated arms only)

The arm `fed_<model>_<agg>_<head>_<norm>_<scenario>` is tuned on its own: learning rate
x batch size, then horizon, selected on the validation accuracy of the reduced clients
(`reduced_val_acc`). Two settings follow N:

* the batch sizes (`batch_grid`): around half a client's training set in U<N>, the
  dataset batches in H<N> (one batch size for every client);
* the horizon (`horizon_grid`): with N trials a client runs ceil(n_train / B) steps per
  round, so the horizon grid grows as N falls, by sqrt(N_ref / N) (at most x4), and
  extends upward when the best value is the longest one.

The TS+LR baseline (rfl/experiments/run_tslr.py) has no training schedule: it chooses
its regularisation inside the run, so it only runs the final seeds (`tslr_jobs`).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from rfl.experiments.grid import Job
from rfl.experiments.protocol import Variant

FEDERATED = ("fed_shared", "fed_fedper")
ALL_KINDS = ("fed_shared", "fed_fedper", "fed_local")
SELECTION = {"prefer_smaller": ["horizon"], "metric": "reduced_val_acc"}


def scenarios(ns: Sequence[int], hetero_ns: Sequence[int], k_reduced: int,
              full: bool = True) -> List[Tuple[str, List[str], Optional[int], bool]]:
    """(name, arguments, N, heterogeneous) of every scenario (module docstring)."""
    out = [(f"U{n}", ["--train-trials-per-class", str(n)], n, False) for n in ns]
    if full:
        out.append(("full", [], None, False))
    out += [(f"H{n}", ["--train-trials-per-class", str(n), "--reduced-subjects",
                       str(k_reduced)], n, True) for n in hetero_ns]
    return out


def horizon_grid(base: Sequence[int], n: Optional[int], n_ref: int = 40) -> List[int]:
    """`base` scaled by min(4, sqrt(n_ref / n)) when n < n_ref (module docstring)."""
    f = 1.0 if n is None or n >= n_ref else min(4.0, math.sqrt(n_ref / n))
    return [max(1, int(round(h * f))) for h in base]


def batch_grid(n: int, n_classes: int, val_fraction: float = 0.2) -> List[int]:
    """[b, b/2, 2b], b the power of two nearest below half a client's training set."""
    b = 2 ** int(math.floor(math.log2(max(4.0, n_train(n, n_classes, val_fraction) / 2))))
    return [b, max(2, b // 2), 2 * b]


def n_train(n: int, n_classes: int, val_fraction: float = 0.2) -> int:
    """Training trials of a client reduced to N trials per class (rfl/data/reduction.py)."""
    v = 0 if n < 2 else min(n - 1, max(1, int(round(val_fraction * n))))
    return n_classes * (n - v)


def extend_rules(base: Dict[str, Dict], n_classes: int):
    """EXTEND of a data-quantity plan: `base`, with the batch size of a U<N> arm bounded
    by the client's training set (a larger batch gives the same runs)."""
    import re

    def rules(arm: str) -> Dict[str, Dict]:
        m = re.search(r"_U(\d+)$", arm)
        if not m or "batch_size" not in base:
            return base
        cap = 2 ** math.ceil(math.log2(max(2, n_train(int(m.group(1)), n_classes))))
        return {**base, "batch_size": {**base["batch_size"],
                                       "max": min(cap, base["batch_size"].get("max", cap))}}
    return rules


def variants(norms: Dict[str, Tuple[Sequence[str], Optional[Sequence[str]]]],
             scens, horizon_base: Sequence[int], full_batches: Sequence[int],
             n_classes: int) -> Dict[str, Variant]:
    """One Variant per (normalisation, scenario), named <norm>_<scenario>.

    `norms` maps a name to (arguments, arm kinds or None for every kind).
    """
    out = {}
    for sname, sargs, n, hetero in scens:
        batches = (list(full_batches) if (n is None or hetero)
                   else batch_grid(n, n_classes))
        space = {"batch_size": batches, "horizon": horizon_grid(horizon_base, n)}
        for norm, (args, kinds) in norms.items():
            out[f"{norm}_{sname}"] = Variant(args=tuple(args) + tuple(sargs), kinds=kinds,
                                             space=dict(space))
    return out


def tslr_jobs(data_args: Sequence[str], scens, seeds: Sequence[int],
              extra: Sequence[str] = ()) -> List[Job]:
    """Final runs of the TS+LR baseline, local and pooled, for every scenario."""
    jobs = []
    for sname, sargs, _, _ in scens:
        for mode in ("local", "pooled"):
            for s in seeds:
                jobs.append(Job(arm=f"tslr_{mode}_{sname}", entry="tslr", seed=s, hp={},
                                args=[*data_args, *extra, *sargs, "--mode", mode]))
    return jobs
