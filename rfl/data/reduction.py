"""Fewer calibration trials per subject: N trials per class, for a data-quantity study.

The split of the runners is made first (rfl/federated/server.py, the same in the
centralised runs), then each reduced subject keeps only N trials per class of its
training and validation trials (its *calibration pool*):

    pool        the subject's training and validation trials of the global split. The
                test trials are not touched: the test set is the same for every N.
    order       'random': each class of the pool is shuffled with a generator seeded by
                (seed, subject position), the same for every N. 'chronological': the
                trials keep their recording order (MOABB loads them in time order).
    keep        the first N trials of each class in that order. The subsets are nested:
                the trials kept at N are kept at 2N.
    split       of the n = min(N, pool) trials kept for a class, the last
                v = min(n - 1, max(1, round(val_fraction * n))) go to validation and the
                others to training: none when n = 1, one when n is 2 to 7 at the default
                fraction 0.2 (a share of 50 % at n = 2, 20 % from n = 5).

A pool with fewer than N trials of a class keeps all of them (the runners log it).

With `reduced_subjects` = K > 0 only K subjects, drawn with the run seed, are reduced
(the heterogeneous scenario: a few clients with little data among clients with all
their data). K = 0 reduces every subject, and K larger than the number of subjects is
refused.

On real EEG the run seed also draws the train/validation/test split. A synthetic .npz
stores its split: there the seed only draws the reduction and the reduced subjects.

The same function serves real EEG (indices into the loaded trials) and synthetic data
(the stored clients, one subject each).
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

ORDERS = ("random", "chronological")


def calibration_positions(y: np.ndarray, n_per_class: int, val_fraction: float,
                          order: str, rng: np.random.Generator
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """(training positions, validation positions) of the kept trials of a pool.

    Positions index `y`, the labels of the pool in recording order (module docstring).
    """
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    y = np.asarray(y)
    train, val = [], []
    for c in np.unique(y):
        pos = np.flatnonzero(y == c)
        if order == "random":
            pos = rng.permutation(pos)
        kept = pos[:n_per_class]
        n = len(kept)
        v = 0 if n < 2 else min(n - 1, max(1, int(round(val_fraction * n))))
        train.append(kept[:n - v])
        val.append(kept[n - v:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(val))


def subject_rng(seed: int, position: int) -> np.random.Generator:
    """Generator of the subject at `position` in the subject list: the same for every N."""
    return np.random.default_rng([int(seed), int(position), 7919])


def reduced_positions(n_subjects: int, k: int, seed: int) -> Set[int]:
    """Positions of the reduced subjects: all when k = 0, else k drawn with `seed`."""
    if k > n_subjects:
        raise ValueError(f"--reduced-subjects {k}: there are only {n_subjects} subjects")
    if k <= 0 or k == n_subjects:
        return set(range(n_subjects))
    rng = np.random.default_rng([int(seed), 104729])
    return {int(i) for i in rng.choice(n_subjects, size=k, replace=False)}


def reduce_pool(train_idx: Sequence[int], val_idx: Sequence[int], y_all: np.ndarray,
                n_per_class: int, val_fraction: float, order: str,
                rng: np.random.Generator) -> Tuple[List[int], List[int], Dict[str, int]]:
    """Reduce one subject given as indices into the loaded trials.

    Returns:
        (training indices, validation indices, info) with info = {'pool': pool size,
        'short_classes': classes with fewer than N trials in the pool}.
    """
    pool = np.array(sorted(list(train_idx) + list(val_idx)), dtype=int)
    yp = np.asarray(y_all)[pool]
    tr, va = calibration_positions(yp, n_per_class, val_fraction, order, rng)
    short = int(sum(int((yp == c).sum()) < n_per_class for c in np.unique(yp)))
    return pool[tr].tolist(), pool[va].tolist(), {"pool": int(len(pool)),
                                                  "short_classes": short}


def reduce_arrays(split: Dict[str, np.ndarray], n_per_class: int, val_fraction: float,
                  order: str, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Reduce one synthetic client (X_train, y_train, X_val, y_val, X_test, y_test).

    The pool is the training trials followed by the validation trials, in stored order.
    """
    X = np.concatenate([split["X_train"], split["X_val"]])
    y = np.concatenate([split["y_train"], split["y_val"]])
    tr, va = calibration_positions(y, n_per_class, val_fraction, order, rng)
    return {"X_train": X[tr], "y_train": y[tr], "X_val": X[va], "y_val": y[va],
            "X_test": split["X_test"], "y_test": split["y_test"]}


def reduce_indices(train_idx, val_idx, y_all: np.ndarray, subject_of: np.ndarray,
                   subjects: Sequence[int], config) -> Tuple[list, list, List[int]]:
    """Reduce the subjects of a pooled split (the centralised runs).

    The same draw as the federated runs (rfl/federated/server.py): subject s at position
    p of `subjects` is reduced when p is drawn, with the generator of p.

    Returns:
        (training indices, validation indices, reduced subjects).
    """
    n = int(config.train_trials_per_class)
    chosen = reduced_positions(len(subjects), int(config.reduced_subjects), int(config.seed))
    subject_of = np.asarray(subject_of)
    tr_all, va_all, red = [], [], []
    for p, sub in enumerate(subjects):
        tr = [i for i in train_idx if subject_of[i] == sub]
        va = [i for i in val_idx if subject_of[i] == sub]
        if p in chosen:
            tr, va, info = reduce_pool(tr, va, y_all, n,
                                       float(config.calibration_val_fraction),
                                       config.calibration_order, subject_rng(config.seed, p))
            red.append(int(sub))
            if info["short_classes"]:
                print(f"[Data] subject {sub}: {info['short_classes']} class(es) with fewer "
                      f"than {n} trials in the pool of {info['pool']}: all of them are kept")
        tr_all += tr
        va_all += va
    return sorted(tr_all), sorted(va_all), red


def reduce_synthetic(splits: List[Dict[str, np.ndarray]], config) -> Tuple[list, List[int]]:
    """Reduce the synthetic clients as the config says. Returns (splits, reduced positions)."""
    n = config.train_trials_per_class
    if not n:
        return splits, []
    chosen = reduced_positions(len(splits), int(config.reduced_subjects), int(config.seed))
    out = [reduce_arrays(s, int(n), float(config.calibration_val_fraction),
                         config.calibration_order, subject_rng(config.seed, i))
           if i in chosen else s for i, s in enumerate(splits)]
    return out, sorted(chosen)


# --------------------------------------------------------------------------- #
#  Command line and configuration (both runners and run_tslr)
# --------------------------------------------------------------------------- #
def add_reduction_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data quantity (rfl/data/reduction.py)")
    g.add_argument("--train-trials-per-class", type=int, default=None,
                   help="Keep N calibration trials per class and subject (training and "
                        "validation), drawn from the subject's training and validation "
                        "trials. The test trials are unchanged. Default: all trials.")
    g.add_argument("--calibration-val-fraction", type=float, default=0.2,
                   help="Share of the N trials of each class used for validation (at "
                        "least one when N >= 2).")
    g.add_argument("--calibration-order", default="random", choices=ORDERS,
                   help="Which N trials: shuffled per subject with the run seed, or the "
                        "first ones in recording order. Nested in N either way.")
    g.add_argument("--reduced-subjects", type=int, default=0,
                   help="Reduce only K subjects, drawn with the run seed; the others "
                        "keep all their trials (heterogeneous scenario). 0: every subject.")


def reduction_config(args) -> dict:
    """The data-quantity fields of the configuration, from parsed arguments."""
    return dict(train_trials_per_class=args.train_trials_per_class,
                calibration_val_fraction=args.calibration_val_fraction,
                calibration_order=args.calibration_order,
                reduced_subjects=args.reduced_subjects)


def validate(config) -> None:
    """Check the data-quantity fields of an ExperimentConfig (ValueError)."""
    n = config.train_trials_per_class
    if n is not None and int(n) < 1:
        raise ValueError(f"train_trials_per_class must be >= 1, got {n}")
    if not 0.0 < float(config.calibration_val_fraction) < 1.0:
        raise ValueError("calibration_val_fraction must lie in (0, 1), got "
                         f"{config.calibration_val_fraction}")
    if config.calibration_order not in ORDERS:
        raise ValueError(f"calibration_order must be one of {ORDERS}, got "
                         f"{config.calibration_order!r}")
    if int(config.reduced_subjects) < 0:
        raise ValueError(f"reduced_subjects must be >= 0, got {config.reduced_subjects}")
