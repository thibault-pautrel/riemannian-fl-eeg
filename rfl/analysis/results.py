"""Load finished runs into a pandas DataFrame, one row per run.

Both runners write a JSON file that contains the full configuration of the run next to its
metrics: ``fl_results.json`` (rfl/experiments/run_federated_experiment.py) and
``centralized_results.json`` (rfl/experiments/run_centralized_experiment.py). The loader reads
the configuration from that file, never from the directory names, so it works with any
output layout.

Columns of the table returned by :func:`load_runs`:

Identification (see ``ID_COLUMNS``)
    ``kind`` (``centralized`` or ``federated``), ``dataset``, ``arch``, ``aggregation``,
    ``scope``, ``clients`` (``subject`` or ``iid``, the client partition of a
    federated run), ``participation``, ``eps``, ``accounting``, ``lr``, ``batch_size``,
    ``horizon`` (epochs or rounds), ``seed``.
Metrics
    ``test_acc``, ``test_f1``: the reported test metrics, at the round or epoch given by
    the run's ``report_rule``.
    ``last_test_acc``: test accuracy at the last round or epoch.
    ``val_acc``, ``val_loss``: validation metrics at the reported round or epoch.
    ``client_mean_acc``, ``client_min_acc``: mean and minimum over the clients of their
    own test accuracy (federated runs). ``test_acc`` pools the predictions, so a client
    with more test trials weighs more. ``client_mean_acc`` weighs every client (every
    subject, with one subject per client) equally, and ``client_min_acc`` is the worst
    client.
    ``refit_test_acc``, ``refit_test_f1``: test metrics after the head re-fit (NaN
    without re-fit).
    ``reduced_val_acc``, ``reduced_test_acc``, ``reduced_refit_test_acc``: mean over the
    reduced clients (rfl/data/reduction.py) of their own accuracies; every client when
    none is reduced (federated runs). With ``--reduced-subjects K`` these are the
    clients with little data. ``n_per_class`` (calibration trials per class, NaN: all)
    and ``reduced_subjects`` (0: all reduced) identify the data quantity.
DP (private federated runs only)
    ``C``, ``sigma``, ``mu0``, ``realized_eps``.
Other
    ``config_id``: hash of the full configuration without the seed and the output
    location. Two runs with the same ``config_id`` differ only by their seed, so a
    group of runs should hold each seed at most once (see ``duplicate_seeds``).
    ``agg_ms``: median aggregation time per round, in milliseconds.
    ``n_steps``: rounds or epochs actually run. ``path``: directory of the run.

``arch`` names the architecture together with its normalisation: ``spdnet``,
``spdnet_<widths>_<floors>`` when the blocks are given explicitly, with ``_bn`` for the
SPD batch norm, ``eegnet_nobn`` (EEGNet without normalisation) or ``eegnet_bn``
(BatchNorm). A normalisation shared by the clients (``fl_fedbn`` false) adds
``shared``: ``eegnet_bnshared``, ``spdnet_bnshared``.

``scope`` names the personalisation of a federated run (``FLConfig.personalisation_name``):
``shared`` (trunk and head), ``fedper`` (personal head trained during the rounds),
``fedper_frozen`` (personal head pinned during the rounds) or ``local`` (nothing
shared), with ``_rnd`` when the trunk is frozen and ``+refit`` with a head re-fit. A
centralised run has no scope, or ``rnd`` when its trunk is frozen.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

RESULT_FILES = {"fl_results.json": "federated",
                "centralized_results.json": "centralized"}

ID_COLUMNS = ["kind", "dataset", "model", "arch", "aggregation", "scope", "clients",
              "n_per_class", "reduced_subjects", "participation", "eps",
              "accounting", "lr", "batch_size", "horizon", "hidden_dim", "clip_C",
              "weight_decay", "local_epochs", "arm", "seed"]
# Hyperparameters: their selected values differ from arm to arm in a final stage.
HP_COLUMNS = ("lr", "batch_size", "horizon", "hidden_dim", "clip_C", "weight_decay",
              "local_epochs")
METRIC_COLUMNS = ["test_acc", "test_f1", "last_test_acc", "val_acc", "val_loss",
                  "client_mean_acc", "client_min_acc",
                  "refit_test_acc", "refit_test_f1",
                  "reduced_val_acc", "reduced_test_acc", "reduced_refit_test_acc",
                  "C", "sigma", "mu0", "realized_eps", "agg_ms", "n_steps"]


def _num(x: Any) -> float:
    """Float value of ``x``, NaN when missing or not a finite number."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def arch_name(cfg: Dict[str, Any]) -> str:
    """Architecture label of a run from its configuration (see module docstring)."""
    model = str(cfg.get("model_type", "?")).lower()
    shared = "" if cfg.get("fl_fedbn", True) else "shared"
    if model == "spdnet":
        name = "spdnet"
        if str(cfg.get("spdnet_hidden_dims") or "").strip():
            q = cfg.get("spdnet_reeig_percentile")
            floors = (f"q{float(q):g}" if q is not None
                      else str(cfg.get('spdnet_epsilons') or '').replace(' ', ''))
            name = f"spdnet_{str(cfg['spdnet_hidden_dims']).replace(' ', '')}_{floors}"
        if cfg.get("spdnet_use_batch_norm"):
            name += "_bn" + shared
        return name
    if model != "eegnet":
        return model
    norm = str(cfg.get("eegnet_norm_type", "none")).lower()
    name = "eegnet_" + {"batch": "bn", "none": "nobn"}.get(norm, norm)
    return name + shared if norm == "batch" else name


def scope_name(cfg: Dict[str, Any]) -> str:
    """Personalisation of a federated run from its configuration (module docstring)."""
    if str(cfg.get("model_type")) == "tslr":           # rfl/experiments/run_tslr.py
        return str(cfg.get("tslr_mode", "local"))
    if cfg.get("fl_local_only"):
        name = "local"
    elif cfg.get("fl_local_fc"):
        frozen = str(cfg.get("fl_local_head_mode", "train")) == "frozen"
        name = "fedper_frozen" if frozen else "fedper"
    else:
        name = "shared"
    if cfg.get("freeze_trunk"):
        name += "_rnd"
    if str(cfg.get("fl_head_refit_epochs") or "").strip().strip("0,; "):
        name += "+refit"
    return name


def _federated_row(r: Dict[str, Any]) -> Dict[str, Any]:
    cfg = r.get("config") or {}
    final = r.get("final_metrics") or {}
    last = r.get("last_round_metrics") or {}
    hist = r.get("round_history") or {}
    dp = r.get("dp_budget") or {}
    private = bool(cfg.get("dp_enabled"))
    pers = r.get("personalized_metrics") or {}
    per = [x for x in (final.get("per_client_test_acc") or []) if x is not None]
    info = r.get("fl_info") or {}
    red = info.get("reduced_clients") or list(range(len(per)))

    def on_reduced(values) -> float:
        v = [values[i] for i in red if values and i < len(values) and values[i] is not None]
        return _num(np.mean(v)) if v else math.nan

    row = {
        "kind": "federated",
        "aggregation": cfg.get("fl_aggregation"),
        "scope": scope_name(cfg),
        "clients": (None if cfg.get("data_path")
                    else cfg.get("fl_client_partition", "subject")),
        "participation": _num(cfg.get("fl_client_fraction", 1.0)),
        "eps": _num(cfg.get("dp_target_eps")) if private else math.nan,
        "accounting": cfg.get("dp_accounting") if private else None,
        "horizon": cfg.get("fl_rounds"),
        "test_acc": _num(final.get("global_test_acc")),
        "test_f1": _num(final.get("global_test_f1")),
        "last_test_acc": _num(last.get("global_test_acc")),
        "val_acc": _num(final.get("global_val_acc")),
        "val_loss": _num(final.get("global_val_loss")),
        "client_mean_acc": _num(np.mean(per)) if per else math.nan,
        "client_min_acc": _num(np.min(per)) if per else math.nan,
        "refit_test_acc": _num(pers.get("personalized_test_acc")),
        "refit_test_f1": _num(pers.get("personalized_test_f1")),
        "reduced_val_acc": on_reduced(final.get("per_client_val_acc")),
        "reduced_test_acc": on_reduced(final.get("per_client_test_acc")),
        "reduced_refit_test_acc": on_reduced(pers.get("personalized_per_client_test_acc")),
        "C": _num(dp.get("C")),
        "sigma": _num(dp.get("sigma")),
        "mu0": _num(dp.get("mu0")),
        "realized_eps": _num(dp.get("realized_eps")),
        "n_steps": len(hist.get("round") or []),
    }
    agg = [x for x in (hist.get("agg_seconds") or []) if x is not None]
    row["agg_ms"] = 1000.0 * float(np.median(agg)) if agg else math.nan
    return row


def _centralized_row(r: Dict[str, Any]) -> Dict[str, Any]:
    cfg = r.get("config") or {}
    tr = r.get("training") or {}
    test = r.get("test_metrics") or {}
    last_rule = tr.get("report_rule") == "last"
    hist = tr.get("test_acc_history") or []
    return {
        "kind": "centralized",
        "aggregation": None,
        "scope": "rnd" if cfg.get("freeze_trunk") else None,
        "clients": None,
        "participation": math.nan,
        "eps": math.nan,
        "accounting": None,
        "horizon": cfg.get("epochs"),
        "test_acc": _num(test.get("accuracy")),
        "test_f1": _num(test.get("f1_macro")),
        "last_test_acc": _num(hist[-1]) if hist else math.nan,
        "val_acc": _num(tr.get("final_val_acc") if last_rule else tr.get("best_val_acc")),
        "val_loss": _num(tr.get("final_val_loss") if last_rule
                         else tr.get("best_val_loss")),
        "n_steps": tr.get("final_epoch"),
    }


_NOT_CONFIG = ("seed", "data_split_seed", "run_name", "experiment_name", "save_root",
               "experiment_dir", "config_hash", "dp_sigma", "device", "description",
               "data_root")


def config_id(cfg: Dict[str, Any]) -> str:
    """8-character hash of a configuration without its seed and output location."""
    import hashlib
    kept = {k: v for k, v in cfg.items() if k not in _NOT_CONFIG}
    return hashlib.md5(json.dumps(kept, sort_keys=True, default=str).encode()).hexdigest()[:8]


def duplicate_seeds(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Groups of ``by`` in which a seed appears more than once.

    Such a group mixes runs that differ in more than the seed (another learning rate,
    another option), so its mean and its paired tests are not meaningful. Returns one
    row per (group, seed) with the number of runs and of distinct configurations.
    """
    cols = list(by) + ["seed"]
    g = df.groupby(cols, dropna=False).agg(n_runs=("path", "size"),
                                           n_configs=("config_id", "nunique"))
    return g[g["n_runs"] > 1].reset_index()


def read_run(path: str) -> Optional[Dict[str, Any]]:
    """One table row from a result file, or None if the file cannot be read."""
    kind = RESULT_FILES.get(os.path.basename(path))
    try:
        with open(path) as f:
            r = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[warn] skipping unreadable {path}: {e}")
        return None
    cfg = r.get("config") or {}
    row = _federated_row(r) if kind == "federated" else _centralized_row(r)
    n_cal = cfg.get("train_trials_per_class")
    row.update(dataset=cfg.get("dataset"), model=cfg.get("model_type"), arch=arch_name(cfg),
               n_per_class=_num(n_cal) if n_cal else math.nan,
               reduced_subjects=int(cfg.get("reduced_subjects") or 0),
               lr=_num(cfg.get("learning_rate")), batch_size=cfg.get("batch_size"),
               hidden_dim=(cfg.get("spdnet_hidden_dims") or cfg.get("spdnet_hidden_dim")
                           if cfg.get("model_type") == "spdnet" else None),
               clip_C=_num(cfg.get("dp_clip_C")) if cfg.get("dp_enabled") else math.nan,
               weight_decay=_num(cfg.get("weight_decay")),
               local_epochs=cfg.get("fl_local_epochs") if kind == "federated" else None,
               arm=_arm_of(path), seed=cfg.get("seed"), config_id=config_id(cfg),
               diverged=bool(r.get("diverged", False)),
               path=os.path.dirname(path))
    return row


def _arm_of(path: str) -> Optional[str]:
    """The plan arm of a run written by experiments/grid.py (.../<arm>/<hp>/seed<s>/)."""
    parts = os.path.normpath(os.path.dirname(path)).split(os.sep)
    return parts[-3] if len(parts) >= 3 and parts[-1].startswith("seed") else None


def find_result_files(roots: Iterable[str]) -> List[str]:
    """Every result file under the given directories, sorted."""
    found = []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            found += [os.path.join(dirpath, f) for f in files if f in RESULT_FILES]
    return sorted(found)


def load_runs(*roots: str) -> pd.DataFrame:
    """Read every run under ``roots`` into a DataFrame with the columns described above.

    Raises:
        FileNotFoundError: if no result file is found.
    """
    rows = [row for p in find_result_files(roots) if (row := read_run(p)) is not None]
    if not rows:
        raise FileNotFoundError(f"no fl_results.json or centralized_results.json under "
                                f"{list(roots)}")
    df = pd.DataFrame(rows)
    return df.reindex(columns=ID_COLUMNS + METRIC_COLUMNS + ["diverged", "config_id", "path"])


def load_curve(path: str, key: str = "test_acc") -> np.ndarray:
    """Per-round (federated) or per-epoch (centralized) curve of one run.

    Args:
        path: run directory, as in the ``path`` column.
        key: ``'test_acc'``, ``'test_loss'``, ``'val_acc'`` or ``'val_loss'``. For a
            federated run the pooled ``global_<key>`` series of ``round_history`` is read,
            for a centralized run the ``<key>_history`` series.

    Returns:
        The curve as a float array, empty if the run did not record it.
    """
    for name, kind in RESULT_FILES.items():
        f = os.path.join(path, name)
        if os.path.exists(f):
            with open(f) as fh:
                r = json.load(fh)
            if kind == "federated":
                values = (r.get("round_history") or {}).get(f"global_{key}")
            else:
                values = (r.get("training") or {}).get(f"{key}_history")
            return np.asarray(values or [], dtype=float)
    raise FileNotFoundError(f"no result file in {path}")


def varying_columns(df: pd.DataFrame, exclude: Iterable[str] = ("seed",)) -> List[str]:
    """Identification columns that take more than one value in ``df``."""
    return [c for c in ID_COLUMNS
            if c not in set(exclude) and df[c].nunique(dropna=False) > 1]
