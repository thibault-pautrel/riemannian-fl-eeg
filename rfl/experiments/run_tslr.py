"""Tangent space + logistic regression: the classical Riemannian baseline, per client.

The clients and their data are those of the federated runs (the same split, the same
covariances, the same reduction of rfl/data/reduction.py), so the baseline is paired
with every federated arm of the same seed. Two modes:

    local    each client maps its covariances to the tangent space at the Riemannian
             mean of its own training covariances (pyRiemann TangentSpace) and fits its
             own logistic regression. Nothing is shared.
    pooled   the same per-client tangent spaces, which re-centre every client on its
             own mean (Zanini et al., 2018), then one logistic regression on the training
             vectors of all clients. A non-federated transfer reference: the data are
             pooled.

The inverse regularisation C is chosen on the validation accuracy of the reduced clients
(all clients without reduction), from --C-grid, and reported in fl_results.json in the
format of the federated runner (rfl/analysis/results.py reads it; model 'tslr', scope
'local' or 'pooled').

    python -m rfl.experiments.run_tslr --dataset BNCI2014_001 --mode local \\
        --train-trials-per-class 10 --seed 1 --save-dir out/tslr
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Any, Dict, List

import numpy as np


def _build_arg_parser() -> argparse.ArgumentParser:
    from rfl.data.reduction import add_reduction_args
    from rfl.data.registry import DATASET_CONFIGS
    from rfl.experiments.cli_checks import HelpFormatter
    p = argparse.ArgumentParser(allow_abbrev=False, formatter_class=HelpFormatter,
                                description="Tangent space + logistic regression per "
                                            "client (local) or on re-centred pooled "
                                            "clients (pooled).")
    g = p.add_argument_group("data")
    g.add_argument("--dataset", default=None, choices=sorted(DATASET_CONFIGS))
    g.add_argument("--synthetic-clients", default=None,
                   help="Synthetic SPD .npz: each stored client is one subject.")
    g.add_argument("--subjects", type=int, nargs="+", default=None)
    g.add_argument("--task", default="all_classes",
                   choices=["binary", "all_classes", "no_rest"])
    g.add_argument("--resample", type=float, default=128.0)
    g.add_argument("--fmin", type=float, default=8.0)
    g.add_argument("--fmax", type=float, default=32.0)
    g.add_argument("--val-size", type=float, default=0.10)
    g.add_argument("--test-size", type=float, default=0.15)
    g.add_argument("--cov-estimator", default="scm", choices=["scm", "lwf"])
    g.add_argument("--cov-shrinkage", type=float, default=0.0)
    g.add_argument("--cov-regularization", type=float, default=0.0)
    add_reduction_args(p)
    g = p.add_argument_group("baseline")
    g.add_argument("--mode", default="local", choices=["local", "pooled"])
    g.add_argument("--C-grid", default="0.01,0.1,1,10,100",
                   help="Inverse regularisations tried; the best on validation is kept.")
    g.add_argument("--seed", type=int, default=42,
                   help="Seed of the split and of the reduction (as the other runners).")
    g.add_argument("--save-dir", required=True)
    return p


def _client_arrays(client) -> Dict[str, tuple]:
    """(matrices, labels) of each split of a client, as float64 arrays."""
    out = {}
    for split, loader in (("train", client.train_loader), ("val", client.val_loader),
                          ("test", client.test_loader)):
        ds = loader.dataset
        X = [np.asarray(ds[i][0], dtype=np.float64) for i in range(len(ds))]
        y = [int(ds[i][1]) for i in range(len(ds))]
        out[split] = (np.stack(X) if X else np.zeros((0, 1, 1)), np.asarray(y))
    return out


def _trace_normalise(X: np.ndarray) -> np.ndarray:
    tr = np.trace(X, axis1=1, axis2=2).clip(min=1e-12)
    return X / tr[:, None, None]


def tangent_features(clients_data: List[Dict[str, tuple]]) -> List[Dict[str, tuple]]:
    """Per client: the vectors of each split at the mean of its training matrices."""
    from pyriemann.tangentspace import TangentSpace
    out = []
    for d in clients_data:
        ts = TangentSpace(metric="riemann").fit(_trace_normalise(d["train"][0]))
        out.append({s: (ts.transform(_trace_normalise(X)) if len(X) else np.zeros((0, 1)), y)
                    for s, (X, y) in d.items()})
    return out


def fit_and_score(feats: List[Dict[str, tuple]], mode: str, C: float) -> List[Dict[str, float]]:
    """Per client: {'val': accuracy, 'test': accuracy, 'val_pred', 'test_pred'}."""
    from sklearn.linear_model import LogisticRegression

    def model():
        return LogisticRegression(C=C, max_iter=5000)

    if mode == "pooled":
        clf = model().fit(np.concatenate([f["train"][0] for f in feats]),
                          np.concatenate([f["train"][1] for f in feats]))
        clfs = [clf] * len(feats)
    else:
        clfs = []
        for f in feats:
            X, y = f["train"]
            if len(np.unique(y)) < 2:           # one class: predict it
                clfs.append(("const", int(y[0])))
            else:
                clfs.append(model().fit(X, y))
    out = []
    for f, clf in zip(feats, clfs):
        row = {}
        for s in ("val", "test"):
            X, y = f[s]
            if not len(y):
                row[s], row[f"{s}_pred"] = float("nan"), np.zeros(0, dtype=int)
                continue
            pred = (np.full(len(y), clf[1]) if isinstance(clf, tuple) else clf.predict(X))
            row[s], row[f"{s}_pred"] = float((pred == y).mean()), pred
        out.append(row)
    return out


def _metrics(feats, scores) -> Dict[str, Any]:
    from sklearn.metrics import f1_score
    out: Dict[str, Any] = {}
    for s in ("val", "test"):
        y = np.concatenate([f[s][1] for f in feats])
        p = np.concatenate([r[f"{s}_pred"] for r in scores])
        accs = [r[s] for r in scores]
        out[f"global_{s}_acc"] = float((p == y).mean()) if len(y) else float("nan")
        out[f"global_{s}_f1"] = (float(f1_score(y, p, average="macro", zero_division=0))
                                 if len(y) else float("nan"))
        out[f"avg_{s}_acc"] = float(np.nanmean(accs))
        out[f"std_{s}_acc"] = float(np.nanstd(accs))
        out[f"per_client_{s}_acc"] = [float(a) for a in accs]
    return out


def _parse(argv=None):
    from rfl.experiments.cli_checks import explicit_dests, inert_option_errors
    p = _build_arg_parser()
    args = p.parse_args(argv)
    if (args.dataset is None) == (args.synthetic_clients is None):
        p.error("provide exactly one of --dataset or --synthetic-clients")
    errors = inert_option_errors(p, args, explicit_dests(p, argv))
    if errors:
        p.error("options without effect in this run:\n  " + "\n  ".join(errors))
    try:
        C_grid = [float(c) for c in args.C_grid.split(",") if c.strip()]
    except ValueError:
        p.error(f"--C-grid {args.C_grid!r}: comma-separated numbers")
    if not C_grid or min(C_grid) <= 0:
        p.error("--C-grid needs positive values")
    from rfl.data.reduction import validate
    try:
        validate(args)
    except ValueError as e:
        p.error(str(e))
    return p, args, C_grid


def check_command(argv) -> None:
    """Parse and check a command line without loading data (rfl/experiments/grid.py)."""
    _parse(argv)


def main(argv=None) -> Dict[str, Any]:
    p, args, C_grid = _parse(argv)
    os.makedirs(args.save_dir, exist_ok=True)

    from rfl.data.reduction import reduction_config
    from rfl.federated.fl_config import FLConfig
    from rfl.federated.server import create_fl_clients, create_fl_clients_synthetic
    import torch
    is_syn = args.synthetic_clients is not None
    if not is_syn and args.subjects is None:
        from rfl.data.registry import get_subjects
        args.subjects = get_subjects(args.dataset)
    src = "synthetic" if is_syn else args.dataset
    config = FLConfig(
        experiment_name="tslr", dataset=src, subjects=[] if is_syn else args.subjects,
        data_path=os.path.abspath(args.synthetic_clients) if is_syn else "",
        model_type="spdnet", data_type="spd" if is_syn else "covariance",
        task=args.task, resample=args.resample, fmin=args.fmin, fmax=args.fmax,
        val_size=args.val_size, test_size=args.test_size,
        cov_estimator=args.cov_estimator, cov_shrinkage=args.cov_shrinkage,
        cov_regularization=args.cov_regularization, fl_aggregation="projavg",
        seed=args.seed, data_split_seed=args.seed, **reduction_config(args))
    device = torch.device("cpu")
    if is_syn:
        from rfl.data.synthetic import load_clients
        clients = create_fl_clients_synthetic(load_clients(args.synthetic_clients),
                                              config, device, verbose=False)
    else:
        clients = create_fl_clients(args.dataset, args.subjects, 1, config, device,
                                    verbose=True)
    data = [_client_arrays(c) for c in clients]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = tangent_features(data)
        reduced = [c.client_id for c in clients if c.reduced] or list(range(len(clients)))
        tried = {}
        for C in C_grid:
            scores = fit_and_score(feats, args.mode, C)
            tried[C] = (float(np.nanmean([scores[i]["val"] for i in reduced])), scores)
    scored = [c for c in C_grid if np.isfinite(tried[c][0])]
    if scored:
        best, rule = max(scored, key=lambda c: (tried[c][0], -C_grid.index(c))), "best val"
    else:
        # No validation trial on the reduced clients (N = 1): C cannot be chosen.
        best = 1.0 if 1.0 in C_grid else C_grid[len(C_grid) // 2]
        rule = "no validation trial: fixed"
        print(f"[tslr] warning: the reduced clients have no validation trial; C={best:g} "
              "is fixed instead of chosen")
    final = _metrics(feats, tried[best][1])

    cfg = config.to_dict()
    cfg.update(model_type="tslr", tslr_mode=args.mode, tslr_C_grid=C_grid)
    results = {
        "config": cfg,
        "fl_info": {"n_clients": len(clients),
                    "client_subjects": [c.subjects for c in clients],
                    "client_train_samples": [len(d["train"][1]) for d in data],
                    "client_val_samples": [len(d["val"][1]) for d in data],
                    "train_trials_per_class": config.train_trials_per_class,
                    "reduced_clients": [c.client_id for c in clients if c.reduced]},
        "personalisation": args.mode,
        "report_rule": "last",
        "final_metrics": final,
        "last_round_metrics": final,
        "personalized_metrics": None,
        "round_history": {},
        "tslr": {"mode": args.mode, "C": best, "C_rule": rule,
                 "val_acc_by_C": {f"{c:g}": tried[c][0] for c in C_grid}},
    }
    with open(os.path.join(args.save_dir, "fl_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[tslr] {args.mode}: C={best:g}  val (reduced clients) {tried[best][0]:.4f}  "
          f"test {final['global_test_acc']:.4f} (pooled), "
          f"{final['avg_test_acc']:.4f} (client mean)")
    return results


if __name__ == "__main__":
    main()
