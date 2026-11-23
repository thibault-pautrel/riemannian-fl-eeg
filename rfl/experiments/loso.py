"""Inter-subject protocol: a model trained on source subjects, adapted to new subjects.

Folds. Each subject is a target once. BNCI2014_001 and Weibo2014 leave one subject out
(9 and 10 folds), PhysionetMI leaves out one of 10 groups of consecutive subjects
(`folds`). A run of a fold gets the targets with --target-subjects.

Split. The trials of every subject are split as in the within-subject protocol (the
global stratified split of the run seed, rfl/data/data_utils.py:split_indices), then
the targets are set apart (`split_targets`). The sources keep their training,
validation and test trials. A target keeps two disjoint parts:

    pool    its training and validation trials: the calibration recording
    test    its test trials, the same for every arm and every N

From the pool, N labeled trials per class are drawn as in the data-quantity study
(rfl/data/reduction.py:reduce_pool, generator of the subject's position and the run
seed): nested in N, and the same trials as the datasize runs of the same seed. The
targets never enter training, validation, early stopping or model selection.

Adaptation (`evaluate_targets`), after training, in the job that trained the model.
The model given to a new subject (`new_subject_model`) is the reported model. Its
normalisation layers are the shared ones (shared batch norm, centralised model), or,
with FedBN, the combination of the clients' layers (`combine_local_norms`): SPD running
means and SPD biases combined in the geometry of the layer, the parameters of torch
batch norms averaged. For each target, N and statistics:

    stats   'none'     those layers as they are
            'pool'     running statistics recomputed on the unlabeled pool
            'labeled'  recomputed on the N x K labeled trials only (N > 0)
    head    'shared'     the trained head as it is (shared head and centralised model)
            'shared_ft'  logistic regression on the frozen trunk's features, with an
                         L2 penalty towards the shared head (N > 0)
            'local'      the same towards 0: a new head (N > 0, every arm)

A head solves min_W CE(W) + lambda ||W - W0||^2 exactly (L-BFGS, float64), the bias
unpenalised. lambda = 1 / (2 C n), the scaling of scikit-learn's C, and C is chosen in
C_GRID by stratified cross-validation on the N labeled trials (accuracy, then loss,
then the smaller C). The trunk stays frozen. The records go to `target_metrics` of the
result file: one per target, N, stats and head, with the accuracy on the target's test.

TS+LR (`tslr_targets`, rfl/experiments/run_tslr.py --mode loso). Each source is
re-centred on its own mean (Zanini et al., 2018), one logistic regression is fitted on
the sources (C on their validation). A target is re-centred on the mean of its pool
('pool') or of its N labeled trials ('labeled'), then scored by the source model
('pooled'), by the source model refitted with its N trials ('pooled_plus', same C), or
by a model of its N trials alone ('local', C by cross-validation as above).

ReEig floors per fold (`python -m rfl.experiments.loso floors`): the floors file of
analyze_covariances, computed without the targets of the fold.
"""

from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROLES = ("adapt", "local")
# Groups of target subjects: None, one subject per fold; k, k groups of consecutive
# subjects.
GROUPS: Dict[str, Optional[int]] = {"BNCI2014_001": None, "Weibo2014": None,
                                    "PhysionetMI": 10}
# Calibration trials per class of the targets. PhysionetMI has about 19 trials per
# class in the pool of a subject: its values are those of the data-quantity study.
NS: Dict[str, List[int]] = {"BNCI2014_001": [0, 5, 10, 20], "Weibo2014": [0, 5, 10, 20],
                            "PhysionetMI": [0, 3, 5, 10]}
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)          # the grid of run_tslr.py
CV_FOLDS = 5


# --------------------------------------------------------------------------- #
#  Folds and configuration
# --------------------------------------------------------------------------- #
def folds(dataset: str, subjects: Optional[Sequence[int]] = None) -> List[List[int]]:
    """Target subjects of each fold of `dataset` (module docstring)."""
    if dataset not in GROUPS:
        raise SystemExit(f"{dataset}: no inter-subject folds (rfl/experiments/loso.py, "
                         f"GROUPS covers {sorted(GROUPS)})")
    from rfl.data.registry import get_subjects
    subs = list(subjects) if subjects is not None else get_subjects(dataset)
    k = GROUPS[dataset]
    if k is None:
        return [[s] for s in subs]
    return [[int(s) for s in g] for g in np.array_split(np.asarray(subs), k)]


def validate(config) -> None:
    """Check the inter-subject fields of an ExperimentConfig (ValueError)."""
    targets = list(config.target_subjects or [])
    ns = list(config.target_ns or [])
    role = config.target_role
    if role not in ROLES:
        raise ValueError(f"target_role must be one of {ROLES}, got {role!r}")
    if not targets:
        if ns or role != "adapt":
            raise ValueError("target_role and target_ns need target_subjects")
        return
    if config.data_path:
        raise ValueError("target_subjects needs a real dataset (--dataset)")
    if len(set(targets)) != len(targets):
        raise ValueError(f"target_subjects has duplicates: {targets}")
    missing = [s for s in targets if s not in config.subjects]
    if missing:
        raise ValueError(f"target_subjects {missing} are not among the subjects")
    if any(int(n) < 0 for n in ns) or len(set(ns)) != len(ns):
        raise ValueError(f"target_ns must be distinct non-negative numbers, got {ns}")
    if getattr(config, "split_type", "random") != "random":
        raise ValueError("target_subjects sets subjects apart after the random split: "
                         "use split_type 'random'")
    if getattr(config, "fl_client_partition", "subject") != "subject":
        raise ValueError("target_subjects sets subjects apart: use it with "
                         "--client-partition subject")
    if role == "adapt":
        if getattr(config, "fl_local_only", False):
            raise ValueError("target_role 'adapt' adapts a federated model: "
                             "--local-only has none (use target_role 'local')")
        if len(targets) >= len(config.subjects):
            raise ValueError("target_subjects leaves no source subject to train on")
        if config.train_trials_per_class:
            raise ValueError("with target_role 'adapt', the targets get their N trials "
                             "from target_ns: train_trials_per_class would reduce the "
                             "sources")
        if (ns and getattr(config, "fl_fedbn", False) and config.spdnet_use_batch_norm
                and config.model_type.lower() == "spdnet"):
            from rfl.models.spd_batchnorm import SHARED_MEAN_TYPES
            if config.spdnet_bn_mean_type not in SHARED_MEAN_TYPES:
                raise ValueError("a new subject combines the FedBN layers of the clients: "
                                 f"spdnet_bn_mean_type must be one of {SHARED_MEAN_TYPES}")
        return
    if ns:
        raise ValueError("target_ns acts with target_role 'adapt'")
    if not config.train_trials_per_class:
        raise ValueError("target_role 'local' trains on N trials per class of the "
                         "targets: set train_trials_per_class")
    if not getattr(config, "fl_local_only", False):
        raise ValueError("target_role 'local' is the local-from-scratch reference: it "
                         "needs the federated runner with local-only training")


def add_target_args(p: argparse.ArgumentParser, roles: bool = True) -> None:
    """--target-subjects, --target-ns and (federated runner) --target-role."""
    g = p.add_argument_group("inter-subject protocol (rfl/experiments/loso.py)")
    g.add_argument("--target-subjects", type=int, nargs="+", default=[],
                   help="Subjects set apart after the split: never trained on, adapted "
                        "and scored after training.")
    g.add_argument("--target-ns", type=int, nargs="+", default=[],
                   help="Calibration trials per class of the target adaptation (0: "
                        "unlabeled trials only). Empty: no adaptation.")
    if roles:
        g.add_argument("--target-role", default="adapt", choices=ROLES,
                       help="'local': the targets are the only clients (with "
                            "--local-only and --train-trials-per-class).")


def target_config(args) -> dict:
    """The inter-subject fields of the configuration, from parsed arguments."""
    return dict(target_subjects=[int(s) for s in args.target_subjects],
                target_ns=[int(n) for n in args.target_ns],
                target_role=getattr(args, "target_role", "adapt"))


def members(all_subjects: Sequence[int], config) -> List[int]:
    """The subjects that become clients (or the centralised training set)."""
    targets = set(config.target_subjects or [])
    if not targets:
        return list(all_subjects)
    if config.target_role == "local":
        return [s for s in all_subjects if s in targets]
    return [s for s in all_subjects if s not in targets]


# --------------------------------------------------------------------------- #
#  Target split
# --------------------------------------------------------------------------- #
@dataclass
class Target:
    """The trials of one target subject (module docstring, Split)."""
    subject: int
    X_pool: np.ndarray
    y_pool: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    labeled: Dict[int, np.ndarray] = field(default_factory=dict)  # N -> pool positions


def split_targets(X_all: np.ndarray, y_all: np.ndarray, subject_of: np.ndarray,
                  train_idx, val_idx, test_idx, all_subjects: Sequence[int],
                  config) -> List[Target]:
    """The Target of each subject of config.target_subjects (module docstring)."""
    from rfl.data.reduction import reduce_pool, subject_rng
    subject_of = np.asarray(subject_of)
    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
    out = []
    for sub in config.target_subjects:
        owned = np.flatnonzero(subject_of == sub)
        tr = [int(i) for i in owned if i in train_set]
        va = [int(i) for i in owned if i in val_set]
        te = np.array(sorted(int(i) for i in owned if i in test_set), dtype=int)
        pool = np.array(sorted(tr + va), dtype=int)
        labeled = {}
        for n in sorted(int(n) for n in config.target_ns if int(n) > 0):
            ktr, kva, _ = reduce_pool(tr, va, y_all, n, float(config.calibration_val_fraction),
                                      config.calibration_order,
                                      subject_rng(config.seed, list(all_subjects).index(sub)))
            labeled[n] = np.searchsorted(pool, np.sort(np.array(ktr + kva, dtype=int)))
        out.append(Target(int(sub), X_all[pool], y_all[pool], X_all[te], y_all[te], labeled))
    return out


# --------------------------------------------------------------------------- #
#  The model of a new subject
# --------------------------------------------------------------------------- #
def combine_local_norms(model, states: Sequence[Dict[str, Any]],
                        weights: Sequence[float]) -> None:
    """Set the normalisation layers of `model` to the combination of `states`, in place.

    SPD batch norm: the running means and the SPD biases are combined with `weights` in
    the geometry of the layer (rfl/models/spd_batchnorm.py). A bias is stored as the
    symmetric matrix S of its parametrization: the biases f(S_i) are combined, then
    mapped back to S. The clients trained their biases apart, so their mean on the
    manifold replaces the mean of the S_i that a shared batch norm takes during the
    rounds, where the S_i stay close. Other parameters (the scalar of 'mean_var_scalar')
    and the torch batch norms: averaged with `weights`.
    """
    import torch
    from rfl.federated.param_groups import identify_bn_buffer_names, identify_bn_param_names
    from rfl.models.spd_batchnorm import weighted_spd_mean
    from rfl.models.spdnet import BatchNormSPD
    spd, bias = {}, {}
    for n, m in model.named_modules():
        if BatchNormSPD is not None and isinstance(m, BatchNormSPD):
            spd[f"{n}.running_mean"] = m.mean_type
            bias[f"{n}.parametrizations.Covbias.original"] = (
                m.mean_type, m.parametrizations.Covbias[0])
    w = np.asarray(weights, dtype=float) / float(np.sum(weights))
    state = model.state_dict()
    for k in identify_bn_param_names(model) | identify_bn_buffer_names(model):
        vals = [s[k] for s in states]
        if k in spd:
            state[k] = weighted_spd_mean(vals, list(w), spd[k])
        elif k in bias:
            mean_type, f = bias[k]
            with torch.no_grad():
                B = weighted_spd_mean([f(v) for v in vals], list(w), mean_type)
                state[k] = f.right_inverse(B).to(vals[0].dtype)
        elif vals[0].is_floating_point():
            state[k] = sum(float(wi) * v.double() for wi, v in zip(w, vals)).to(vals[0].dtype)
        else:
            state[k] = torch.stack(vals).max()
    model.load_state_dict(state)


def new_subject_model(clients: Sequence[Any], reported_state: Dict[str, Any],
                      weights: Sequence[float]):
    """The reported federated model as a new subject receives it (module docstring)."""
    model = copy.deepcopy(clients[0].local_model)
    model.load_state_dict(reported_state)
    if clients[0].use_fedbn and clients[0]._bn_buffer_names:
        combine_local_norms(model, [c.local_model.state_dict() for c in clients], weights)
    return model


# --------------------------------------------------------------------------- #
#  Logistic heads
# --------------------------------------------------------------------------- #
def fit_head(F: np.ndarray, y: np.ndarray, n_classes: int, W0: np.ndarray,
             b0: np.ndarray, C: float) -> Tuple[np.ndarray, np.ndarray]:
    """argmin_{W, b} mean CE(F W^T + b, y) + lambda ||W - W0||^2, lambda = 1 / (2 C n).

    L-BFGS from (W0, b0) in float64. The problem is convex in (W, b).
    """
    import torch
    import torch.nn.functional as tf
    X = torch.as_tensor(F, dtype=torch.float64)
    Y = torch.as_tensor(np.asarray(y), dtype=torch.long)
    W0t = torch.as_tensor(W0, dtype=torch.float64).reshape(n_classes, X.shape[1])
    W = W0t.clone().requires_grad_(True)
    b = torch.as_tensor(b0, dtype=torch.float64).reshape(n_classes).clone().requires_grad_(True)
    lam = 1.0 / (2.0 * float(C) * len(Y))
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=500, history_size=20,
                            tolerance_grad=1e-10, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = tf.cross_entropy(X @ W.T + b, Y) + lam * ((W - W0t) ** 2).sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    return W.detach().numpy(), b.detach().numpy()


def _logits(F: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return F @ W.T + b


def _loss(logits: np.ndarray, y: np.ndarray) -> float:
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def choose_C(X: np.ndarray, y: np.ndarray, fit: Callable, grid: Sequence[float],
             seed: int) -> float:
    """C of `grid` by stratified cross-validation on (X, y): best accuracy, then lowest
    loss, then the smaller C. `fit(X, y, C)` returns a function of X giving logits (or
    log-probabilities). Without two trials of each class, the middle of the grid."""
    from sklearn.model_selection import StratifiedKFold
    counts = np.bincount(y)
    k = min(CV_FOLDS, int(counts[counts > 0].min()))
    if k < 2:
        return float(sorted(grid)[len(grid) // 2])
    splits = list(StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y))
    scores = []
    for C in grid:
        acc, loss = [], []
        for tr, va in splits:
            logits = fit(X[tr], y[tr], C)(X[va])
            acc.append(float((logits.argmax(1) == y[va]).mean()))
            loss.append(_loss(logits, y[va]))
        scores.append((float(np.mean(acc)), -float(np.mean(loss)), -float(C)))
    return float(-max(scores)[2])


def _scores(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import f1_score
    return {"acc": float((pred == y).mean()),
            "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "n_test": int(len(y))}


# --------------------------------------------------------------------------- #
#  Network adaptation
# --------------------------------------------------------------------------- #
def _dataset(config, X, y):
    from rfl.data.datasets import build_dataset
    return build_dataset(X, y, config.data_type, normalization_type=config.normalization_type,
                         estimator=config.cov_estimator, shrinkage=config.cov_shrinkage,
                         regularization=config.cov_regularization)


def features(model, ds, device) -> np.ndarray:
    """Inputs of the classifier head over `ds`, the model in evaluation mode."""
    import torch
    from torch.utils.data import DataLoader
    out: List[Any] = []
    handle = model.classifier.register_forward_pre_hook(
        lambda _m, args: out.append(args[0].detach().double().cpu()))
    model.eval()
    try:
        with torch.no_grad():
            for x, _ in DataLoader(ds, batch_size=256, shuffle=False):
                model(x.to(device))
    finally:
        handle.remove()
    return torch.cat(out).reshape(len(ds), -1).numpy()


def recalibrated(model, ds, device):
    """A copy of `model` whose batch-norm statistics are those of `ds`
    (rfl/models/bn_recalibration.py)."""
    from torch.utils.data import DataLoader
    from rfl.models.bn_recalibration import recalibrate
    m = copy.deepcopy(model)
    recalibrate(m, DataLoader(ds, batch_size=256, shuffle=False), device)
    return m


def _cv_seed(config, subject: int) -> int:
    """Seed of the cross-validation folds of one target, fixed by the run seed."""
    return int(np.random.default_rng([int(config.seed), int(subject), 31]).integers(2 ** 31))


def _record(t: Target, n: int, stats: str, head: str, logits: np.ndarray,
            C: Optional[float]) -> Dict[str, Any]:
    return {"subject": t.subject, "n": n, "stats": stats, "head": head, "C": C,
            **_scores(logits.argmax(1), t.y_test)}


def _fitted_heads(F_lab, y_lab, F_test, starts, n_classes: int, seed: int):
    """(head, test logits, C) of each start (name, (W0, b0)), C by cross-validation."""
    out = []
    for name, (W0, b0) in starts:
        def fit(X, y, C, W0=W0, b0=b0):
            W, b = fit_head(X, y, n_classes, W0, b0, C)
            return lambda Z: _logits(Z, W, b)
        C = choose_C(F_lab, y_lab, fit, C_GRID, seed)
        out.append((name, fit(F_lab, y_lab, C)(F_test), C))
    return out


def evaluate_targets(model, targets: Sequence[Target], config, device, n_classes: int,
                     shared_head: bool, verbose: bool = True) -> Dict[str, Any]:
    """The adaptations of the module docstring, for every target. Returns the
    `target_metrics` entry of the result file."""
    has_bn = bool(config.has_batch_norm())
    shared = (model.classifier.weight.detach().double().cpu().numpy(),
              model.classifier.bias.detach().double().cpu().numpy())
    starts = ([("shared_ft", shared)] if shared_head else []) + \
        [("local", tuple(np.zeros_like(a) for a in shared))]
    ns = sorted(int(n) for n in config.target_ns)
    labeled_ns = [n for n in ns if n > 0]
    records: List[Dict[str, Any]] = []
    for t in targets:
        test_ds = _dataset(config, t.X_test, t.y_test)
        pool_ds = _dataset(config, t.X_pool, t.y_pool)
        seed = _cv_seed(config, t.subject)
        # Statistics of the sources, or of the pool: the same for every N.
        for stats in ("none", "pool") if has_bn else ("none",):
            m = recalibrated(model, pool_ds, device) if stats == "pool" else model
            F_pool, F_test = features(m, pool_ds, device), features(m, test_ds, device)
            if shared_head and 0 in ns:
                records.append(_record(t, 0, stats, "shared", _logits(F_test, *shared), None))
            for n in labeled_ns:
                pos = t.labeled[n]
                for head, logits, C in _fitted_heads(F_pool[pos], t.y_pool[pos], F_test,
                                                     starts, n_classes, seed):
                    records.append(_record(t, n, stats, head, logits, C))
        # Statistics of the N labeled trials.
        for n in labeled_ns if has_bn else []:
            pos = t.labeled[n]
            lab_ds = _dataset(config, t.X_pool[pos], t.y_pool[pos])
            m = recalibrated(model, lab_ds, device)
            F_lab, F_test = features(m, lab_ds, device), features(m, test_ds, device)
            if shared_head:
                records.append(_record(t, n, "labeled", "shared", _logits(F_test, *shared),
                                       None))
            for head, logits, C in _fitted_heads(F_lab, t.y_pool[pos], F_test, starts,
                                                 n_classes, seed):
                records.append(_record(t, n, "labeled", head, logits, C))
    out = _summary(records, targets, ns)
    if verbose:
        _print_summary(out)
    return out


def _summary(records, targets, ns) -> Dict[str, Any]:
    return {"ns": ns, "subjects": [t.subject for t in targets],
            "pool_sizes": {str(t.subject): int(len(t.y_pool)) for t in targets},
            "test_sizes": {str(t.subject): int(len(t.y_test)) for t in targets},
            "labeled_sizes": {str(t.subject): {str(n): int(len(p)) for n, p in t.labeled.items()}
                              for t in targets},
            "records": records}


def _print_summary(out: Dict[str, Any]) -> None:
    rows: Dict[tuple, List[float]] = {}
    for r in out["records"]:
        rows.setdefault((r["stats"], r["head"], r["n"]), []).append(r["acc"])
    print("[targets] mean test accuracy over the targets " + str(out["subjects"]))
    for (stats, head, n), v in sorted(rows.items()):
        print(f"  N={n:<3d} stats={stats:8s} head={head:10s} {np.mean(v):.4f}")


# --------------------------------------------------------------------------- #
#  TS+LR
# --------------------------------------------------------------------------- #
def tslr_targets(X_src: np.ndarray, y_src: np.ndarray, C: float,
                 targets: Sequence[Target], config, grid: Sequence[float] = C_GRID,
                 verbose: bool = True) -> Dict[str, Any]:
    """TS+LR on the targets (module docstring). (X_src, y_src): the re-centred tangent
    vectors of the sources' training trials, C: the C chosen on their validation."""
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression

    from rfl.experiments.run_tslr import _trace_normalise
    n_classes = int(y_src.max()) + 1

    def lr_fit(X, y, c):
        clf = LogisticRegression(C=c, max_iter=5000).fit(X, y)

        def logp(Z):                    # a class absent from y gets probability 0
            out = np.full((len(Z), n_classes), -np.inf)
            out[:, clf.classes_] = clf.predict_log_proba(Z)
            return out
        return logp

    pooled = lr_fit(X_src, y_src, C)
    ns = sorted(int(n) for n in config.target_ns)
    records: List[Dict[str, Any]] = []
    for t in targets:
        pool = _trace_normalise(_dataset(config, t.X_pool, t.y_pool).covs.astype(np.float64))
        test = _trace_normalise(_dataset(config, t.X_test, t.y_test).covs.astype(np.float64))
        seed = _cv_seed(config, t.subject)

        def recentred(ref):             # tangent vectors at the mean of `ref`
            ts = TangentSpace(metric="riemann").fit(ref)
            return ts.transform(pool), ts.transform(test)
        at_pool = recentred(pool)
        if 0 in ns:
            records.append(_record(t, 0, "pool", "pooled", pooled(at_pool[1]), C))
        for n in (n for n in ns if n > 0):
            pos = t.labeled[n]
            for stats, (V_pool, V_test) in (("pool", at_pool),
                                            ("labeled", recentred(pool[pos]))):
                if stats == "labeled":
                    records.append(_record(t, n, stats, "pooled", pooled(V_test), C))
                Vl, yl = V_pool[pos], t.y_pool[pos]
                both = lr_fit(np.concatenate([X_src, Vl]), np.concatenate([y_src, yl]), C)
                records.append(_record(t, n, stats, "pooled_plus", both(V_test), C))
                if len(np.unique(yl)) > 1:
                    c = choose_C(Vl, yl, lr_fit, grid, seed)
                    records.append(_record(t, n, stats, "local", lr_fit(Vl, yl, c)(V_test), c))
    out = _summary(records, targets, ns)
    if verbose:
        _print_summary(out)
    return out


# --------------------------------------------------------------------------- #
#  Plan building blocks (rfl/experiments/plans/mi_loso*.py)
# --------------------------------------------------------------------------- #
SOURCE_KINDS = ("central", "fed_shared", "fed_fedper")
FEDERATED_KINDS = ("fed_shared", "fed_fedper")


def fold_args(dataset: str, fold: int, floors: bool) -> Tuple[str, ...]:
    """The arguments of every run of a fold: its targets, and its own ReEig floors."""
    out = ("--target-subjects", *map(str, folds(dataset)[fold]))
    return out + (("--floors-file", floors_path(dataset, fold)) if floors else ())


def fold_variants(fold: int, bn: Sequence[str], ns: Sequence[int], n_classes: int):
    """The Variants of one fold: the source arms with the batch norm kept on each
    client (FedBN, also the centralised arm) or shared, and the local-from-scratch arms
    of the targets, one per N > 0, with their batch sizes (rfl/experiments/datasize.py)."""
    from rfl.experiments.datasize import batch_grid
    from rfl.experiments.protocol import Variant
    bn = tuple(bn)
    out = {f"bn_f{fold}": Variant(args=bn, kinds=SOURCE_KINDS),
           f"bnshared_f{fold}": Variant(args=bn + ("--no-fedbn",), kinds=FEDERATED_KINDS)}
    for n in (n for n in ns if n > 0):
        out[f"bn_U{n}_f{fold}"] = Variant(
            args=bn + ("--target-role", "local", "--train-trials-per-class", str(n),
                       "--bn-recalibrate"),
            kinds=("fed_local",), space={"batch_size": batch_grid(n, n_classes)})
    return out


def with_target_ns(jobs: Sequence[Any], ns: Sequence[int]) -> List[Any]:
    """The final runs of the source arms adapt their model to the targets: --target-ns.
    The sweeps leave it out, since they are read for their source validation only."""
    from rfl.experiments.grid import Job
    return [j if "--target-role" in j.args else
            Job(arm=j.arm, entry=j.entry, seed=j.seed, hp=dict(j.hp),
                args=[*j.args, "--target-ns", *map(str, ns)]) for j in jobs]


def extend_rules(base: Dict[str, Dict], n_classes: int) -> Callable[[str], Dict]:
    """EXTEND of an inter-subject plan: `base`, with the batch size of a local arm
    (..._U<N>_f<k>) bounded by the training set of a target reduced to N."""
    import math
    import re
    from rfl.experiments.datasize import n_train

    def rules(arm: str) -> Dict[str, Dict]:
        m = re.search(r"_U(\d+)_f\d+$", arm)
        if not m or "batch_size" not in base:
            return base
        cap = 2 ** math.ceil(math.log2(max(2, n_train(int(m.group(1)), n_classes))))
        return {**base, "batch_size": {**base["batch_size"], "min": 2,
                                       "max": min(cap, base["batch_size"].get("max", cap))}}
    return rules


# --------------------------------------------------------------------------- #
#  ReEig floors per fold
# --------------------------------------------------------------------------- #
def floors_dir(dataset: str, fold: int) -> str:
    """Folder of the floors file of one fold."""
    from rfl.experiments.grid import results_root
    return str(results_root() / "reeig_floors" / "loso" / f"{dataset}_f{fold}")


def floors_path(dataset: str, fold: int) -> str:
    from rfl.experiments.analyze_covariances import floors_file
    return floors_file(dataset, floors_dir(dataset, fold))


def floors_ready(dataset: str, architectures: Sequence[str]) -> bool:
    """Every fold has a floors file covering `architectures`, computed without the
    targets of the fold as they are now."""
    import json
    for k, targets in enumerate(folds(dataset)):
        path = Path(floors_path(dataset, k))
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        if (not all(a in data.get("architectures", {}) for a in architectures)
                or sorted(data.get("meta", {}).get("exclude_subjects", [])) != sorted(targets)):
            return False
    return True


def write_fold_floors(dataset: str, architectures: Sequence[str]) -> None:
    """The floors file of each fold, without its targets (analyze_covariances)."""
    from rfl.experiments import analyze_covariances as AC
    for k, targets in enumerate(folds(dataset)):
        print(f"fold {k}: targets {targets}")
        AC.main(["--datasets", dataset, "--max-subjects", "0",
                 "--exclude-subjects", *map(str, targets),
                 "--architectures", *architectures, "--floors-dir", floors_dir(dataset, k),
                 "--output-dir", os.path.join(floors_dir(dataset, k), "analysis")])


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Inter-subject folds and their ReEig floors.")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("folds", help="print the target subjects of each fold")
    f.add_argument("--dataset", required=True)
    fl = sub.add_parser("floors", help="write the floors file of every fold")
    fl.add_argument("--dataset", required=True)
    fl.add_argument("--architectures", nargs="+", required=True)
    a = p.parse_args(argv)
    if a.cmd == "folds":
        for k, t in enumerate(folds(a.dataset)):
            print(f"f{k}: {' '.join(map(str, t))}")
    else:
        write_fold_floors(a.dataset, a.architectures)


if __name__ == "__main__":
    main()
