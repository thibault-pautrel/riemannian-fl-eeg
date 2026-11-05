#!/usr/bin/env python3
# file: rfl/experiments/analyze_covariances.py
"""
Covariance spectrum analysis of MOABB datasets, used to choose the SPDNet ReEig floors.

ReEig maps each eigenvalue lambda to max(lambda, eps). The script estimates
covariances as the training pipeline does (build_dataset(..., 'covariance')) and
reports the raw spectrum and the spectrum in the calibration frame (trace-normalised
by default, as in SPDNet). The recommended epsilon_1 is a percentile of that spectrum,
so that ReEig clamps about that fraction of the eigenvalues.

With --hidden-dim-block1 it also calibrates a two-block SPDNet. After a BiMap
congruence by a semi-orthogonal W, the spectrum stays inside the previous spectral
interval (Poincare separation). So the second floor must satisfy eps_2 > eps_1, or the
second ReEig is the identity and the two blocks collapse into one (Oswald and Massart,
arXiv:2606.02490, Prop. 2 and Cor. 2). The script therefore
  * re-calibrates eps_1 on the spectrum after the first congruence, which is the one
    the first ReEig sees,
  * takes eps_2 at the same percentile of the eigenvalues left strictly above eps_1,
    which guarantees eps_2 > eps_1.
The congruence uses a random Stiefel W. This fixes the order of magnitude only, since a
trained W shifts the spectrum.

With --architectures it calibrates every floor where its ReEig acts, for each listed
architecture ("16", "64,32", ...), with and without the SPD batch norm, at each
percentile of --percentiles, and writes them to <floors-dir>/<dataset>.json. The runners
read them with --reeig-percentile Q (rfl/experiments/cli_checks.py), and the plans of the
batch-norm study tune Q (rfl/experiments/plans/mi_spdnet_bn_calib.py). Block k sees the output of block k - 1
(after its ReEig, and after its batch norm when there is one) through the congruence by
a random Stiefel W_k. Its floor is the percentile of that spectrum. Without batch norm a
floor must exceed the previous one: when the percentile does not, it is taken over the
eigenvalues left strictly above the previous floor (as for eps_2 above). With batch
norm, the layer follows each ReEig (bn_position 'after_reeig') and whitens the matrices
by their GAH mean (the library's `geometric_arithmetic_harmonic_mean`, the running mean
of a layer whose bias is still the identity), so the next block sees matrices centred
near the identity.

Writes covariance_analysis.json and covariance_analysis.txt in --output-dir. It never
trains a model.

Usage (from any folder once the repository is installed, see README):
    python -m rfl.experiments.analyze_covariances
    python -m rfl.experiments.analyze_covariances --datasets BNCI2014_001 --hidden-dim-block1 16

Requirements: MOABB (datasets are downloaded on first use).
"""

import os
import json
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


# MOABB and data_utils are imported lazily, in load_covariances.
from rfl.data.registry import DATASET_CONFIGS, get_subjects, get_sampling_rate


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts NumPy arrays and scalars to Python types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# --------------------------------------------------------------------------- #
#  Spectrum analysis (no MOABB needed)
# --------------------------------------------------------------------------- #
def spectrum_stats(eigs: np.ndarray) -> Dict[str, Any]:
    """Summary statistics of an (N, C) array of ascending eigenvalues, one row per matrix."""
    N, C = eigs.shape
    flat = eigs.reshape(-1)
    per_min = eigs[:, 0]
    per_max = eigs[:, -1]
    pos = np.clip(per_min, 1e-30, None)
    cond = per_max / pos
    pcts = [0.1, 1, 5, 15, 50, 95, 99, 99.9]
    return {
        "n_matrices": int(N), "dim": int(C),
        "eig_mean": float(flat.mean()), "eig_median": float(np.median(flat)),
        "eig_percentiles": {f"p{p}": float(np.percentile(flat, p)) for p in pcts},
        "min_eig_mean": float(per_min.mean()),
        "min_eig_p1": float(np.percentile(per_min, 1)),
        "min_eig_min": float(per_min.min()),
        "max_eig_mean": float(per_max.mean()), "max_eig_max": float(per_max.max()),
        "log10_cond_mean": float(np.log10(cond).mean()),
        "log10_cond_p99": float(np.percentile(np.log10(cond), 99)),
        "spd_percentage": float(100.0 * np.mean(per_min > 0)),
        "effective_rank_mean": float(np.mean(np.sum(eigs > 1e-6 * per_max[:, None], axis=1))),
    }


def epsilon_clamp_table(eigs_scaled: np.ndarray, candidates: List[float]) -> List[Dict[str, float]]:
    """Fraction of eigenvalues, and of matrices, that each candidate epsilon would clamp."""
    flat = eigs_scaled.reshape(-1)
    table = []
    for eps in sorted(candidates):
        below = eigs_scaled < eps
        table.append({
            "epsilon": float(eps),
            "frac_eigenvalues_clamped": float(np.mean(flat < eps)),
            "frac_matrices_clamped": float(np.mean(below.any(axis=1))),
        })
    return table


def _round_1sig(x: float) -> float:
    """Round to one significant figure."""
    if not np.isfinite(x) or x <= 0:
        return float(x)
    from math import floor, log10
    return float(round(x, -int(floor(log10(x)))))


def recommend_epsilon(eigs_cal: np.ndarray, percentile: float) -> float:
    """ReEig floor that clamps about `percentile` percent of the eigenvalues.

    Args:
        eigs_cal: Eigenvalues in the calibration frame, i.e. the spectrum ReEig sees.
            Use the trace-normalised spectrum when the model trace-normalises.
        percentile: Target clamp percentage.

    Returns:
        The `percentile`-th percentile of all eigenvalues.
    """
    return float(np.percentile(eigs_cal.reshape(-1), percentile))


def post_block_spectrum(covs: np.ndarray, h: int, eps1: float, seed: int = 0,
                        trace_normalize: bool = True) -> Dict[str, Any]:
    """Spectrum entering the second ReEig, after one BiMap and ReEig block.

    W is a random point of St(d, h) drawn with `seed`. This is enough to fix the order
    of magnitude of eps_2. A trained W shifts the spectrum somewhat.

    Args:
        covs: SPD matrices of shape (N, d, d).
        h: Output dimension of the first BiMap.
        eps1: Floor of the first ReEig.
        seed: Seed of the random Stiefel W.
        trace_normalize: Divide each matrix by its trace first, as SPDNet does.

    Returns:
        Dict with 'eigs' (sorted eigenvalues after the eps1 floor, shape (N, h)),
        'eigs_prefloor' (the same before the floor) and 'frac_clamped_by_eps1'.

    Raises:
        ValueError: if h > d.
    """
    x = np.asarray(covs, dtype=np.float64)
    if trace_normalize:
        x = x / np.trace(x, axis1=-2, axis2=-1)[:, None, None]
    d = x.shape[-1]
    if h > d:
        raise ValueError(f"hidden dim {h} > input dim {d}")
    W, _ = np.linalg.qr(np.random.RandomState(seed).normal(size=(d, h)))
    lam = np.linalg.eigvalsh(np.einsum('ij,njk,kl->nil', W.T, x, W))
    lam = np.sort(lam, axis=1)
    clamped = float(np.mean(lam < eps1))
    return {"eigs": np.clip(lam, eps1, None), "eigs_prefloor": lam,
            "frac_clamped_by_eps1": clamped}


def _round_floor(exact: float, previous: Optional[float]) -> float:
    """Two significant figures, or three when two would not exceed `previous`."""
    r = float(f"{exact:.2g}")
    if previous is not None and r <= previous:
        r = float(f"{exact:.3g}")
    return r


def _gah_whiten(y: np.ndarray) -> np.ndarray:
    """y_k -> M^{-1/2} y_k M^{-1/2}, M the GAH mean of the batch (yetanotherspdnet)."""
    import torch
    from yetanotherspdnet.functions.spd_geometries.kullback_leibler_symmetrized import (
        geometric_arithmetic_harmonic_mean)
    M = geometric_arithmetic_harmonic_mean(torch.from_numpy(y)).numpy()
    lam, U = np.linalg.eigh((M + M.T) / 2)
    isq = (U / np.sqrt(lam)) @ U.T
    return isq @ y @ isq


def architecture_floors(covs: np.ndarray, dims: List[int], percentile: float,
                        batch_norm: bool, seed: int = 0,
                        trace_normalize: bool = True) -> Dict[str, Any]:
    """ReEig floors of one architecture, each calibrated on the spectrum it acts on.

    See the module docstring (--architectures). W_k is a random point of St(d_k, h_k)
    drawn with `seed + k`.

    Returns:
        Dict with 'floors' (one per block), 'exact' (before rounding) and 'clamped'
        (fraction of the eigenvalues each floor clamps).

    Raises:
        ValueError: on a dimension above the previous one, or when a floor without
            batch norm cannot exceed the previous one (the block would be inert).
    """
    z = np.asarray(covs, dtype=np.float64)
    if trace_normalize:
        z = z / np.trace(z, axis1=-2, axis2=-1)[:, None, None]
    floors: List[float] = []
    exact: List[float] = []
    clamped: List[float] = []
    for k, h in enumerate(dims):
        d = z.shape[-1]
        if h > d:
            raise ValueError(f"block {k + 1}: hidden dim {h} > input dim {d}")
        W, _ = np.linalg.qr(np.random.RandomState(seed + k).normal(size=(d, h)))
        y = W.T @ z @ W
        lam, U = np.linalg.eigh((y + np.swapaxes(y, 1, 2)) / 2)
        e = recommend_epsilon(lam, percentile)
        prev = floors[-1] if floors and not batch_norm else None
        if prev is not None and e <= prev:
            e = recommend_epsilon2(lam, prev, percentile)
        f = _round_floor(e, prev)
        if prev is not None and f <= prev:
            raise ValueError(f"block {k + 1}: no floor above {prev:g} at the "
                             f"{percentile:g}th percentile: the block would be inert.")
        floors.append(f)
        exact.append(float(e))
        clamped.append(float(np.mean(lam < f)))
        z = (U * np.maximum(lam, f)[:, None, :]) @ np.swapaxes(U, 1, 2)   # ReEig
        if batch_norm and k + 1 < len(dims):
            z = _gah_whiten(z)
    return {"floors": floors, "exact": exact, "clamped": clamped}


PERCENTILES = (1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80)


def floors_file(dataset: str, floors_dir: Optional[str] = None) -> str:
    """Path of the floors of `dataset`: <floors_dir>/<dataset>.json, by default under
    $RFL_RESULTS/reeig_floors (rfl/experiments/grid.py:results_root)."""
    if floors_dir is None:
        from rfl.experiments.grid import results_root
        floors_dir = str(results_root() / "reeig_floors")
    return os.path.join(floors_dir, f"{dataset}.json")


def write_floors(dataset: str, covs: np.ndarray, architectures: List[str], args) -> str:
    """Calibrate the floors of each architecture, with and without batch norm, at each
    percentile of args.percentiles, and write them to floors_file(dataset).

    Layout: {"meta": ..., "architectures": {arch: {"nobn"|"bn": {"<q>": {"floors",
    "exact", "clamped"}}}}}. An architecture or percentile whose floors cannot increase
    without batch norm is left out (the runner then refuses it). Returns the path.
    """
    out: Dict[str, Any] = {}
    for arch in architectures:
        dims = [int(x) for x in arch.split(",")]
        out[arch] = {"nobn": {}, "bn": {}}
        for bn in (False, True):
            for q in args.percentiles:
                try:
                    r = architecture_floors(covs, dims, float(q), bn,
                                            trace_normalize=args.trace_normalize)
                except ValueError as e:
                    print(f"  {arch} q={q:g} {'bn' if bn else 'nobn'}: {e}")
                    continue
                out[arch]["bn" if bn else "nobn"][f"{float(q):g}"] = r
            r = out[arch]["bn" if bn else "nobn"].get(f"{float(args.epsilon_clamp_percentile):g}")
            if r:
                print(f"  {arch:>10} {'with' if bn else 'without'} batch norm, "
                      f"q={args.epsilon_clamp_percentile:g}: --hidden-dims {arch} "
                      f"--epsilons {','.join(f'{f:g}' for f in r['floors'])}   (clamps "
                      + ", ".join(f"{100 * c:.0f}%" for c in r["clamped"]) + ")")
    path = floors_file(dataset, args.floors_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {"dataset": dataset, "percentiles": [float(q) for q in args.percentiles],
            "frame": "spectrum entering each ReEig, random Stiefel BiMaps (seed 0)",
            "trace_normalize": args.trace_normalize, "max_subjects": args.max_subjects,
            "max_samples": args.max_samples, "task": args.task,
            "estimator": args.estimator, "resample": args.resample,
            "generated": datetime.now().isoformat()}
    with open(path, "w") as f:
        json.dump({"meta": meta, "architectures": out}, f, indent=2, cls=NumpyEncoder)
    print(f"  floors written: {path}")
    return path


def floors_for(dataset: str, arch: str, batch_norm: bool, percentile: float,
               path: Optional[str] = None) -> str:
    """The floors "e1,e2,..." of an architecture at a percentile, from the floors file.

    Raises:
        FileNotFoundError: without the floors file.
        KeyError: when the file has no floors for this architecture and percentile.
    """
    path = path or floors_file(dataset)
    arch = ",".join(str(int(x)) for x in str(arch).split(","))
    q = f"{float(percentile):g}"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found: run python -m rfl.experiments.analyze_covariances "
            f"--datasets {dataset} --architectures {arch}")
    with open(path) as f:
        data = json.load(f)
    try:
        floors = data["architectures"][arch]["bn" if batch_norm else "nobn"][q]["floors"]
    except KeyError:
        raise KeyError(f"{path}: no floors for --hidden-dims {arch} "
                       f"{'with' if batch_norm else 'without'} batch norm at the {q}th "
                       "percentile (rerun analyze_covariances with these --architectures "
                       "and --percentiles)") from None
    return ",".join(f"{x:g}" for x in floors)


def recommend_epsilon2(post_eigs: np.ndarray, eps1: float, percentile: float) -> float:
    """Second ReEig floor: a percentile of the eigenvalues strictly above eps_1.

    Taking the percentile of the unconditioned spectrum would return eps_1 itself as
    soon as eps_1 clamps more than `percentile` percent, which makes the second ReEig
    the identity.

    Raises:
        ValueError: if every eigenvalue sits at the eps_1 floor.
    """
    free = post_eigs.reshape(-1)
    free = free[free > eps1]
    if free.size == 0:
        raise ValueError(
            f"every eigenvalue sits at the eps_1 floor {eps1:g}: the first ReEig "
            "saturates the spectrum, and a second block cannot add anything.")
    return float(np.percentile(free, percentile))


# --------------------------------------------------------------------------- #
#  Loading (needs MOABB)
# --------------------------------------------------------------------------- #
def load_covariances(
    dataset_name: str, subjects: List[int], task: str, estimator: str,
    shrinkage: float, regularization: float, resample: Optional[float],
    max_samples: Optional[int], seed: int = 0,
) -> Dict[str, Any]:
    """Estimate trial covariances as the training pipeline does (8-32 Hz band-pass).

    The subjects are loaded one at a time, so the memory is that of one subject. With
    `max_samples`, each subject keeps a random share of its trials, so that every
    subject is represented and the total is at most `max_samples`.

    Returns:
        Dict with 'covs' (N, C, C) float64, 'n_channels', 'n_timepoints' and 'n_classes'.
    """
    from rfl.data.data_utils import load_moabb_dataset, filter_binary_task, resolve_resample
    from rfl.data.datasets import build_dataset

    target = resolve_resample(resample, get_sampling_rate(dataset_name))
    rng = np.random.RandomState(seed)
    per = None if max_samples is None else max(1, max_samples // len(subjects))
    covs, labels, shape = [], set(), None
    for s in subjects:
        X, y, meta = load_moabb_dataset(dataset_name, [s], resample=target,
                                        fmin=8.0, fmax=32.0, verbose=False)
        X, y, meta = filter_binary_task(X=X, y=y, meta=meta, task=task,
                                        dataset_name=dataset_name, verbose=False)
        if per is not None and len(X) > per:
            keep = rng.choice(len(X), per, replace=False)
            X, y = X[keep], y[keep]
        ds = build_dataset(X, y, "covariance", estimator=estimator, shrinkage=shrinkage,
                           regularization=regularization)
        covs.append(np.asarray(ds.covs, dtype=np.float64))
        labels.update(np.unique(y).tolist())
        shape = X.shape
    covs = np.concatenate(covs)
    return {"covs": covs, "n_channels": int(shape[1]),
            "n_timepoints": int(shape[2]), "n_classes": len(labels)}


def analyze_one(dataset_name: str, args) -> Optional[Dict[str, Any]]:
    """Analyse one dataset and print a summary. Returns None if loading fails."""
    subjects = get_subjects(dataset_name)
    if args.max_subjects:                                # 0: every subject
        subjects = subjects[: args.max_subjects]
    print(f"\n{'='*78}\n{dataset_name}  (subjects: {subjects})\n{'='*78}")
    try:
        loaded = load_covariances(
            dataset_name, subjects, args.task, args.estimator, args.shrinkage,
            args.regularization, args.resample, args.max_samples)
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print(f"[skip] {dataset_name}: {e}")
        return None

    covs = loaded["covs"]
    eigs = np.linalg.eigvalsh(covs)                          # (N, C), ascending
    eigs = np.clip(eigs, 0.0, None)                          # round-off negatives to 0
    raw = spectrum_stats(eigs)

    # Calibration frame: the spectrum the model feeds to ReEig. With trace
    # normalisation the eigenvalues of each matrix sum to 1.
    if args.trace_normalize:
        cal = eigs / eigs.sum(axis=1, keepdims=True)
        frame = "trace-normalized"
    else:
        cal = eigs
        frame = "raw (no trace-normalization)"
    cal_stats = spectrum_stats(cal)

    pct = args.epsilon_clamp_percentile
    eps_exact = recommend_epsilon(cal, pct)
    eps = _round_1sig(eps_exact)
    table = epsilon_clamp_table(cal, args.epsilon_candidates)

    block2 = None
    if args.hidden_dim_block1:
        try:
            h1 = args.hidden_dim_block1
            # When the BiMap reduces dimension, Poincare interlacing gives
            # mu_i >= lambda_{i+d-h} for the eigenvalues of W^T X W. The congruence
            # removes the bottom tail, so an eps_1 calibrated on the input spectrum
            # clamps almost nothing. eps_1 is re-calibrated after the congruence.
            pre = post_block_spectrum(covs, h1, 0.0,
                                      trace_normalize=args.trace_normalize)
            frac_input_frame = float(np.mean(pre["eigs_prefloor"] < eps))
            eps1_post_exact = recommend_epsilon(pre["eigs_prefloor"], pct)
            eps1_post = _round_1sig(eps1_post_exact)

            pb = post_block_spectrum(covs, h1, eps1_post,
                                     trace_normalize=args.trace_normalize)
            eps2_exact = recommend_epsilon2(pb["eigs"], eps1_post, pct)
            # Rounding must keep eps_2 > eps_1. Otherwise use three significant figures.
            eps2 = _round_1sig(eps2_exact)
            if eps2 <= eps1_post:
                eps1_post = float(f"{eps1_post_exact:.3g}")
                eps2 = float(f"{eps2_exact:.3g}")
            if eps2 <= eps1_post:
                raise ValueError(
                    f"eps_2 ({eps2_exact:.4g}) is not separated from eps_1 "
                    f"({eps1_post_exact:.4g}) at this percentile: the post-block spectrum "
                    "is too concentrated for a second ReEig to bite. Lower "
                    "--epsilon-clamp-percentile, or take the second block as pointless.")
            block2 = {
                "hidden_dim_block1": h1,
                "frac_clamped_by_input_frame_epsilon1": frac_input_frame,
                "recommended_epsilon1_post": eps1_post,
                "recommended_epsilon1_post_exact": eps1_post_exact,
                "frac_clamped_by_epsilon1": pb["frac_clamped_by_eps1"],
                "post_congruence_spectrum": spectrum_stats(pre["eigs_prefloor"]),
                "post_block_spectrum": spectrum_stats(pb["eigs"]),
                "recommended_epsilon2": eps2,
                "recommended_epsilon2_exact": eps2_exact,
                "epsilon2_clamp_table": epsilon_clamp_table(pb["eigs"],
                                                            args.epsilon_candidates),
            }
        except ValueError as e:
            print(f"  [block2] {e}")

    if args.architectures:
        write_floors(dataset_name, covs, args.architectures, args)

    C, T = loaded["n_channels"], loaded["n_timepoints"]
    res = {
        "dataset": dataset_name, "n_channels": C, "n_timepoints": T,
        "n_classes": loaded["n_classes"], "tc_ratio": float(T / C),
        "sampling_rate": get_sampling_rate(dataset_name),
        "estimator": args.estimator, "regularization": args.regularization,
        "trace_normalize": args.trace_normalize, "calibration_frame": frame,
        "raw_spectrum": raw,
        "calibration_spectrum": cal_stats,
        "epsilon_clamp_percentile": pct,
        "recommended_epsilon": eps,
        "recommended_epsilon_exact": eps_exact,
        "epsilon_clamp_table": table,
        "block2": block2,
    }
    _print_one(res)
    return res


def _print_one(r: Dict[str, Any]) -> None:
    """Print the analysis of one dataset."""
    raw, cal = r["raw_spectrum"], r["calibration_spectrum"]
    print(f"  dim={r['n_channels']}  T/C={r['tc_ratio']:.1f}  SPD={raw['spd_percentage']:.1f}%  "
          f"log10(cond) mean={raw['log10_cond_mean']:.1f} p99={raw['log10_cond_p99']:.1f}")
    print(f"  raw   eig: mean={raw['eig_mean']:.2e} median={raw['eig_median']:.2e} "
          f"max={raw['max_eig_mean']:.2e} min(p1)={raw['min_eig_p1']:.2e}")
    if r["tc_ratio"] < 4:
        print("  [warn] T/C < 4: covariances likely rank-deficient -> consider lwf shrinkage")
    elif r["tc_ratio"] < 10:
        print("  [note] T/C < 10: mild ill-conditioning; lwf shrinkage may help")
    print(f"  calibration frame: {r['calibration_frame']}  "
          f"-> eig mean={cal['eig_mean']:.2e} median={cal['eig_median']:.2e}")
    print(f"  recommended epsilon_1 = {r['recommended_epsilon']:g} "
          f"(exact {r['recommended_epsilon_exact']:.2e}; clamps ~{r['epsilon_clamp_percentile']:g}% "
          f"of eigenvalues, ReEig as rectifier)")
    print(f"  {'epsilon':>10}{'%eig clamped':>16}{'%matrices clamped':>20}")
    for row in r["epsilon_clamp_table"]:
        print(f"  {row['epsilon']:>10.0e}{100*row['frac_eigenvalues_clamped']:>15.2f}%"
              f"{100*row['frac_matrices_clamped']:>19.2f}%")

    b = r.get("block2")
    if b:
        print(f"  -- after the block-1 congruence ({r['n_channels']} -> "
              f"{b['hidden_dim_block1']}, random Stiefel W) --")
        print(f"  the INPUT-frame epsilon_1 = {r['recommended_epsilon']:g} clamps only "
              f"{100*b['frac_clamped_by_input_frame_epsilon1']:.2f}% here: the congruence "
              "already removed the bottom tail (Poincare interlacing), so a threshold "
              "calibrated before it is largely inert.")
        print(f"  RE-calibrated epsilon_1 = {b['recommended_epsilon1_post']:g} "
              f"(exact {b['recommended_epsilon1_post_exact']:.2e}) -> clamps "
              f"{100*b['frac_clamped_by_epsilon1']:.2f}%")
        print(f"  recommended epsilon_2 = {b['recommended_epsilon2']:g} "
              f"(exact {b['recommended_epsilon2_exact']:.2e}; "
              f"{r['epsilon_clamp_percentile']:g}th percentile of the eigenvalues left "
              "FREE by epsilon_1)")
        print(f"  --hidden-dims {b['hidden_dim_block1']},<h2>  "
              f"--epsilons {b['recommended_epsilon1_post']:g},{b['recommended_epsilon2']:g}")


def write_outputs(results: List[Dict[str, Any]], output_dir: str, meta: Dict[str, Any]) -> None:
    """Write covariance_analysis.json and a text summary table to `output_dir`."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "covariance_analysis.json"), "w") as f:
        json.dump({"meta": meta, "datasets": results}, f, indent=2, cls=NumpyEncoder)

    lines = ["COVARIANCE ANALYSIS: ReEig calibration",
             f"generated {datetime.now().isoformat()}",
             f"frame: {'trace-normalized' if meta.get('trace_normalize') else 'raw'};  "
             f"epsilon clamps ~{meta.get('epsilon_clamp_percentile')}% of eigenvalues "
             "(ReEig as rectifier)",
             "=" * 104,
             f"{'dataset':<20}{'dim':>5}{'T/C':>7}{'SPD%':>7}{'log10cond':>11}"
             f"{'eps_in':>10}{'%clamp_in':>11}{'eps_1':>11}{'eps_2':>11}{'%clamp1':>9}",
             "-" * 104]
    for r in results:
        b = r.get("block2") or {}
        e1 = f"{b['recommended_epsilon1_post']:g}" if b else "-"
        e2 = f"{b['recommended_epsilon2']:g}" if b else "-"
        cin = f"{100*b['frac_clamped_by_input_frame_epsilon1']:.1f}" if b else "-"
        c1 = f"{100*b['frac_clamped_by_epsilon1']:.1f}" if b else "-"
        lines.append(
            f"{r['dataset']:<20}{r['n_channels']:>5}{r['tc_ratio']:>7.1f}"
            f"{r['raw_spectrum']['spd_percentage']:>7.1f}"
            f"{r['raw_spectrum']['log10_cond_mean']:>11.1f}"
            f"{r['recommended_epsilon']:>10g}{cin:>11}{e1:>11}{e2:>11}{c1:>9}")
    lines += ["=" * 104,
              "Apply: --epsilons eps_1[,eps_2] together with --hidden-dims.",
              "eps_2 must strictly exceed eps_1. Otherwise the second ReEig is the",
              "identity and the two blocks collapse into one.",
              "eps_in: epsilon calibrated on the input spectrum. %clamp_in: what it clamps",
              "after the BiMap congruence (near zero when the BiMap reduces dimension).",
              "eps_1: the same percentile after the congruence. %clamp1: what eps_1 clamps."]
    report = "\n".join(lines)
    with open(os.path.join(output_dir, "covariance_analysis.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\nWritten: {output_dir}/covariance_analysis.json and .txt")


def main():
    from rfl.utils.logs import setup_logging
    setup_logging()
    p = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Covariance spectrum analysis -> ReEig epsilon calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets", type=str, nargs="+", default=sorted(DATASET_CONFIGS),
                   choices=sorted(DATASET_CONFIGS), help="Datasets to analyze.")
    p.add_argument("--max-subjects", type=int, default=5,
                   help="Use the first N subjects of each dataset (0: all). 5 is the "
                        "value the calibration of the paper's floors ran with.")
    p.add_argument("--max-samples", type=int, default=2000,
                   help="Cap on the covariance matrices analysed per dataset, shared "
                        "equally between the subjects (random subsample).")
    p.add_argument("--task", type=str, default="all_classes",
                   choices=["binary", "all_classes", "no_rest"])
    p.add_argument("--estimator", type=str, default="scm", choices=["scm", "lwf"])
    p.add_argument("--shrinkage", type=float, default=0.0, help="lwf shrinkage intensity.")
    p.add_argument("--regularization", type=float, default=0.0,
                   help="Diagonal floor added before analysis (training default is 0.0).")
    p.add_argument("--resample", type=float, default=128.0,
                   help="Target sampling rate in Hz. 0 keeps the native rate. Never "
                        "upsamples.")
    tn = p.add_mutually_exclusive_group()
    tn.add_argument("--trace-normalize", dest="trace_normalize", action="store_true",
                    help="Calibrate on the trace-normalized spectrum (matches SPDNet).")
    tn.add_argument("--no-trace-normalize", dest="trace_normalize", action="store_false",
                    help="Calibrate on the raw (un-normalized) spectrum instead.")
    p.set_defaults(trace_normalize=True)
    p.add_argument("--epsilon-clamp-percentile", type=float, default=15.0,
                   help="Target ReEig clamp percentage: epsilon is this percentile of "
                        "the calibration spectrum.")
    p.add_argument("--hidden-dim-block1", type=int, default=None,
                   help="Also calibrate a two-block SPDNet whose first BiMap maps d to "
                        "this dimension, and report eps_1 and eps_2.")
    p.add_argument("--architectures", type=str, nargs="+", default=None,
                   help="Calibrate every ReEig floor of these architectures (block "
                        "dimensions, e.g. 16 or 64,32), with and without batch norm, and "
                        "write them for the plans (see --floors-dir).")
    p.add_argument("--percentiles", type=float, nargs="+", default=list(PERCENTILES),
                   help="Percentiles at which --architectures computes the floors.")
    p.add_argument("--floors-dir", type=str, default=None,
                   help="Folder of the floors files (default $RFL_RESULTS/reeig_floors).")
    p.add_argument("--epsilon-candidates", type=float, nargs="+",
                   default=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   help="Epsilon values for the clamp-fraction table.")
    p.add_argument("--output-dir", type=str, default="results/covariance_analysis")
    args = p.parse_args()

    print(f"Analyzing {len(args.datasets)} dataset(s): {args.datasets}  "
          f"(frame: {'trace-normalized' if args.trace_normalize else 'raw'}, "
          f"epsilon clamp ~{args.epsilon_clamp_percentile:g}%"
          + (f", block-1 hidden dim {args.hidden_dim_block1}"
             if args.hidden_dim_block1 else "") + ")")
    results = [r for r in (analyze_one(d, args) for d in args.datasets) if r is not None]
    if not results:
        print("No datasets analyzed successfully.")
        return
    meta = {"task": args.task, "estimator": args.estimator, "shrinkage": args.shrinkage,
            "regularization": args.regularization, "resample": args.resample,
            "max_subjects": args.max_subjects, "max_samples": args.max_samples,
            "trace_normalize": args.trace_normalize,
            "hidden_dim_block1": args.hidden_dim_block1,
            "epsilon_clamp_percentile": args.epsilon_clamp_percentile}
    write_outputs(results, args.output_dir, meta)


if __name__ == "__main__":
    main()
