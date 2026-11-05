"""Synthetic SPD data for SPDNet experiments (Wishart samples, per-client splits).

Class model. All classes share an AR(1) Toeplitz base B_ij = rho^|i - j| (SPD for
|rho| < 1). Class k adds a low-rank term in its own orthonormal subspace V_k of shape
(d, signal_rank)::

    Sigma_k = B + signal_strength * V_k V_k^T.

The class signal lives in a low-dimensional subspace under the shared base and the
within-class noise, so a learned projection (the BiMap layer) is useful. Samples are
S ~ Wishart(Sigma_k, df) / df with df = max(d, round(df_factor * d)). Separability
increases with `signal_strength` and with `df_factor`.

Client heterogeneity, two independent mechanisms:

* label skew: each client draws its class proportions from Dirichlet(label_skew_alpha)
  (smaller alpha means more skew, None means balanced),
* channel gain: each client applies a fixed congruence S -> D S D with
  D = diag(exp(N(0, channel_gain_std))). It changes the spectrum, as per-subject channel
  gains do in EEG (0 disables it).

`mode='noniid'` (default) sets label_skew_alpha=0.5 and channel_gain_std=0.1 unless they
are given. `mode='iid'` disables both.

Two more mechanisms make the clients differ as EEG subjects do (both off by default):

* pattern rotation: client i rotates the class subspaces, V_k -> R_i V_k with
  R_i = expm(pattern_rotation * (A - A^T) / sqrt(2)), A with i.i.d. N(0, 1/d) entries.
  The discriminant directions then differ between subjects, and re-centring each
  client on its mean no longer aligns them. Around 0.45, a model per client beats a
  model on pooled re-centred clients with many trials, and loses with few trials.
* signal range: client i draws its signal_strength uniformly in signal_strength_range,
  so some subjects are easier to decode than others.

These two draw from a generator of their own: the label counts, gains and samples are
those of the same seed without them. `class_scale_matrices` then holds the matrices
before rotation, at the base signal_strength.

Output format (read by `load_clients` and `data.datasets.build_dataset(kind='spd')`)::

    {
      'data_kind': 'spd',
      'clients': [{X_train, y_train, X_val, y_val, X_test, y_test}, ...],
      'class_scale_matrices': [Sigma_0, ..., Sigma_{K-1}],
      'metadata': {...},
    }

X arrays are float32 of shape (N, d, d). Run the module as a script to write a .npz file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "make_class_scale_matrices",
    "sample_wishart_spd",
    "generate_synthetic_spd",
    "save_dataset",
    "load_dataset",
    "load_clients",
    "pool_clients",
]


def make_class_scale_matrices(
    n_channels: int,
    n_classes: int,
    rho: float = 0.6,
    signal_strength: float = 0.4,
    signal_rank: int = 1,
    seed: int = 0,
) -> List[np.ndarray]:
    """Class scale matrices Sigma_k = B + signal_strength V_k V_k^T (module docstring).

    The subspaces V_k are disjoint column blocks of one random orthonormal basis.

    Args:
        n_channels: SPD dimension d.
        n_classes: Number of classes K.
        rho: Base correlation in (-1, 1).
        signal_strength: Weight of the class-specific low-rank term.
        signal_rank: Rank of each class term.
        seed: Seed of the orthonormal basis.

    Returns:
        List of K float64 SPD matrices of shape (d, d).

    Raises:
        ValueError: If |rho| >= 1, or signal_rank * n_classes > d.
    """
    d = n_channels
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    if signal_rank < 1 or signal_rank * n_classes > d:
        raise ValueError(
            f"signal_rank={signal_rank} * n_classes={n_classes} must be <= d={d} "
            "so that class subspaces fit (and are disjoint)")

    idx = np.arange(d)
    base = (rho ** np.abs(idx[:, None] - idx[None, :])).astype(np.float64)

    # One fixed orthonormal basis. Class k uses its own block of `signal_rank` columns.
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(d, d))

    scales: List[np.ndarray] = []
    for k in range(n_classes):
        cols = [k * signal_rank + r for r in range(signal_rank)]
        V = Q[:, cols]                                   # (d, signal_rank)
        scales.append(base + signal_strength * (V @ V.T))
    return scales


def sample_wishart_spd(
    scale_matrix: np.ndarray,
    df: int,
    n_samples: int,
    rng: np.random.RandomState,
    regularization: float = 1e-6,
) -> np.ndarray:
    """Draw `n_samples` matrices Wishart(scale_matrix, df) / df + regularization I.

    Args:
        scale_matrix: (d, d) SPD scale matrix (the mean of the draws).
        df: Degrees of freedom, at least d.
        n_samples: Number of draws.
        rng: NumPy random state (consumed).
        regularization: Diagonal loading added to each draw.

    Returns:
        (n_samples, d, d) float32 array.

    Raises:
        ValueError: If df < d.
    """
    d = scale_matrix.shape[0]
    if df < d:
        raise ValueError(f"df={df} must be >= d={d} for a non-degenerate Wishart")
    if n_samples == 0:
        return np.empty((0, d, d), dtype=np.float32)

    L = np.linalg.cholesky(scale_matrix)
    out = np.empty((n_samples, d, d), dtype=np.float32)
    eye = np.eye(d, dtype=np.float64)
    for i in range(n_samples):
        Z = rng.randn(d, df)
        X = L @ Z
        S = (X @ X.T) / df + regularization * eye
        out[i] = S.astype(np.float32)
    return out


def _client_class_counts(
    per_class: int, n_classes: int, alpha: Optional[float], rng: np.random.RandomState
) -> np.ndarray:
    """Per-class counts of one client, summing to per_class * n_classes.

    Balanced when alpha is None, otherwise proportional to a Dirichlet(alpha) draw.
    """
    total = per_class * n_classes
    if alpha is None:
        return np.full(n_classes, per_class, dtype=int)
    p = rng.dirichlet([alpha] * n_classes)
    counts = np.floor(p * total).astype(int)
    # Give the rounding remainder to the most probable classes.
    remainder = total - int(counts.sum())
    for j in np.argsort(-p)[:max(remainder, 0)]:
        counts[j] += 1
    return counts


def _channel_gain(d: int, gain_std: float, rng: np.random.RandomState) -> Optional[np.ndarray]:
    """Diagonal of D = diag(exp(N(0, gain_std))), or None when gain_std <= 0."""
    if gain_std <= 0.0:
        return None
    return np.exp(rng.normal(0.0, gain_std, size=d)).astype(np.float64)


def _apply_gain(spd: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Diagonal congruence S -> D S D for a batch (n, d, d)."""
    out = spd.astype(np.float64) * D[None, :, None] * D[None, None, :]
    return out.astype(np.float32)


def generate_synthetic_spd(
    n_clients: int,
    samples_per_client: int,
    n_channels: int,
    n_classes: int,
    rho: float = 0.6,
    signal_strength: float = 0.4,
    signal_rank: int = 1,
    df_factor: float = 1.5,
    mode: str = "noniid",
    label_skew_alpha: Optional[float] = None,
    channel_gain_std: Optional[float] = None,
    regularization: float = 1e-6,
    val_ratio: float = 0.10,
    test_ratio: float = 0.15,
    seed: int = 42,
    pattern_rotation: float = 0.0,
    signal_strength_range: Optional[tuple] = None,
) -> Dict:
    """Generate a per-client synthetic SPD dataset.

    Each client gets about samples_per_client / (1 - val_ratio - test_ratio) samples,
    shuffled, then split in order into train, validation and test.

    Args:
        n_clients: Number of clients (1 for a centralised pool).
        samples_per_client: Approximate number of training samples per client.
        n_channels: SPD dimension d.
        n_classes: Number of classes K.
        rho: Base correlation (see `make_class_scale_matrices`).
        signal_strength: Weight of the class term.
        signal_rank: Rank of the class term.
        df_factor: Wishart df = max(d, round(df_factor * d)). Lower is harder.
        mode: 'noniid' or 'iid' (presets for the two heterogeneity knobs).
        label_skew_alpha: Dirichlet concentration. Overrides the preset.
        channel_gain_std: Channel-gain strength. Overrides the preset.
        regularization: Diagonal loading of each sample.
        val_ratio: Validation fraction of each client.
        test_ratio: Test fraction of each client.
        seed: Seed of every random draw.

    Returns:
        Dataset dict (see the module docstring).

    Raises:
        ValueError: On an unknown mode, or invalid class-model parameters.
    """
    if mode not in ("iid", "noniid"):
        raise ValueError(f"mode must be 'iid' or 'noniid', got {mode!r}")

    if label_skew_alpha is None:
        label_skew_alpha = 0.5 if mode == "noniid" else None
    if channel_gain_std is None:
        channel_gain_std = 0.1 if mode == "noniid" else 0.0

    rng = np.random.RandomState(seed)
    df = max(n_channels, int(round(df_factor * n_channels)))

    total_per_client = int(round(samples_per_client / (1.0 - val_ratio - test_ratio)))
    per_class = max(1, total_per_client // n_classes)

    scale_matrices = make_class_scale_matrices(
        n_channels, n_classes, rho=rho,
        signal_strength=signal_strength, signal_rank=signal_rank, seed=seed)

    per_subject = pattern_rotation > 0 or signal_strength_range is not None
    if signal_strength_range is not None:
        lo, hi = (float(v) for v in signal_strength_range)
        if not 0 <= lo <= hi:
            raise ValueError("signal_strength_range must be (a, b) with 0 <= a <= b, got "
                             f"{signal_strength_range}")
    if pattern_rotation < 0:
        raise ValueError(f"pattern_rotation must be >= 0, got {pattern_rotation}")
    if per_subject:
        # A generator of its own, so that the other draws (label counts, gains, samples)
        # are those of the same seed without these options.
        subject_rng = np.random.RandomState([int(seed), 4049])
        unit = make_class_scale_matrices(n_channels, n_classes, rho=rho, signal_strength=1.0,
                                         signal_rank=signal_rank, seed=seed)
        idx = np.arange(n_channels)
        base = (rho ** np.abs(idx[:, None] - idx[None, :])).astype(np.float64)
        patterns = [u - base for u in unit]            # V_k V_k^T

    clients: List[Dict[str, np.ndarray]] = []
    for _c in range(n_clients):
        counts = _client_class_counts(per_class, n_classes, label_skew_alpha, rng)
        gain = _channel_gain(n_channels, channel_gain_std, rng)
        scales = scale_matrices
        if per_subject:
            from scipy.linalg import expm
            strength = (subject_rng.uniform(*signal_strength_range)
                        if signal_strength_range is not None else signal_strength)
            R = np.eye(n_channels)
            if pattern_rotation > 0:
                A = subject_rng.randn(n_channels, n_channels) / np.sqrt(n_channels)
                R = expm(pattern_rotation * (A - A.T) / np.sqrt(2))
            scales = [base + strength * (R @ P @ R.T) for P in patterns]

        X_parts, y_parts = [], []
        for k in range(n_classes):
            if counts[k] == 0:
                continue
            samples = sample_wishart_spd(scales[k], df, int(counts[k]), rng, regularization)
            if gain is not None:
                samples = _apply_gain(samples, gain)
            X_parts.append(samples)
            y_parts.append(np.full(int(counts[k]), k, dtype=np.int64))

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]

        n_total = len(y)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val - n_test

        clients.append({
            "X_train": X[:n_train],                    "y_train": y[:n_train],
            "X_val":   X[n_train:n_train + n_val],      "y_val":   y[n_train:n_train + n_val],
            "X_test":  X[n_train + n_val:],             "y_test":  y[n_train + n_val:],
        })

    metadata = {
        "data_kind": "spd",
        "n_clients": n_clients,
        "n_channels": n_channels,
        "n_classes": n_classes,
        "samples_per_client_train": int(clients[0]["X_train"].shape[0]),
        "df": df,
        "rho": rho,
        "signal_strength": (signal_strength if signal_strength_range is None else None),
        "signal_rank": signal_rank,
        "mode": mode,
        "label_skew_alpha": label_skew_alpha,
        "channel_gain_std": channel_gain_std,
        "regularization": regularization,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "pattern_rotation": pattern_rotation,
        "signal_strength_range": (list(signal_strength_range)
                                  if signal_strength_range is not None else None),
    }

    return {
        "data_kind": "spd",
        "clients": clients,
        "class_scale_matrices": scale_matrices,
        "metadata": metadata,
    }


def save_dataset(dataset: Dict, output_path: str) -> None:
    """Save a dataset dict to a compressed .npz (splits, scale matrices, metadata JSON)."""
    save_dict: Dict[str, np.ndarray] = {}
    for i, cd in enumerate(dataset["clients"]):
        for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
            save_dict[f"client_{i}_{key}"] = cd[key]
    for k, Sigma in enumerate(dataset["class_scale_matrices"]):
        save_dict[f"scale_matrix_{k}"] = Sigma
    save_dict["metadata_json"] = np.array(json.dumps(dataset["metadata"]))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(output_path, **save_dict)


def load_dataset(path: str) -> Dict:
    """Load a dataset previously written by `save_dataset`."""
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata_json"]))
    n_clients = metadata["n_clients"]
    n_classes = metadata["n_classes"]

    clients = [{
        "X_train": data[f"client_{i}_X_train"], "y_train": data[f"client_{i}_y_train"],
        "X_val":   data[f"client_{i}_X_val"],   "y_val":   data[f"client_{i}_y_val"],
        "X_test":  data[f"client_{i}_X_test"],  "y_test":  data[f"client_{i}_y_test"],
    } for i in range(n_clients)]

    scale_matrices = [data[f"scale_matrix_{k}"] for k in range(n_classes)]
    return {
        "data_kind": "spd",
        "clients": clients,
        "class_scale_matrices": scale_matrices,
        "metadata": metadata,
    }


def load_clients(source) -> List[Dict[str, np.ndarray]]:
    """Return the per-client split list from a path or an in-memory dataset dict."""
    if isinstance(source, str):
        source = load_dataset(source)
    return source["clients"]


def pool_clients(dataset: Dict):
    """Concatenate all clients into {'train': (X, y), 'val': (X, y), 'test': (X, y)}."""
    clients = dataset["clients"]
    out = {}
    for split in ("train", "val", "test"):
        X = np.concatenate([c[f"X_{split}"] for c in clients], axis=0)
        y = np.concatenate([c[f"y_{split}"] for c in clients], axis=0)
        out[split] = (X, y)
    return out


def main():
    """Command-line entry point: generate a dataset and save it as .npz."""
    from rfl.utils.logs import setup_logging
    setup_logging()
    p = argparse.ArgumentParser(allow_abbrev=False, description="Generate synthetic Wishart-SPD data for SPDNet")
    p.add_argument("--n-clients", type=int, default=10)
    p.add_argument("--samples-per-client", type=int, default=400)
    p.add_argument("--n-channels", type=int, default=22)
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument("--rho", type=float, default=0.6)
    p.add_argument("--signal-strength", type=float, default=0.4)
    p.add_argument("--signal-rank", type=int, default=1)
    p.add_argument("--df-factor", type=float, default=1.5)
    p.add_argument("--mode", choices=["iid", "noniid"], default="noniid")
    p.add_argument("--label-skew-alpha", type=float, default=None,
                   help="Dirichlet concentration for label skew (overrides mode preset).")
    p.add_argument("--channel-gain-std", type=float, default=None,
                   help="Per-client non-isometric diagonal congruence std (overrides preset).")
    p.add_argument("--regularization", type=float, default=1e-6)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pattern-rotation", type=float, default=0.0,
                   help="Rotation of the class subspaces per client (0: none). See the "
                        "module docstring.")
    p.add_argument("--signal-strength-range", type=str, default=None,
                   help="'a,b': per-client signal strength drawn uniformly in [a, b] "
                        "(replaces --signal-strength).")
    p.add_argument("--output", type=str, default="synthetic_spd.npz")
    args = p.parse_args()
    rng_range = (tuple(float(v) for v in args.signal_strength_range.split(","))
                 if args.signal_strength_range else None)

    ds = generate_synthetic_spd(
        n_clients=args.n_clients, samples_per_client=args.samples_per_client,
        n_channels=args.n_channels, n_classes=args.n_classes,
        rho=args.rho, signal_strength=args.signal_strength, signal_rank=args.signal_rank,
        df_factor=args.df_factor, mode=args.mode,
        label_skew_alpha=args.label_skew_alpha, channel_gain_std=args.channel_gain_std,
        regularization=args.regularization, val_ratio=args.val_ratio,
        test_ratio=args.test_ratio, seed=args.seed,
        pattern_rotation=args.pattern_rotation, signal_strength_range=rng_range,
    )
    save_dataset(ds, args.output)

    m = ds["metadata"]
    cd0 = ds["clients"][0]
    ev = np.linalg.eigvalsh(cd0["X_train"][0])
    logger.info(f"Saved {args.output}")
    logger.info(f"  clients={args.n_clients}  train/client={m['samples_per_client_train']}"
          f"  d={args.n_channels}  K={args.n_classes}  df={m['df']}  mode={args.mode}")
    strength = (m['signal_strength'] if m['signal_strength'] is not None
                else f"uniform in {m['signal_strength_range']}")
    logger.info(f"  rho={m['rho']}  signal_strength={strength}  signal_rank={m['signal_rank']}"
                f"  pattern_rotation={m['pattern_rotation']}")
    logger.info(f"  label_skew_alpha={m['label_skew_alpha']}  channel_gain_std={m['channel_gain_std']}")
    logger.info(f"  sample-0 eigenvalues: min={ev.min():.4f} max={ev.max():.4f}  SPD={bool((ev > 0).all())}")


if __name__ == "__main__":
    main()