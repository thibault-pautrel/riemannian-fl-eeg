"""PyTorch datasets for raw EEG, covariance and precomputed SPD inputs.

Every pipeline (centralised or federated, real or synthetic) builds its datasets with
`build_dataset(X, y, kind)`::

    kind = 'raw'         RawEEGDataset      EEGNet, input (N, C, T)
    kind = 'covariance'  CovarianceDataset  SPDNet, covariance estimated from (N, C, T)
    kind = 'spd'         SPDMatrixDataset   SPDNet, precomputed (N, d, d), unchanged

All preprocessing is per trial: no statistic is shared between trials or splits.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "RawEEGDataset",
    "CovarianceDataset",
    "SPDMatrixDataset",
    "build_dataset",
    "make_loader",
]


class RawEEGDataset(Dataset):
    """Raw EEG trials for EEGNet, z-scored at construction.

    Items are (x, y) with x of shape (1, C, T).

    Args:
        X: (N, C, T) signals.
        y: (N,) integer labels.
        normalize: Apply z-scoring.
        normalization_type: 'per_trial' (each trial and channel over time, the default
            of the papers) or 'per_trial_global' (each trial over all its channels and
            time: one mean and one scale per trial, so the relative power of the
            channels, the main motor-imagery feature, is kept, as SPDNet's trace
            normalisation keeps it). Neither shares a statistic between trials.

    Raises:
        ValueError: If X is not 3-D or the normalisation type is unknown.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 normalize: bool = True, normalization_type: str = "per_trial"):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"RawEEGDataset expects (N, C, T), got {X.shape}")
        if normalize:
            if normalization_type in ("per_trial", "per_trial_global"):
                axes = (-1,) if normalization_type == "per_trial" else (-2, -1)
                mean = X.mean(axis=axes, keepdims=True)
                std = np.maximum(X.std(axis=axes, keepdims=True), 1e-8)
                X = ((X - mean) / std).astype(np.float32)
            else:
                raise ValueError(f"Unknown normalization_type: {normalization_type!r}")
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float().unsqueeze(0)  # (1, C, T)
        return x, torch.tensor(self.y[idx], dtype=torch.long)


class CovarianceDataset(Dataset):
    """Per-trial spatial covariance matrices computed from raw EEG, for SPDNet.

    With centred trials X_c, the sample covariance is S = X_c X_c^T / (T - 1).
    'scm' returns S + reg I. 'lwf' returns (1 - a) S + a (tr(S) / C) I + reg I with the
    fixed intensity a = `shrinkage` (the intensity is not estimated from the data).

    Args:
        X: (N, C, T) signals.
        y: (N,) labels.
        estimator: 'scm' or 'lwf'.
        shrinkage: Shrinkage intensity a, used by 'lwf'.
        regularization: Diagonal loading reg, for strict positive definiteness.

    Raises:
        ValueError: If X is not 3-D or the estimator is unknown.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 estimator: str = "scm", shrinkage: float = 0.0,
                 regularization: float = 1e-6):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"CovarianceDataset expects raw (N, C, T), got {X.shape}. "
                             "For precomputed SPD matrices use SPDMatrixDataset / kind='spd'.")
        self.covs = self._compute(X, estimator, shrinkage, regularization)
        self.y = np.asarray(y, dtype=np.int64)

    @staticmethod
    def _compute(X, estimator, shrinkage, reg):
        """(N, C, C) float32 covariances (see the class docstring)."""
        N, C, _T = X.shape
        T = X.shape[2]
        covs = np.empty((N, C, C), dtype=np.float32)
        eye = np.eye(C, dtype=np.float32)
        for i in range(N):
            tr = X[i] - X[i].mean(axis=1, keepdims=True)
            cov = (tr @ tr.T) / (T - 1)
            cov = 0.5 * (cov + cov.T)  # exact symmetry
            if estimator == "scm":
                covs[i] = cov + reg * eye
            elif estimator == "lwf":
                shrunk = (1 - shrinkage) * cov + shrinkage * (np.trace(cov) / C) * eye
                covs[i] = shrunk + reg * eye
            else:
                raise ValueError(f"Unknown estimator: {estimator!r}. Use 'scm' or 'lwf'.")
        return covs

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.covs[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)


class SPDMatrixDataset(Dataset):
    """Precomputed SPD matrices (for example synthetic Wishart data), passed through.

    Use this rather than CovarianceDataset for SPD input, which would otherwise be
    treated as a raw signal.

    Args:
        X: (N, d, d) SPD matrices.
        y: (N,) labels.
        regularization: Optional diagonal loading. The synthetic generator already
            adds its own.

    Raises:
        ValueError: If X is not of shape (N, d, d).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 regularization: float = 0.0):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3 or X.shape[1] != X.shape[2]:
            raise ValueError(f"SPDMatrixDataset expects (N, d, d) SPD input, got {X.shape}")
        if regularization and regularization > 0.0:
            X = X + np.float32(regularization) * np.eye(X.shape[1], dtype=np.float32)[None]
        self.covs = X
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.covs[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)


def build_dataset(
    X: np.ndarray,
    y: np.ndarray,
    kind: str,
    *,
    normalization_type: str = "per_trial",
    estimator: str = "scm",
    shrinkage: float = 0.0,
    regularization: Optional[float] = None,
) -> Dataset:
    """Build the dataset for `kind` ('raw', 'covariance' or 'spd').

    Args:
        X: Raw (N, C, T) signals, or (N, d, d) SPD matrices for 'spd'.
        y: (N,) labels.
        kind: 'raw', 'covariance' or 'spd' ('spd_precomputed' is an alias).
        normalization_type: Passed to RawEEGDataset.
        estimator: Passed to CovarianceDataset.
        shrinkage: Passed to CovarianceDataset.
        regularization: Diagonal loading. None means 1e-6 for 'covariance' and 0.0
            for 'spd'. Ignored for 'raw'.

    Returns:
        A torch Dataset yielding (input, label) pairs.

    Raises:
        ValueError: On an unknown kind.
    """
    kind = kind.lower()
    if kind == "raw":
        return RawEEGDataset(X, y, normalize=True, normalization_type=normalization_type)
    if kind == "covariance":
        reg = 1e-6 if regularization is None else regularization
        return CovarianceDataset(X, y, estimator=estimator, shrinkage=shrinkage,
                                 regularization=reg)
    if kind in ("spd", "spd_precomputed"):
        reg = 0.0 if regularization is None else regularization
        return SPDMatrixDataset(X, y, regularization=reg)
    raise ValueError(f"Unknown data kind {kind!r}. Use 'raw', 'covariance', or 'spd'.")


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool,
                num_workers: int = 0, avoid_single_batch: bool = False) -> DataLoader:
    """DataLoader with `batch_size` when shuffling, and twice that for evaluation.

    With `avoid_single_batch` (training with SPD batch norm), a last batch of a single
    record is dropped: batch statistics are undefined on one matrix.
    """
    bs = batch_size if shuffle else batch_size * 2
    drop_last = bool(avoid_single_batch and shuffle and len(dataset) % bs == 1
                     and len(dataset) > 1)
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


