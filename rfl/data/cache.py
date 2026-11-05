"""Cache of the preprocessed trials of each subject of a MOABB dataset.

`load_moabb_dataset` reads the raw files of a subject, band-pass filters, resamples and
cuts the trials. The result depends only on the dataset, the subject and these
settings, so it is written once per subject to

    <MNE_DATA>/rfl-cache/<dataset>/<key>/subject_<id>.npz

and read back by every later load. The key is a hash of everything the trials depend
on: the classes, the band, the sampling rate, the window, the dataset options, the
versions of MOABB and MNE and the format of the file (`params.json`, next to the files,
lists them). A change of any of them gives another key, so a stale file is never read.

The arrays are stored exactly (float64, uncompressed): a load from the cache returns
the arrays a load from the raw files returns. Files are written to a temporary name
and renamed, so parallel jobs never read a partial file. `condor/fetch_data.sh` fills
the cache on the user interface, and the jobs only read it.

RFL_DATA_CACHE=0 turns the cache off, RFL_DATA_CACHE=refresh recomputes and rewrites the
files of the subjects that are loaded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_FORMAT = 1


def cache_mode() -> str:
    """'on' (read, write when missing), 'refresh' (always rewrite) or 'off'."""
    v = os.environ.get("RFL_DATA_CACHE", "on").strip().lower()
    if v in ("0", "off", "no", "false"):
        return "off"
    return "refresh" if v == "refresh" else "on"


def cache_root() -> Path:
    """The cache folder, inside the MNE data folder the runs use."""
    from rfl.data.registry import mne_data_root
    return Path(mne_data_root()[1]) / "rfl-cache"


def cache_key(dataset: str, settings: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """(short hash, full description) of the trials of `dataset` under `settings`."""
    import mne
    import moabb
    full = {"format": CACHE_FORMAT, "dataset": dataset, "moabb": moabb.__version__,
            "mne": mne.__version__, **settings}
    blob = json.dumps(full, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16], full


def subject_file(dataset: str, key: str, subject: int) -> Path:
    return cache_root() / dataset / key / f"subject_{int(subject):03d}.npz"


def _put(arrays: Dict[str, np.ndarray], name: str, values) -> None:
    a = np.asarray(values)
    arrays[f"{name}.object"] = np.array(a.dtype == object)
    arrays[name] = a.astype(str) if a.dtype == object else a


def _get(data, name: str) -> np.ndarray:
    a = data[name]
    return a.astype(object) if bool(data[f"{name}.object"]) else a


def save_subject(path: Path, chunks: List[tuple], description: Dict[str, Any]) -> None:
    """Write the (X, y, meta, ch_names) chunks of one subject. Errors are logged only:
    the cache is a speed-up, never a reason for a run to fail."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        params = path.parent / "params.json"
        if not params.exists():
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".params.", suffix=".json")
            with os.fdopen(fd, "w") as f:
                json.dump(description, f, indent=1, sort_keys=True, default=str)
            os.replace(tmp, params)
        arrays: Dict[str, np.ndarray] = {"n_chunks": np.array(len(chunks))}
        for i, (X, y, meta, ch_names) in enumerate(chunks):
            arrays[f"{i}.X"] = np.asarray(X)
            _put(arrays, f"{i}.y", y)
            arrays[f"{i}.ch_names"] = np.asarray(list(ch_names), dtype=str)
            arrays[f"{i}.meta_columns"] = np.asarray([str(c) for c in meta.columns], dtype=str)
            for j, c in enumerate(meta.columns):
                _put(arrays, f"{i}.meta.{j}", meta[c].to_numpy())
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".npz")
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)
    except Exception as err:                                       # noqa: BLE001
        logger.warning(f"could not write the trial cache {path}: {err}")
        try:
            os.remove(tmp)                                       # type: ignore[possibly-undefined]
        except Exception:                                          # noqa: BLE001
            pass


def load_subject(path: Path) -> Optional[List[tuple]]:
    """The chunks written by `save_subject`, or None if the file is missing or unreadable."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            chunks = []
            for i in range(int(data["n_chunks"])):
                cols = [str(c) for c in data[f"{i}.meta_columns"]]
                meta = pd.DataFrame({c: _get(data, f"{i}.meta.{j}")
                                     for j, c in enumerate(cols)}, columns=cols)
                chunks.append((data[f"{i}.X"], _get(data, f"{i}.y"), meta,
                               [str(c) for c in data[f"{i}.ch_names"]]))
            return chunks
    except Exception as err:                                       # noqa: BLE001
        logger.warning(f"unreadable trial cache {path}, recomputed: {err}")
        return None
