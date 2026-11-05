"""Dataset statistics used to check a run before it starts.

Two numbers decide whether a command line can run on a dataset, and both are only
known from the data: the number of EEG channels (the BiMap input dimension) and the
size of the smallest client training set (under DP the batch size must not exceed it).

* The channel count comes from the MOABB metadata of the dataset (an upper bound: the
  loader may keep only the channels common to all subjects), or from the statistics
  file below when it exists.
* The statistics file `<MNE_DATA>/rfl-stats/<dataset>.json` holds the number of
  channels and the number of trials of each subject for each event, after the session
  filter of the registry. It is written by `load_moabb_dataset` at every load (merged
  subject by subject) and by `rfl/experiments/fetch_dataset.py` after a download.

`rfl/experiments/cli_checks.py:dataset_errors` uses both to refuse a run before it loads
any data. Without the file, only the channel check applies.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


def stats_path(dataset: str) -> Path:
    """Path of the statistics file, in the MNE data folder the runs use
    (`rfl.data.registry.mne_data_root`, also when MNE_DATA is not set)."""
    from rfl.data.registry import mne_data_root
    return Path(mne_data_root()[1]) / "rfl-stats" / f"{dataset}.json"


def load_stats(dataset: str) -> Optional[dict]:
    """The statistics of `dataset`, or None if there is no (readable) file."""
    path = stats_path(dataset)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def update_stats(dataset: str, meta, n_channels: int) -> None:
    """Merge the trial counts of the loaded subjects into the statistics file.

    `meta` is the MOABB metadata of the loaded trials (columns 'subject' and 'event').
    The write is atomic (temporary file, then rename), so parallel runs never leave a
    truncated file. Errors are ignored: the file is a convenience for the checks.
    """
    path = stats_path(dataset)
    if meta is None or "event" not in getattr(meta, "columns", ()):
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        old = load_stats(dataset) or {}
        trials: Dict[str, Dict[str, int]] = dict(old.get("trials", {}))
        counts = meta.groupby(["subject", "event"]).size()
        new: Dict[str, Dict[str, int]] = {}
        for (subject, event), n in counts.items():
            new.setdefault(str(int(subject)), {})[str(event)] = int(n)
        trials.update(new)
        prev = old.get("n_channels")
        data = {"dataset": dataset,
                "n_channels": int(n_channels) if prev is None else min(int(prev),
                                                                     int(n_channels)),
                "trials": dict(sorted(trials.items(), key=lambda kv: int(kv[0])))}
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{dataset}.", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)
    except Exception:                                    # noqa: BLE001
        pass


def eeg_channel_count(dataset: str) -> Optional[int]:
    """EEG channels of `dataset`: statistics file, else MOABB metadata, else None."""
    st = load_stats(dataset)
    if st and st.get("n_channels"):
        return int(st["n_channels"])
    try:
        from rfl.data.registry import get_moabb_class
        types = get_moabb_class(dataset)().metadata.acquisition.channel_types
        return int(types["eeg"])
    except Exception:                                    # noqa: BLE001
        return None


def task_events(dataset: str, task: str) -> List[str]:
    """Event names kept by `task` (rfl/data/data_utils.py:filter_binary_task)."""
    from rfl.data.registry import get_events
    events = get_events(dataset)
    if task == "binary":
        return ["left_hand", "right_hand"]
    if task == "no_rest":
        return [e for e in events if e != "rest"]
    return events


def client_trial_counts(dataset: str, subjects: List[int], subjects_per_client: int,
                        task: str) -> Optional[List[int]]:
    """Number of trials of each client for `task`, or None without statistics."""
    st = load_stats(dataset)
    if not st:
        return None
    trials = st.get("trials", {})
    if any(str(s) not in trials for s in subjects):
        return None
    from rfl.federated.fl_config import create_client_subject_mapping
    keep = set(task_events(dataset, task))
    return [sum(c for s in group for e, c in trials[str(s)].items() if e in keep)
            for group in create_client_subject_mapping(list(subjects), subjects_per_client)]


def client_train_sizes(dataset: str, subjects: List[int], subjects_per_client: int,
                       task: str, train_fraction: float) -> Optional[List[int]]:
    """Expected training-set size of each client, or None without statistics.

    The split is stratified over all trials, not per client, so the actual size of a
    client varies around this value from seed to seed (binomial spread).
    """
    st = load_stats(dataset)
    if not st:
        return None
    trials = st.get("trials", {})
    if any(str(s) not in trials for s in subjects):
        return None
    from rfl.federated.fl_config import create_client_subject_mapping
    keep = set(task_events(dataset, task))
    sizes = []
    for group in create_client_subject_mapping(list(subjects), subjects_per_client):
        n = sum(c for s in group for e, c in trials[str(s)].items() if e in keep)
        sizes.append(int(train_fraction * n))
    return sizes
