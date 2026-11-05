"""Metadata of the supported MOABB motor-imagery datasets.

`DATASET_CONFIGS` maps a dataset name to its classes, MOABB event names, subject ids,
native sampling rate (Hz) and, where needed, sessions and epoch window (tmin, tmax in
seconds). The MOABB classes are imported lazily by `get_moabb_class`, so this module
works without moabb installed (the synthetic-data workflow needs no MOABB).

`DATASET_DEFAULTS` holds the SPDNet architecture used for each dataset in the two papers
built on this code (DP-RFedProj, TMLR submission, Table 5, and the ICASSP 2027 paper):
the BiMap output dimension d and the ReEig floor. The floors are the 15th percentile of
the trace-normalised covariance spectra (rfl/experiments/analyze_covariances.py). The
runners use them when --hidden-dim and --epsilon are not given.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# PhysionetMI: MOABB also exposes 'rest'. Only the four balanced imagery classes are
# used, since adding 'rest' makes the class distribution imbalanced. Subjects 88, 92 and
# 100 follow another protocol (128 Hz; runs of 19 cues of 5.1 s with 1.4 s of rest for 88
# and 92, of 12 cues for 100, against 15 cues of 4.1 s elsewhere): they are left out, as
# in the literature and in the papers built on this code.
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "BNCI2014_001": {
        "n_classes": 4,
        "events": ["left_hand", "right_hand", "feet", "tongue"],
        "subjects": list(range(1, 10)),
        "sampling_rate": 250,
    },
    "Weibo2014": {
        "n_classes": 7,
        "events": ["left_hand", "right_hand", "hands", "feet",
                   "left_hand_right_foot", "right_hand_left_foot", "rest"],
        "subjects": list(range(1, 11)),
        "sampling_rate": 200,
    },
    "Cho2017": {
        "n_classes": 2,
        "events": ["left_hand", "right_hand"],
        "subjects": list(range(1, 53)),
        "sampling_rate": 512,
    },
    "Schirrmeister2017": {
        "n_classes": 4,
        "events": ["left_hand", "right_hand", "feet", "rest"],
        "subjects": list(range(1, 15)),
        "sampling_rate": 500,
    },
    "PhysionetMI": {
        "n_classes": 4,
        "events": ["left_hand", "right_hand", "hands", "feet"],
        "subjects": [s for s in range(1, 110) if s not in (88, 92, 100)],
        "sampling_rate": 160,
        "tmin": 0.0,
        "tmax": 3.0,
    },
    "Stieger2021": {
        "n_classes": 4,
        "events": ["left_hand", "right_hand", "both_hand", "rest"],
        "subjects": list(range(1, 63)),
        "sampling_rate": 1000,
        "sessions": [1, 2],
        "tmin": 0.0,
        "tmax": 3.0,
    },
    "Lee2019_MI": {
        "n_classes": 2,
        "events": ["left_hand", "right_hand"],
        "subjects": list(range(1, 55)),
        "sampling_rate": 1000,
        "sessions": None,
        "tmin": 0.0,
        "tmax": 4.0,
    },
}


DATASET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "BNCI2014_001": {"spdnet_hidden_dim": 16, "spdnet_reeig": 8e-5},
    "Weibo2014": {"spdnet_hidden_dim": 20, "spdnet_reeig": 2e-5},
    "Cho2017": {"spdnet_hidden_dim": 16, "spdnet_reeig": 6e-5},
    "Schirrmeister2017": {"spdnet_hidden_dim": 16, "spdnet_reeig": 4e-6},
}


# MNE configuration key that MOABB reads for the download folder of each dataset
# (checked by tests/test_data.py against the installed MOABB). Lee2019 has a hyphen in
# its key, which a shell `export` cannot set: configure_mne_paths sets them in Python.
MNE_PATH_KEYS: Dict[str, str] = {
    "BNCI2014_001": "MNE_DATASETS_BNCI_PATH",
    "Weibo2014": "MNE_DATASETS_WEIBO_PATH",
    "Cho2017": "MNE_DATASETS_GIGADB_PATH",
    "Schirrmeister2017": "MNE_DATASETS_SCHIRRMEISTER2017_PATH",
    "PhysionetMI": "MNE_DATASETS_EEGBCI_PATH",
    "Stieger2021": "MNE_DATASETS_STIEGER2021_PATH",
    "Lee2019_MI": "MNE_DATASETS_LEE2019-MI_PATH",
}


def mne_data_root(data_dir: Optional[str] = None) -> Tuple[str, str]:
    """(folder as given, its real path) of the MNE data folder, without side effects.

    `data_dir`, else the environment variable MNE_DATA, else MNE_DATA in the MNE
    configuration file, else ~/mne_data. The rule of `configure_mne_paths`.
    """
    import os
    from pathlib import Path

    d = data_dir or os.environ.get("MNE_DATA")
    if not d:
        try:
            import mne
            d = mne.get_config("MNE_DATA", use_env=False)
        except ImportError:
            d = None
    d = d or str(Path.home() / "mne_data")
    return d, os.path.realpath(os.path.expanduser(d))


_ANNOUNCED: set = set()


def configure_mne_paths(data_dir: Optional[str] = None, verbose: bool = True) -> str:
    """Point MNE and MOABB at one data folder, for this process, and return it.

    The folder is `data_dir`, else the environment variable MNE_DATA, else MNE_DATA in
    the MNE configuration file (~/.mne/mne-python.json), else ~/mne_data. It is
    resolved to its real path (symbolic links followed) and created if missing. MNE_DATA
    and every key of MNE_PATH_KEYS are then set in the environment of this process.
    Environment variables take precedence over the configuration file, so entries of
    that file pointing elsewhere (for example an old symbolic link) are ignored. They
    are listed once per process, and the file is never written.
    """
    import os

    import mne

    d, real = mne_data_root(data_dir)
    os.makedirs(real, exist_ok=True)
    if not os.access(real, os.W_OK):
        raise PermissionError(f"MNE data folder {real} is not writable")
    keys = ["MNE_DATA", *MNE_PATH_KEYS.values()]
    stale = {}
    for k in keys:
        v = mne.get_config(k, use_env=False)
        if v and os.path.realpath(os.path.expanduser(v)) != real:
            stale[k] = v
        os.environ[k] = real
    if verbose and real not in _ANNOUNCED:
        _ANNOUNCED.add(real)
        link = f" (from {d})" if os.path.abspath(os.path.expanduser(d)) != real else ""
        logger.info(f"MNE data folder: {real}{link}")
        if stale:
            logger.info("  ignored entries of the MNE configuration file "
                  f"({mne.get_config_path()}):")
            for k, v in stale.items():
                logger.info(f"    {k} = {v}")
    return real


def dataset_default(name: str, key: str, default: Any = None) -> Any:
    """Default setting `key` for dataset `name`, or `default` if there is none."""
    return DATASET_DEFAULTS.get(name, {}).get(key, default)


def is_known_dataset(name: str) -> bool:
    """Whether `name` is in the registry."""
    return name in DATASET_CONFIGS


def get_config(name: str) -> Dict[str, Any]:
    """Registry entry of `name`. Raises KeyError for an unknown name."""
    if name not in DATASET_CONFIGS:
        raise KeyError(
            f"Unknown dataset '{name}'. Known: {sorted(DATASET_CONFIGS)}. "
            "(Synthetic datasets carry their own metadata and are not in the registry.)"
        )
    return DATASET_CONFIGS[name]


def get_sampling_rate(name: str, default: Optional[float] = None) -> Optional[float]:
    """Native sampling rate (Hz), or `default` for unknown (for example synthetic) names."""
    cfg = DATASET_CONFIGS.get(name)
    return cfg["sampling_rate"] if cfg else default


def get_events(name: str) -> List[str]:
    """MOABB event names used for `name`."""
    return list(get_config(name)["events"])


def get_subjects(name: str) -> List[int]:
    """Subject ids used for `name`."""
    return list(get_config(name)["subjects"])


def get_moabb_class(name: str):
    """Lazily import and return the MOABB dataset class (requires moabb)."""
    import moabb.datasets as mds
    return getattr(mds, name)


