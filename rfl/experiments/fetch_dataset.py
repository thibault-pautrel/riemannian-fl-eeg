#!/usr/bin/env python3
# file: rfl/experiments/fetch_dataset.py
"""
Download a MOABB dataset subject by subject, with retries, before running experiments.

Downloading once up front avoids many parallel runs fetching the same files at the
same time, which can overload the data servers (HTTP 502) and corrupt files. MNE checks
the files already present against their published hash and downloads them again when
they are incomplete. Rerun the script until it reports no missing subject.

Files go to one folder: --data-dir, else MNE_DATA, else ~/mne_data (see
rfl/data/registry.py:configure_mne_paths, the same rule as the runners). The MNE
configuration file is read but never written.

Usage (from any folder once the repository is installed, see README):
    python -m rfl.experiments.fetch_dataset --dataset BNCI2014_001
    python -m rfl.experiments.fetch_dataset --dataset PhysionetMI --data-dir /path/to/mne_data

Requirements: MOABB and network access.
"""
import argparse
import sys
import time


from rfl.data.registry import DATASET_CONFIGS

def main() -> int:
    """Download every subject of one dataset. Returns 1 if a subject is still missing."""
    from rfl.utils.logs import setup_logging
    setup_logging()
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS),
                   help="A dataset of rfl/data/registry.py.")
    p.add_argument("--data-dir", default=None,
                   help="Set the MNE data variables to this directory for this run.")
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--wait", type=float, default=30.0, help="seconds between retries")
    a = p.parse_args()

    # Before MOABB reads them: MNE_DATA and the per-dataset keys (rfl/data/registry.py).
    from rfl.data.registry import configure_mne_paths, get_subjects
    configure_mne_paths(a.data_dir)

    # Each subject is loaded once, as the runners load it: recent MOABB versions resolve
    # the files lazily (data_path alone may download nothing), so only a load proves that
    # the data are on disk. The load also records the trial counts (rfl/data/stats.py)
    # and writes the trials to the cache the jobs read (rfl/data/cache.py). A subject
    # already cached is not read again: RFL_DATA_CACHE=refresh rereads the raw files.
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)    # MNE montage notices
    from rfl.data.data_utils import load_moabb_dataset, resolve_resample
    from rfl.data.registry import get_sampling_rate
    from rfl.data.stats import stats_path
    # The rate of the runs (--resample 128 by default), so the trial cache written here
    # is the one the runs read (rfl/data/cache.py).
    rate = resolve_resample(128.0, get_sampling_rate(a.dataset))
    missing = []
    for s in get_subjects(a.dataset):
        for attempt in range(1, a.retries + 1):
            try:
                load_moabb_dataset(a.dataset, [s], resample=rate, verbose=False)
                print(f"subject {s:3d}: ok (on disk, trials counted)", flush=True)
                break
            except Exception as e:                      # noqa: BLE001
                print(f"subject {s:3d}: attempt {attempt} failed ({type(e).__name__}: "
                      f"{str(e)[:100]})", flush=True)
                time.sleep(a.wait)
        else:
            missing.append(s)
    from rfl.data.cache import cache_mode, cache_root
    print(f"\n{a.dataset}: {len(missing)} subject(s) missing" +
          (f": {missing}" if missing else f". Statistics: {stats_path(a.dataset)}"))
    if cache_mode() != "off" and not missing:
        print(f"Trials of the runs cached in {cache_root() / a.dataset}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
